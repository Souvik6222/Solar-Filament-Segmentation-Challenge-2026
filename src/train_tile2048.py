"""Native-2048px 1024x1024 sliding-tile trainer + 2048px full-res probability map exporter."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import data, export, metrics, paths, train_seg

CACHE_1024 = paths.CACHE_1024
CACHE_2048 = paths.CACHE_2048


class Tile2048Dataset(Dataset):
    def __init__(self, stems: Sequence[str], fg_centers: Dict[str, np.ndarray], crop_size: int = 1024):
        self.stems = list(stems)
        self.fg_centers = fg_centers
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        soft = cv2.imread(str(CACHE_2048 / "soft" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        spine = cv2.imread(str(CACHE_2048 / "spine" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        H, W, _ = img.shape
        cs = self.crop_size
        pts = self.fg_centers.get(stem)
        if pts is not None and len(pts) > 0 and np.random.rand() < 0.80:
            py, px = pts[np.random.randint(len(pts))]
            cy = int(py) + np.random.randint(-cs // 3, cs // 3 + 1)
            cx = int(px) + np.random.randint(-cs // 3, cs // 3 + 1)
            y0 = max(0, min(H - cs, cy - cs // 2))
            x0 = max(0, min(W - cs, cx - cs // 2))
        else:
            y0 = np.random.randint(0, H - cs + 1)
            x0 = np.random.randint(0, W - cs + 1)

        img_c = np.transpose(img[y0 : y0 + cs, x0 : x0 + cs], (2, 0, 1)).astype(np.float32) / 255.0
        soft_c = soft[y0 : y0 + cs, x0 : x0 + cs].astype(np.float32) / 255.0
        spine_c = spine[y0 : y0 + cs, x0 : x0 + cs].astype(np.float32) / 255.0
        return (
            stem,
            torch.from_numpy(img_c),
            torch.from_numpy(soft_c).unsqueeze(0),
            torch.from_numpy(spine_c).unsqueeze(0),
        )


def _hann2d(size: int = 1024) -> torch.Tensor:
    w1d = np.hanning(size + 2)[1:-1].astype(np.float32)
    # Raise floor slightly inside central 512px so interior weights dominate cleanly
    w1d = np.clip(w1d, 0.05, 1.0)
    w2d = np.outer(w1d, w1d)
    return torch.from_numpy(w2d)


def disk_mask_2048(geom_1024: Sequence[float], margin: float = 1.005) -> np.ndarray:
    cx, cy, r = geom_1024[0] * 2.0, geom_1024[1] * 2.0, geom_1024[2] * 2.0
    yy, xx = np.ogrid[:2048, :2048]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * margin) ** 2


@torch.no_grad()
def predict_tiled_2048(
    model: nn.Module,
    img_2048_hwc: np.ndarray,
    device: torch.device,
    n_tta: int = 2,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """3x3 overlapping 1024x1024 sliding-window prediction at native 2048x2048 with optional D4 TTA."""
    offsets = (0, 512, 1024)
    win = _hann2d(1024).to(device)
    img_chw = torch.from_numpy(np.transpose(img_2048_hwc, (2, 0, 1)).astype(np.float32) / 255.0)
    tta_keys = [0, 4, 1, 5, 2, 6, 3, 7][:n_tta]

    p_sum = torch.zeros((2048, 2048), device=device, dtype=torch.float32)
    s_sum = torch.zeros((2048, 2048), device=device, dtype=torch.float32)
    w_sum = torch.zeros((2048, 2048), device=device, dtype=torch.float32)

    for y0 in offsets:
        for x0 in offsets:
            tile = img_chw[:, y0 : y0 + 1024, x0 : x0 + 1024].to(device, non_blocking=True)
            tp = torch.zeros((1024, 1024), device=device, dtype=torch.float32)
            ts = torch.zeros((1024, 1024), device=device, dtype=torch.float32)
            for k in tta_keys:
                xk = train_seg.apply_d4(tile, k).unsqueeze(0)
                if channels_last:
                    xk = xk.to(memory_format=torch.channels_last)
                with torch.amp.autocast(xk.device.type, dtype=amp_dtype, enabled=(xk.device.type == "cuda")):
                    logits = model(train_seg.normalize_imagenet(xk))
                    probs = torch.sigmoid(logits.float())
                tp += train_seg.invert_d4(probs[0, 0], k)
                ts += train_seg.invert_d4(probs[0, 1], k)
            tp /= len(tta_keys)
            ts /= len(tta_keys)
            p_sum[y0 : y0 + 1024, x0 : x0 + 1024] += tp * win
            s_sum[y0 : y0 + 1024, x0 : x0 + 1024] += ts * win
            w_sum[y0 : y0 + 1024, x0 : x0 + 1024] += win

    p2048 = (p_sum / w_sum).cpu().numpy().astype(np.float32)
    s2048 = (s_sum / w_sum).cpu().numpy().astype(np.float32)
    return p2048, s2048


def eval_2048_pq(
    probs_2048: Dict[str, np.ndarray],
    gt_recs: Dict[str, List[dict]],
    rec_to_stem: Dict[str, str],
    thresholds: Sequence[float] = (0.40, 0.46),
    min_area: int = 200,
) -> Tuple[float, float]:
    best_pq = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        pred_by_stem = {}
        for stem, p in probs_2048.items():
            bw = (p >= thr).astype(np.uint8)
            n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            crops = []
            for k in range(1, n_lbl):
                if int(stats[k, cv2.CC_STAT_AREA]) < min_area:
                    continue
                x1 = int(stats[k, cv2.CC_STAT_LEFT])
                y1 = int(stats[k, cv2.CC_STAT_TOP])
                x2 = x1 + int(stats[k, cv2.CC_STAT_WIDTH])
                y2 = y1 + int(stats[k, cv2.CC_STAT_HEIGHT])
                sub = lbl[y1:y2, x1:x2] == k
                crops.append((x1, y1, x2, y2, sub, float(p[y1:y2, x1:x2][sub].mean())))
            pred_by_stem[stem] = export.encode_disjoint_crops_2048(crops, min_area=min_area, top_k=40)
        res, _ = metrics.evaluate_pq(gt_recs, rec_to_stem, pred_by_stem)
        if res.pq > best_pq:
            best_pq = res.pq
            best_thr = thr
    return best_pq, best_thr


def save_u16_png(path: Path, arr_f32: np.ndarray) -> None:
    u16 = np.clip(np.round(arr_f32 * 65535.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), u16, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def train_fold(
    fold: int,
    arch: str,
    encoder: str,
    run_dir: Path,
    epochs: int = 24,
    batch_size: int = 4,
    grad_accum: int = 1,
    lr: float = 4e-4,
    eval_every: int = 3,
    n_tta_val: int = 2,
    channels_last: bool = True,
    seed: int = 100,
) -> float:
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    meta = json.loads((CACHE_1024 / "meta.json").read_text())
    folds_map = meta["folds"]
    geom_map = meta["geom_1024"]
    train_stems = sorted(s for s in meta["train_stems"] if folds_map[s] != fold)
    val_stems = sorted(s for s in meta["train_stems"] if folds_map[s] == fold)
    test_stems = sorted(meta["test_stems"])

    coco = data.load_coco_grouped()
    val_gt_recs = {r: coco["gt_rles_2048"][r] for s in val_stems for r in coco["recs_by_stem"][s]}

    # Precompute foreground pixel coords at 2048x2048 scale from fast 1024 soft cache
    fg_centers: Dict[str, np.ndarray] = {}
    for s in train_stems:
        sf = np.load(CACHE_1024 / "soft" / f"{s}.npy")
        ys, xs = np.nonzero(sf >= 128)
        fg_centers[s] = (np.stack([ys, xs], axis=1) * 2).astype(np.int16)

    tr_ds = Tile2048Dataset(train_stems, fg_centers=fg_centers, crop_size=1024)
    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True
    )

    model = train_seg.build_model(arch=arch, encoder=encoder, pretrained=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = (len(tr_loader) // grad_accum) * epochs
    warmup_steps = max(len(tr_loader) // grad_accum, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best.pt"
    best_pq = -1.0
    best_ep = -1

    print(f"[Tile2048 {arch}:{encoder}] Fold {fold}: tr={len(train_stems)}, va={len(val_stems)}, ep={epochs}", flush=True)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_b = 0
        optimizer.zero_grad(set_to_none=True)
        for b_idx, (_, img_t, soft_t, spine_t) in enumerate(tr_loader):
            img_t = img_t.to(device, non_blocking=True)
            soft_t = soft_t.to(device, non_blocking=True)
            spine_t = spine_t.to(device, non_blocking=True)
            img_t, soft_t, spine_t = train_seg.gpu_augment_batch(img_t, soft_t, spine_t, do_affine=True)
            if channels_last:
                img_t = img_t.to(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(train_seg.normalize_imagenet(img_t))
                loss = train_seg.compute_loss(logits, soft_t, spine_t) / grad_accum
            loss.backward()
            loss_sum += float(loss.item()) * grad_accum
            n_b += 1
            if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        msg = f"  ep {ep:02d}/{epochs} | loss={loss_sum/max(n_b,1):.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s"
        if ep == epochs or (ep >= 6 and ep % eval_every == 0):
            model.eval()
            t_ev = time.time()
            val_probs = {}
            for stem in val_stems:
                hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
                p, _ = predict_tiled_2048(model, hwc, device, n_tta=1, amp_dtype=amp_dtype, channels_last=channels_last)
                p *= disk_mask_2048(geom_map[stem], margin=1.005)
                val_probs[stem] = p
            pq, tb = eval_2048_pq(val_probs, val_gt_recs, coco["rec_to_stem"])
            msg += f" | val_pq={pq:.4f} (@{tb:.2f}, {time.time()-t_ev:.1f}s)"
            if pq > best_pq:
                best_pq = pq
                best_ep = ep
                torch.save({"model": model.state_dict(), "epoch": ep, "pq": pq}, ckpt_path)
                msg += " *BEST*"
        print(msg, flush=True)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    (run_dir / "oof_prob_2048").mkdir(parents=True, exist_ok=True)
    (run_dir / "oof_prob").mkdir(parents=True, exist_ok=True)
    (run_dir / "oof_spine").mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_prob_2048").mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_prob").mkdir(parents=True, exist_ok=True)

    oof_p = {}
    for stem in val_stems:
        hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        p, s = predict_tiled_2048(model, hwc, device, n_tta=n_tta_val, amp_dtype=amp_dtype, channels_last=channels_last)
        dm = disk_mask_2048(geom_map[stem], margin=1.005)
        p *= dm
        s *= dm
        oof_p[stem] = p
        save_u16_png(run_dir / "oof_prob_2048" / f"{stem}.png", p)
        np.save(run_dir / "oof_prob" / f"{stem}.npy", cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
        np.save(run_dir / "oof_spine" / f"{stem}.npy", cv2.resize(s, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))

    tta_pq, tb = eval_2048_pq(oof_p, val_gt_recs, coco["rec_to_stem"], thresholds=(0.36, 0.40, 0.44, 0.48, 0.52))
    print(f"Tile2048 Fold {fold} Best (ep {best_ep}) {n_tta_val}x-TTA Val PQ = {tta_pq:.4f} (@{tb:.2f})", flush=True)

    for stem in test_stems:
        hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        p, _ = predict_tiled_2048(model, hwc, device, n_tta=n_tta_val, amp_dtype=amp_dtype, channels_last=channels_last)
        p *= disk_mask_2048(geom_map[stem], margin=1.005)
        save_u16_png(fold_dir / "test_prob_2048" / f"{stem}.png", p)
        np.save(fold_dir / "test_prob" / f"{stem}.npy", cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
    return tta_pq


def infer_test_fold(
    fold: int,
    arch: str,
    encoder: str,
    run_dir: Path,
    n_tta_val: int = 2,
    channels_last: bool = False,
) -> None:
    """Export 2048px and 1024px test probability maps for one fold checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    meta = json.loads((CACHE_1024 / "meta.json").read_text())
    geom_map = meta["geom_1024"]
    test_stems = sorted(meta["test_stems"])

    fold_dir = run_dir / f"fold{fold}"
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")

    model = train_seg.build_model(arch=arch, encoder=encoder, pretrained=False).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    (fold_dir / "test_prob_2048").mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_prob").mkdir(parents=True, exist_ok=True)
    for stem in test_stems:
        hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        p, _ = predict_tiled_2048(model, hwc, device, n_tta=n_tta_val, amp_dtype=amp_dtype, channels_last=channels_last)
        p *= disk_mask_2048(geom_map[stem], margin=1.005)
        save_u16_png(fold_dir / "test_prob_2048" / f"{stem}.png", p)
        np.save(fold_dir / "test_prob" / f"{stem}.npy", cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
    print(f"[{run_dir.name}] Fold {fold} exported {len(test_stems)} 2048px test maps ({n_tta_val}x-TTA)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--run-name", default="tile2048_unet_r34")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--n-tta", type=int, default=2)
    ap.add_argument("--infer-test", action="store_true")
    args = ap.parse_args()

    run_dir = paths.RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_list = [int(x) for x in args.folds.split(",") if x.strip()]
    if args.infer_test:
        for f in fold_list:
            infer_test_fold(fold=f, arch=args.arch, encoder=args.encoder, run_dir=run_dir, n_tta_val=args.n_tta)
        return
    scores = {}
    for f in fold_list:
        scores[f] = train_fold(
            fold=f, arch=args.arch, encoder=args.encoder, run_dir=run_dir,
            epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
            lr=args.lr, eval_every=args.eval_every, n_tta_val=args.n_tta,
        )
    print(f"Summary [{args.run_name}] fold PQs: {scores}", flush=True)


if __name__ == "__main__":
    main()
