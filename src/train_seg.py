"""Fast multi-task segmentation trainer (soft consensus + spine head) with GPU augmentations & D4 TTA."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import data, export, metrics, paths

CACHE_DIR = paths.CACHE_1024


def apply_d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply dihedral D4 transform index 0..7 on (..., H, W) tensor."""
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
        k -= 4
    if k > 0:
        x = torch.rot90(x, k=k, dims=(-2, -1))
    return x


def invert_d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """Invert dihedral D4 transform index 0..7 on (..., H, W) tensor."""
    flipped = k >= 4
    rot_k = k - 4 if flipped else k
    if rot_k > 0:
        x = torch.rot90(x, k=(4 - rot_k) % 4, dims=(-2, -1))
    if flipped:
        x = torch.flip(x, dims=(-1,))
    return x


class FilamentCacheDataset(Dataset):
    def __init__(self, stems: Sequence[str], is_train: bool = True):
        self.stems = list(stems)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = np.load(CACHE_DIR / "img" / f"{stem}.npy")
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0)
        if not self.is_train:
            return stem, img_t
        soft = np.load(CACHE_DIR / "soft" / f"{stem}.npy")
        spine = np.load(CACHE_DIR / "spine" / f"{stem}.npy")
        soft_t = torch.from_numpy(soft.astype(np.float32) / 255.0).unsqueeze(0)
        spine_t = torch.from_numpy(spine.astype(np.float32) / 255.0).unsqueeze(0)
        return stem, img_t, soft_t, spine_t


def gpu_augment_batch(
    img: torch.Tensor, soft: torch.Tensor, spine: torch.Tensor, do_affine: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast GPU D4 + optional continuous affine rotation/scale + photometric jitter."""
    b, _, h, w = img.shape
    device = img.device

    # Per-sample D4 transform
    out_img, out_soft, out_spine = [], [], []
    for i in range(b):
        k = int(torch.randint(0, 8, (1,)).item())
        out_img.append(apply_d4(img[i], k))
        out_soft.append(apply_d4(soft[i], k))
        out_spine.append(apply_d4(spine[i], k))
    img = torch.stack(out_img, dim=0)
    soft = torch.stack(out_soft, dim=0)
    spine = torch.stack(out_spine, dim=0)

    if do_affine:
        angles = (torch.rand(b, device=device) * 30.0 - 15.0) * (math.pi / 180.0)
        scales = torch.rand(b, device=device) * 0.14 + 0.93  # [0.93, 1.07]
        tx = torch.rand(b, device=device) * 0.04 - 0.02
        ty = torch.rand(b, device=device) * 0.04 - 0.02
        cos_a = torch.cos(angles) / scales
        sin_a = torch.sin(angles) / scales
        theta = torch.stack(
            [
                torch.stack([cos_a, -sin_a, tx], dim=1),
                torch.stack([sin_a, cos_a, ty], dim=1),
            ],
            dim=1,
        )
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
        soft = F.grid_sample(soft, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        spine = F.grid_sample(spine, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Mild photometric contrast/brightness/gamma jitter preserving mean ~0.5
    if torch.rand(1).item() < 0.8:
        contrast = (torch.rand(b, 3, 1, 1, device=device) * 0.24 + 0.88)
        bright = (torch.rand(b, 3, 1, 1, device=device) * 0.10 - 0.05)
        img = torch.clamp((img - 0.5) * contrast + 0.5 + bright, 0.0, 1.0)
    if torch.rand(1).item() < 0.35:
        noise = torch.randn_like(img) * 0.015
        img = torch.clamp(img + noise, 0.0, 1.0)

    return img, torch.clamp(soft, 0.0, 1.0), torch.clamp(spine, 0.0, 1.0)


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def compute_loss(logits: torch.Tensor, soft: torch.Tensor, spine: torch.Tensor) -> torch.Tensor:
    lm = logits[:, 0:1].float()
    ls = logits[:, 1:2].float()
    pm = torch.sigmoid(lm)
    ps = torch.sigmoid(ls)

    bce_m = F.binary_cross_entropy_with_logits(lm, soft)
    dice_m = soft_dice_loss(pm, soft)
    loss_m = 0.65 * bce_m + 0.35 * dice_m

    pos_w = torch.tensor([5.0], device=logits.device, dtype=torch.float32)
    bce_s = F.binary_cross_entropy_with_logits(ls, spine, pos_weight=pos_w)
    dice_s = soft_dice_loss(ps, spine)
    loss_s = 0.5 * bce_s + 0.5 * dice_s

    return loss_m + 0.25 * loss_s


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def build_model(arch: str, encoder: str, pretrained: bool = True) -> nn.Module:
    weights = "imagenet" if pretrained else None
    return smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=2,
    )


def disk_mask_1024(geom: Sequence[float], margin: float = 1.005) -> np.ndarray:
    cx, cy, r = geom
    yy, xx = np.ogrid[:1024, :1024]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * margin) ** 2


@torch.no_grad()
def predict_d4_tta(
    model: nn.Module,
    img_t: torch.Tensor,
    n_tta: int = 8,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict calibrated probability and spine maps with D4 TTA (n_tta in {1,2,4,8})."""
    tta_keys = [0, 4, 1, 5, 2, 6, 3, 7][:n_tta]
    p_acc = torch.zeros((1, 1024, 1024), device=img_t.device, dtype=torch.float32)
    s_acc = torch.zeros((1, 1024, 1024), device=img_t.device, dtype=torch.float32)
    for k in tta_keys:
        xk = apply_d4(img_t, k).unsqueeze(0)
        if channels_last:
            xk = xk.to(memory_format=torch.channels_last)
        with torch.amp.autocast(xk.device.type, dtype=amp_dtype, enabled=(xk.device.type == "cuda")):
            logits = model(normalize_imagenet(xk))
            probs = torch.sigmoid(logits.float())
        p_inv = invert_d4(probs[0, 0], k)
        s_inv = invert_d4(probs[0, 1], k)
        p_acc[0] += p_inv
        s_acc[0] += s_inv
    p_np = (p_acc[0] / len(tta_keys)).cpu().numpy().astype(np.float32)
    s_np = (s_acc[0] / len(tta_keys)).cpu().numpy().astype(np.float32)
    return p_np, s_np


def fast_pq_eval(
    val_probs: Dict[str, np.ndarray],
    gt_recs: Dict[str, List[dict]],
    rec_to_stem: Dict[str, str],
    thresholds: Sequence[float] = (0.40, 0.46),
    min_area_2048: int = 200,
) -> Tuple[float, float]:
    """Evaluate exact 2048px connected-component PQ across thresholds (bilinear prob upscaling)."""
    best_pq = -1.0
    best_thr = thresholds[0]
    k3 = np.ones((3, 3), dtype=np.uint8)
    for thr in thresholds:
        pred_by_stem: Dict[str, List[dict]] = {}
        for stem, p1024 in val_probs.items():
            bw = (p1024 >= thr).astype(np.uint8)
            n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            cands = []
            for k in range(1, n_lbl):
                area1024 = int(stats[k, cv2.CC_STAT_AREA])
                if area1024 * 4 < min_area_2048:
                    continue
                x1 = int(stats[k, cv2.CC_STAT_LEFT])
                y1 = int(stats[k, cv2.CC_STAT_TOP])
                x2 = x1 + int(stats[k, cv2.CC_STAT_WIDTH])
                y2 = y1 + int(stats[k, cv2.CC_STAT_HEIGHT])
                score = float(p1024[y1:y2, x1:x2][lbl[y1:y2, x1:x2] == k].mean())
                cands.append((score, k, max(0, x1 - 2), max(0, y1 - 2), min(1024, x2 + 2), min(1024, y2 + 2)))
            cands.sort(key=lambda t: t[0], reverse=True)
            crops = []
            for score, k, px1, py1, px2, py2 in cands[:40]:
                anchor = cv2.dilate((lbl[py1:py2, px1:px2] == k).astype(np.uint8), k3)
                p_hi = cv2.resize(p1024[py1:py2, px1:px2], ((px2 - px1) * 2, (py2 - py1) * 2), interpolation=cv2.INTER_LINEAR)
                a_hi = cv2.resize(anchor, ((px2 - px1) * 2, (py2 - py1) * 2), interpolation=cv2.INTER_NEAREST) > 0
                crops.append((px1 * 2, py1 * 2, px2 * 2, py2 * 2, (p_hi >= thr) & a_hi, score))
            pred_by_stem[stem] = export.encode_disjoint_crops_2048(crops, min_area=min_area_2048, top_k=40)
        res, _ = metrics.evaluate_pq(gt_recs, rec_to_stem, pred_by_stem)
        if res.pq > best_pq:
            best_pq = res.pq
            best_thr = thr
    return best_pq, best_thr


def train_fold(
    fold: int,
    arch: str,
    encoder: str,
    run_dir: Path,
    epochs: int = 24,
    batch_size: int = 2,
    grad_accum: int = 2,
    lr: float = 3e-4,
    eval_every: int = 2,
    n_tta_val: int = 8,
    channels_last: bool = False,
    seed: int = 42,
) -> float:
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds_map: Dict[str, int] = meta["folds"]
    geom_map: Dict[str, List[float]] = meta["geom_1024"]
    train_stems = sorted(s for s in meta["train_stems"] if folds_map[s] != fold)
    val_stems = sorted(s for s in meta["train_stems"] if folds_map[s] == fold)
    test_stems = sorted(meta["test_stems"])

    coco = data.load_coco_grouped()
    val_gt_recs = {r: coco["gt_rles_2048"][r] for s in val_stems for r in coco["recs_by_stem"][s]}

    tr_ds = FilamentCacheDataset(train_stems, is_train=True)
    va_ds = FilamentCacheDataset(val_stems, is_train=False)
    te_ds = FilamentCacheDataset(test_stems, is_train=False)

    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
    )

    model = build_model(arch=arch, encoder=encoder, pretrained=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = (len(tr_loader) // grad_accum) * epochs
    warmup_steps = max(len(tr_loader) // grad_accum, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best.pt"

    best_pq = -1.0
    best_epoch = -1
    print(f"[{arch}:{encoder}] Fold {fold}: tr={len(train_stems)}, va={len(val_stems)}, epochs={epochs}, bs={batch_size}x{grad_accum}")

    global_step = 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for b_idx, (_, img_t, soft_t, spine_t) in enumerate(tr_loader):
            img_t = img_t.to(device, non_blocking=True)
            soft_t = soft_t.to(device, non_blocking=True)
            spine_t = spine_t.to(device, non_blocking=True)
            img_t, soft_t, spine_t = gpu_augment_batch(img_t, soft_t, spine_t, do_affine=True)
            if channels_last:
                img_t = img_t.to(memory_format=torch.channels_last)

            with torch.amp.autocast(xk.device.type, dtype=amp_dtype, enabled=(xk.device.type == "cuda")):
                logits = model(normalize_imagenet(img_t))
                loss = compute_loss(logits, soft_t, spine_t) / grad_accum
            loss.backward()
            loss_sum += float(loss.item()) * grad_accum
            n_batches += 1

            if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        msg = f"  ep {ep:02d}/{epochs} | loss={loss_sum/max(n_batches,1):.4f} | lr={cur_lr:.2e} | {dt:.1f}s"

        if ep == epochs or (ep >= 4 and ep % eval_every == 0):
            model.eval()
            t_ev = time.time()
            val_probs = {}
            for stem, img_t in va_ds:
                p, _ = predict_d4_tta(model, img_t.to(device), n_tta=2, amp_dtype=amp_dtype, channels_last=channels_last)
                p *= disk_mask_1024(geom_map[stem], margin=1.005)
                val_probs[stem] = p
            pq, thr_b = fast_pq_eval(val_probs, val_gt_recs, coco["rec_to_stem"])
            msg += f" | val_pq={pq:.4f} (@{thr_b:.2f}, {time.time()-t_ev:.1f}s)"
            if pq > best_pq:
                best_pq = pq
                best_epoch = ep
                torch.save({"model": model.state_dict(), "epoch": ep, "pq": pq}, ckpt_path)
                msg += " *BEST*"
        print(msg, flush=True)

    # Load best checkpoint and export full 8x D4 TTA OOF + Test probability maps
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    (run_dir / "oof_prob").mkdir(parents=True, exist_ok=True)
    (run_dir / "oof_spine").mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_prob").mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_spine").mkdir(parents=True, exist_ok=True)

    oof_probs = {}
    for stem, img_t in va_ds:
        p, s = predict_d4_tta(model, img_t.to(device), n_tta=n_tta_val, amp_dtype=amp_dtype, channels_last=channels_last)
        dm = disk_mask_1024(geom_map[stem], margin=1.005)
        p *= dm
        s *= dm
        oof_probs[stem] = p
        np.save(run_dir / "oof_prob" / f"{stem}.npy", p.astype(np.float16))
        np.save(run_dir / "oof_spine" / f"{stem}.npy", s.astype(np.float16))

    tta_pq, tta_thr = fast_pq_eval(oof_probs, val_gt_recs, coco["rec_to_stem"], thresholds=(0.36, 0.40, 0.44, 0.48, 0.52))
    print(f"Fold {fold} Best (ep {best_epoch}) 8x-TTA Val PQ = {tta_pq:.4f} (@{tta_thr:.2f})", flush=True)

    for stem, img_t in te_ds:
        p, s = predict_d4_tta(model, img_t.to(device), n_tta=n_tta_val, amp_dtype=amp_dtype, channels_last=channels_last)
        dm = disk_mask_1024(geom_map[stem], margin=1.005)
        np.save(fold_dir / "test_prob" / f"{stem}.npy", (p * dm).astype(np.float16))
        np.save(fold_dir / "test_spine" / f"{stem}.npy", (s * dm).astype(np.float16))

    return tta_pq


def infer_test_fold(
    fold: int,
    arch: str,
    encoder: str,
    run_dir: Path,
    n_tta_val: int = 8,
    channels_last: bool = False,
) -> None:
    """Run saved checkpoint test-set D4-TTA inference for one fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    geom_map: Dict[str, List[float]] = meta["geom_1024"]
    test_stems = sorted(meta["test_stems"])
    te_ds = FilamentCacheDataset(test_stems, is_train=False)

    fold_dir = run_dir / f"fold{fold}"
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")

    model = build_model(arch=arch, encoder=encoder, pretrained=False).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    (fold_dir / "test_prob").mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_spine").mkdir(parents=True, exist_ok=True)
    for stem, img_t in te_ds:
        p, s = predict_d4_tta(model, img_t.to(device), n_tta=n_tta_val, amp_dtype=amp_dtype, channels_last=channels_last)
        dm = disk_mask_1024(geom_map[stem], margin=1.005)
        np.save(fold_dir / "test_prob" / f"{stem}.npy", (p * dm).astype(np.float16))
        np.save(fold_dir / "test_spine" / f"{stem}.npy", (s * dm).astype(np.float16))
    print(f"[{run_dir.name}] Fold {fold} exported {len(test_stems)} test maps ({n_tta_val}x-TTA)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="UnetPlusPlus")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--run-name", default="unetpp_r34")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--n-tta", type=int, default=8)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--infer-test", action="store_true", help="Skip training and export test probs from fold*_best checkpoints")
    args = ap.parse_args()

    run_dir = paths.RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_list = [int(x) for x in args.folds.split(",") if x.strip()]
    if args.infer_test:
        for f in fold_list:
            infer_test_fold(
                fold=f,
                arch=args.arch,
                encoder=args.encoder,
                run_dir=run_dir,
                n_tta_val=args.n_tta,
                channels_last=args.channels_last,
            )
        return
    scores = {}
    for f in fold_list:
        scores[f] = train_fold(
            fold=f,
            arch=args.arch,
            encoder=args.encoder,
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            eval_every=args.eval_every,
            n_tta_val=args.n_tta,
            channels_last=args.channels_last,
        )
    print(f"Summary [{args.run_name}] fold PQs: {scores}", flush=True)


if __name__ == "__main__":
    main()
