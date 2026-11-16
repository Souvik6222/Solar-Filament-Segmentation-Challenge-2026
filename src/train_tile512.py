"""Native-2048px 512x512 sliding-tile trainer (with optional 2x zoom & Lovasz fine-tuning)."""

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

from src import data, metrics, paths, train_seg, train_tile2048

CACHE_1024 = paths.CACHE_1024
CACHE_2048 = paths.CACHE_2048


class Tile512Dataset(Dataset):
    """Samples random 512x512 native (or 2x-zoomed 256x256->512) crops with foreground bias."""

    def __init__(
        self,
        stems: Sequence[str],
        fg_centers: Dict[str, np.ndarray],
        crop_size: int = 512,
        zoom: int = 1,
        crops_per_read: int = 2,
        fg_prob: float = 0.78,
    ):
        self.stems = list(stems)
        self.fg_centers = fg_centers
        self.crop_size = crop_size
        self.zoom = zoom
        self.native_crop = crop_size // zoom
        self.crops_per_read = crops_per_read
        self.fg_prob = fg_prob

    def __len__(self) -> int:
        return len(self.stems)

    def _sample_window(
        self, img: np.ndarray, soft: np.ndarray, spine: np.ndarray, pts: np.ndarray | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W, _ = img.shape
        nc = self.native_crop
        if pts is not None and len(pts) > 0 and np.random.rand() < self.fg_prob:
            py, px = pts[np.random.randint(len(pts))]
            cy = int(py) + np.random.randint(-nc // 3, nc // 3 + 1)
            cx = int(px) + np.random.randint(-nc // 3, nc // 3 + 1)
            y0 = max(0, min(H - nc, cy - nc // 2))
            x0 = max(0, min(W - nc, cx - nc // 2))
        else:
            y0 = np.random.randint(0, H - nc + 1)
            x0 = np.random.randint(0, W - nc + 1)

        im_c = img[y0 : y0 + nc, x0 : x0 + nc]
        sf_c = soft[y0 : y0 + nc, x0 : x0 + nc]
        sp_c = spine[y0 : y0 + nc, x0 : x0 + nc]
        if self.zoom > 1:
            sz = (self.crop_size, self.crop_size)
            im_c = cv2.resize(im_c, sz, interpolation=cv2.INTER_LINEAR)
            sf_c = cv2.resize(sf_c, sz, interpolation=cv2.INTER_LINEAR)
            sp_c = cv2.resize(sp_c, sz, interpolation=cv2.INTER_LINEAR)

        img_t = torch.from_numpy(np.transpose(im_c, (2, 0, 1)).astype(np.float32) / 255.0)
        soft_t = torch.from_numpy(sf_c.astype(np.float32) / 255.0).unsqueeze(0)
        spine_t = torch.from_numpy(sp_c.astype(np.float32) / 255.0).unsqueeze(0)
        return img_t, soft_t, spine_t

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        soft = cv2.imread(str(CACHE_2048 / "soft" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        spine = cv2.imread(str(CACHE_2048 / "spine" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        pts = self.fg_centers.get(stem)
        imgs, softs, spines = [], [], []
        for _ in range(self.crops_per_read):
            im, sf, sp = self._sample_window(img, soft, spine, pts)
            imgs.append(im)
            softs.append(sf)
            spines.append(sp)
        return torch.stack(imgs), torch.stack(softs), torch.stack(spines)


def _collate_crops(batch):
    imgs = torch.cat([b[0] for b in batch], dim=0)
    softs = torch.cat([b[1] for b in batch], dim=0)
    spines = torch.cat([b[2] for b in batch], dim=0)
    return imgs, softs, spines


def build_model(arch: str, encoder: str, pretrained: bool = True) -> nn.Module:
    weights = (True if encoder.startswith("tu-") else "imagenet") if pretrained else None
    return smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=2,
    )


def build_optimizer(model: nn.Module, lr: float, encoder_lr_scale: float = 0.25, weight_decay: float = 1e-2):
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    norms = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)
    groups: Dict[Tuple[bool, bool], List[torch.nn.Parameter]] = {}
    for mod in model.modules():
        for name, param in mod.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            decay = not (name == "bias" or isinstance(mod, norms))
            groups.setdefault((decay, id(param) in encoder_ids), []).append(param)
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "weight_decay": weight_decay if decay else 0.0,
                "lr": lr * encoder_lr_scale if in_enc else lr,
            }
            for (decay, in_enc), params in groups.items()
        ]
    )


def lovasz_hinge_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge surrogate of 1 - IoU on soft targets (>=0.5 positive)."""
    losses = []
    with torch.amp.autocast(logits.device.type, enabled=False):
        for lg, tg in zip(logits.float().flatten(1), targets.float().flatten(1)):
            labels = (tg >= 0.5).float()
            errors, order = torch.sort(1.0 - lg * (2.0 * labels - 1.0), descending=True)
            sl = labels[order]
            pos = sl.sum()
            jaccard = 1.0 - (pos - sl.cumsum(0)) / (pos + (1.0 - sl).cumsum(0)).clamp_min(1e-6)
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
            losses.append(torch.dot(torch.relu(errors), jaccard))
    return torch.stack(losses).mean()


def compute_loss(
    logits: torch.Tensor,
    soft: torch.Tensor,
    spine: torch.Tensor,
    pos_weight: float = 2.0,
    lovasz_w: float = 0.0,
) -> torch.Tensor:
    lm = logits[:, 0:1].float()
    ls = logits[:, 1:2].float()
    pw = torch.tensor([pos_weight], device=logits.device, dtype=torch.float32)
    bce_m = F.binary_cross_entropy_with_logits(lm, soft, pos_weight=pw)

    # Subtract log(pos_weight) so Dice, Lovasz, and inference align at true P(y=1|x)=0.5
    lm_cal = lm - math.log(max(pos_weight, 1e-6))
    pm_cal = torch.sigmoid(lm_cal)
    dice_m = train_seg.soft_dice_loss(pm_cal, soft)
    loss_m = 0.50 * bce_m + 0.50 * dice_m
    if lovasz_w > 0.0:
        loss_m = loss_m + lovasz_w * lovasz_hinge_loss(lm_cal, soft)

    ps = torch.sigmoid(ls)
    pw_s = torch.tensor([5.0], device=logits.device, dtype=torch.float32)
    bce_s = F.binary_cross_entropy_with_logits(ls, spine, pos_weight=pw_s)
    loss_s = 0.50 * bce_s + 0.50 * train_seg.soft_dice_loss(ps, spine)
    return loss_m + 0.20 * loss_s


def _hann2d(size: int) -> torch.Tensor:
    w1 = np.clip(np.hanning(size + 2)[1:-1].astype(np.float32), 0.05, 1.0)
    return torch.from_numpy(np.outer(w1, w1))


def _tile_offsets(total: int, win: int, overlap: float) -> List[int]:
    stride = max(1, int(round(win * (1.0 - overlap))))
    offs = list(range(0, total - win + 1, stride))
    if offs[-1] != total - win:
        offs.append(total - win)
    return offs


@torch.no_grad()
def predict_tiled_2048(
    model: nn.Module,
    img_2048_hwc: np.ndarray,
    device: torch.device,
    crop_size: int = 512,
    zoom: int = 1,
    overlap: float = 0.25,
    n_tta: int = 4,
    pos_weight: float = 2.0,
    tile_batch: int = 16,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched sliding-window 2048x2048 inference with cosine blending and D4 TTA."""
    nc = crop_size // zoom
    offs = _tile_offsets(2048, nc, overlap)
    win = _hann2d(nc).to(device)
    img_chw = torch.from_numpy(np.transpose(img_2048_hwc, (2, 0, 1)).astype(np.float32) / 255.0)
    logit_shift = math.log(max(pos_weight, 1e-6))
    tta_keys = [0, 4, 1, 5, 2, 6, 3, 7][:n_tta]

    coords = [(y0, x0) for y0 in offs for x0 in offs]
    patches = torch.stack([img_chw[:, y0 : y0 + nc, x0 : x0 + nc] for y0, x0 in coords], dim=0)
    if zoom > 1:
        patches = F.interpolate(patches, size=(crop_size, crop_size), mode="bilinear", align_corners=False)

    p_acc = torch.zeros((len(coords), nc, nc), device=device, dtype=torch.float32)
    s_acc = torch.zeros((len(coords), nc, nc), device=device, dtype=torch.float32)

    for start in range(0, len(coords), tile_batch):
        xb = patches[start : start + tile_batch].to(device, non_blocking=True)
        bp = torch.zeros((xb.size(0), crop_size, crop_size), device=device, dtype=torch.float32)
        bs = torch.zeros((xb.size(0), crop_size, crop_size), device=device, dtype=torch.float32)
        for k in tta_keys:
            xk = train_seg.apply_d4(xb, k)
            if channels_last:
                xk = xk.to(memory_format=torch.channels_last)
            with torch.amp.autocast(xk.device.type, dtype=amp_dtype, enabled=(xk.device.type == "cuda")):
                logits = model(train_seg.normalize_imagenet(xk))
            pm = torch.sigmoid(logits[:, 0].float() - logit_shift)
            ps = torch.sigmoid(logits[:, 1].float())
            bp += train_seg.invert_d4(pm, k)
            bs += train_seg.invert_d4(ps, k)
        bp /= len(tta_keys)
        bs /= len(tta_keys)
        if zoom > 1:
            bp = F.interpolate(bp.unsqueeze(1), size=(nc, nc), mode="bilinear", align_corners=False).squeeze(1)
            bs = F.interpolate(bs.unsqueeze(1), size=(nc, nc), mode="bilinear", align_corners=False).squeeze(1)
        p_acc[start : start + tile_batch] = bp
        s_acc[start : start + tile_batch] = bs

    p_sum = torch.zeros((2048, 2048), device=device, dtype=torch.float32)
    s_sum = torch.zeros((2048, 2048), device=device, dtype=torch.float32)
    w_sum = torch.zeros((2048, 2048), device=device, dtype=torch.float32)
    for idx, (y0, x0) in enumerate(coords):
        p_sum[y0 : y0 + nc, x0 : x0 + nc] += p_acc[idx] * win
        s_sum[y0 : y0 + nc, x0 : x0 + nc] += s_acc[idx] * win
        w_sum[y0 : y0 + nc, x0 : x0 + nc] += win

    p2048 = (p_sum / w_sum).cpu().numpy().astype(np.float32)
    s2048 = (s_sum / w_sum).cpu().numpy().astype(np.float32)
    return p2048, s2048


def _resolve_run_dir(run_name: str, use_arch: bool = False) -> Path:
    target = paths.RUNS_DIR / run_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def train_fold(fold: int, args: argparse.Namespace, run_dir: Path) -> float:
    fold_dir = run_dir / f"fold{fold}"
    done_flag = fold_dir / ".done"
    if done_flag.exists() and not args.force:
        print(f"[Skip] {args.run_name} fold {fold} already finished (.done present)", flush=True)
        return float(json.loads(done_flag.read_text()).get("tta_pq", 0.0))

    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    if torch.cuda.is_available():
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

    fg_centers: Dict[str, np.ndarray] = {}
    for s in train_stems:
        sf = np.load(CACHE_1024 / "soft" / f"{s}.npy")
        ys, xs = np.nonzero(sf >= 128)
        fg_centers[s] = (np.stack([ys, xs], axis=1) * 2).astype(np.int16)

    tr_ds = Tile512Dataset(
        train_stems,
        fg_centers=fg_centers,
        crop_size=args.crop_size,
        zoom=args.zoom,
        crops_per_read=args.crops_per_read,
        fg_prob=args.fg_prob,
    )
    steps_per_ep = math.ceil(len(tr_ds) / args.img_batch_size) * args.reads_per_epoch
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.img_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
        collate_fn=_collate_crops,
    )

    model = build_model(args.arch, args.encoder, pretrained=True).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, lr=args.lr, encoder_lr_scale=args.encoder_lr_scale)
    total_steps = max((steps_per_ep // args.grad_accum) * args.epochs, 1)
    warmup_steps = max(steps_per_ep // args.grad_accum, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best.pt"
    best_pq = -1.0
    best_ep = -1
    lovasz_start = int(math.ceil(args.epochs * args.lovasz_start_frac))

    print(
        f"[Tile{args.crop_size} z={args.zoom} {args.arch}:{args.encoder}] "
        f"Fold {fold}: tr={len(train_stems)}, va={len(val_stems)}, ep={args.epochs}, "
        f"lovasz@{lovasz_start}+, pw={args.pos_weight}",
        flush=True,
    )
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        lovasz_w = args.lovasz_weight if ep >= lovasz_start else 0.0
        loss_sum = 0.0
        n_b = 0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.reads_per_epoch):
            for img_t, soft_t, spine_t in tr_loader:
                img_t = img_t.to(device, non_blocking=True)
                soft_t = soft_t.to(device, non_blocking=True)
                spine_t = spine_t.to(device, non_blocking=True)
                img_t, soft_t, spine_t = train_seg.gpu_augment_batch(img_t, soft_t, spine_t, do_affine=True)
                if args.channels_last:
                    img_t = img_t.to(memory_format=torch.channels_last)
                with torch.amp.autocast(xk.device.type, dtype=amp_dtype, enabled=(xk.device.type == "cuda")):
                    logits = model(train_seg.normalize_imagenet(img_t))
                    loss = compute_loss(
                        logits, soft_t, spine_t, pos_weight=args.pos_weight, lovasz_w=lovasz_w
                    ) / args.grad_accum
                loss.backward()
                loss_sum += float(loss.item()) * args.grad_accum
                n_b += 1
                if n_b % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
        if n_b % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        msg = (
            f"  ep {ep:02d}/{args.epochs} (lov={lovasz_w:.2f}) | "
            f"loss={loss_sum/max(n_b,1):.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s"
        )
        should_eval = ep == args.epochs or (ep >= args.eval_start and ep % args.eval_every == 0)
        if should_eval:
            model.eval()
            t_ev = time.time()
            val_probs = {}
            for stem in val_stems:
                hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
                p, _ = predict_tiled_2048(
                    model, hwc, device, crop_size=args.crop_size, zoom=args.zoom, overlap=args.overlap,
                    n_tta=1, pos_weight=args.pos_weight, tile_batch=args.tile_batch, amp_dtype=amp_dtype,
                    channels_last=args.channels_last,
                )
                p *= train_tile2048.disk_mask_2048(geom_map[stem], margin=1.005)
                val_probs[stem] = p
            pq, tb = train_tile2048.eval_2048_pq(val_probs, val_gt_recs, coco["rec_to_stem"], thresholds=(0.44, 0.50, 0.56))
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
    for d in (
        run_dir / "oof_prob_2048",
        run_dir / "oof_prob",
        run_dir / "oof_spine",
        fold_dir / "test_prob_2048",
        fold_dir / "test_prob",
        fold_dir / "test_spine",
    ):
        d.mkdir(parents=True, exist_ok=True)

    oof_p = {}
    for stem in val_stems:
        hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        p, s = predict_tiled_2048(
            model, hwc, device, crop_size=args.crop_size, zoom=args.zoom, overlap=args.overlap,
            n_tta=args.n_tta, pos_weight=args.pos_weight, tile_batch=args.tile_batch, amp_dtype=amp_dtype,
            channels_last=args.channels_last,
        )
        dm = train_tile2048.disk_mask_2048(geom_map[stem], margin=1.005)
        p *= dm
        s *= dm
        oof_p[stem] = p
        train_tile2048.save_u16_png(run_dir / "oof_prob_2048" / f"{stem}.png", p)
        np.save(run_dir / "oof_prob" / f"{stem}.npy", cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
        np.save(run_dir / "oof_spine" / f"{stem}.npy", cv2.resize(s, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))

    tta_pq, tb = train_tile2048.eval_2048_pq(
        oof_p, val_gt_recs, coco["rec_to_stem"], thresholds=(0.36, 0.40, 0.44, 0.48, 0.52, 0.56)
    )
    print(f"Tile512 Fold {fold} Best (ep {best_ep}) {args.n_tta}x-TTA Val PQ = {tta_pq:.4f} (@{tb:.2f})", flush=True)

    for stem in test_stems:
        hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        p, s = predict_tiled_2048(
            model, hwc, device, crop_size=args.crop_size, zoom=args.zoom, overlap=args.overlap,
            n_tta=args.n_tta, pos_weight=args.pos_weight, tile_batch=args.tile_batch, amp_dtype=amp_dtype,
            channels_last=args.channels_last,
        )
        dm = train_tile2048.disk_mask_2048(geom_map[stem], margin=1.005)
        p *= dm
        s *= dm
        train_tile2048.save_u16_png(fold_dir / "test_prob_2048" / f"{stem}.png", p)
        np.save(fold_dir / "test_prob" / f"{stem}.npy", cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
        np.save(fold_dir / "test_spine" / f"{stem}.npy", cv2.resize(s, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))

    done_flag.write_text(json.dumps({"fold": fold, "best_ep": best_ep, "tta_pq": tta_pq, "thr": tb}))
    return tta_pq


def infer_test_fold(fold: int, args: argparse.Namespace, run_dir: Path) -> None:
    """Export 2048px prob + 1024px prob/spine test predictions from saved fold checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    meta = json.loads((CACHE_1024 / "meta.json").read_text())
    geom_map = meta["geom_1024"]
    test_stems = sorted(meta["test_stems"])

    fold_dir = run_dir / f"fold{fold}"
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")

    model = train_seg.build_model(arch=args.arch, encoder=args.encoder, pretrained=False).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    for d in (fold_dir / "test_prob_2048", fold_dir / "test_prob", fold_dir / "test_spine"):
        d.mkdir(parents=True, exist_ok=True)

    for stem in test_stems:
        hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
        p, s = predict_tiled_2048(
            model,
            hwc,
            device,
            crop_size=args.crop_size,
            zoom=args.zoom,
            overlap=args.overlap,
            n_tta=args.n_tta,
            pos_weight=args.pos_weight,
            tile_batch=args.tile_batch,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
        )
        dm = train_tile2048.disk_mask_2048(geom_map[stem], margin=1.005)
        p *= dm
        s *= dm
        train_tile2048.save_u16_png(fold_dir / "test_prob_2048" / f"{stem}.png", p)
        np.save(fold_dir / "test_prob" / f"{stem}.npy", cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
        np.save(fold_dir / "test_spine" / f"{stem}.npy", cv2.resize(s, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))
    print(f"[{run_dir.name}] Fold {fold} exported {len(test_stems)} 2048px test maps ({args.n_tta}x-TTA)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="tu-hrnet_w32")
    ap.add_argument("--run-name", default="tile512_hrnet32")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--zoom", type=int, default=1)
    ap.add_argument("--crops-per-read", type=int, default=2)
    ap.add_argument("--reads-per-epoch", type=int, default=2)
    ap.add_argument("--img-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.25)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--lovasz-weight", type=float, default=0.35)
    ap.add_argument("--lovasz-start-frac", type=float, default=0.70)
    ap.add_argument("--fg-prob", type=float, default=0.78)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-start", type=int, default=8)
    ap.add_argument("--n-tta", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--channels-last", action="store_true", default=True)
    ap.add_argument("--no-arch", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--infer-test", action="store_true")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_name, use_arch=not args.no_arch)
    fold_list = [int(x) for x in args.folds.split(",") if x.strip()]
    if args.infer_test:
        for f in fold_list:
            infer_test_fold(f, args, run_dir)
        return
    scores = {}
    for f in fold_list:
        scores[f] = train_fold(f, args, run_dir)
    print(f"Summary [{args.run_name}] fold PQs: {scores}", flush=True)


if __name__ == "__main__":
    main()
