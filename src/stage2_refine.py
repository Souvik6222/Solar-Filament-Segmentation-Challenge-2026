"""Stage-2 instance-conditioned rotated-crop refiner (128x384) and PQ evaluator."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

from src import data, eval_oof_breakdown, export, metrics, paths, run_tree_dp, train_seg

RUNS_DIR = paths.RUNS_DIR
CACHE_1024 = paths.CACHE_1024
CACHE_2048 = paths.CACHE_2048
SUB_DIR = paths.SUB_DIR


def _canon_rot_affine(
    pts_xy: np.ndarray,
    crop_h: int = 128,
    crop_w: int = 384,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Compute canonical major-axis affine matrices (M_fwd, M_inv) and 2048px ROI (x1, y1, x2, y2)."""
    rect = cv2.minAreaRect(pts_xy.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float64)
    center = box.mean(axis=0)
    e0 = box[1] - box[0]
    e1 = box[2] - box[1]
    l0 = float(np.linalg.norm(e0))
    l1 = float(np.linalg.norm(e1))
    if l0 >= l1 and l0 > 1e-3:
        u_vec = e0 / l0
        major, minor = l0, l1
    elif l1 > 1e-3:
        u_vec = e1 / l1
        major, minor = l1, l0
    else:
        u_vec = np.array([1.0, 0.0], dtype=np.float64)
        major, minor = 32.0, 16.0

    if u_vec[0] < -1e-6 or (abs(u_vec[0]) <= 1e-6 and u_vec[1] < 0.0):
        u_vec = -u_vec
    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=np.float64)

    box_w = max(major * 1.45 + 44.0, 128.0)
    box_h = max(minor * 1.80 + 40.0, box_w / 4.0, 48.0)
    box_w = max(box_w, box_h * 1.5)

    hw, hh = 0.5 * box_w, 0.5 * box_h
    src = np.array(
        [
            center - hw * u_vec - hh * v_vec,
            center + hw * u_vec - hh * v_vec,
            center - hw * u_vec + hh * v_vec,
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [-0.5, -0.5],
            [crop_w - 0.5, -0.5],
            [-0.5, crop_h - 0.5],
        ],
        dtype=np.float32,
    )
    M_fwd = cv2.getAffineTransform(src, dst)
    M_inv = cv2.invertAffineTransform(M_fwd)

    corners_crop = np.array(
        [
            [-0.5, -0.5, 1.0],
            [crop_w - 0.5, -0.5, 1.0],
            [-0.5, crop_h - 0.5, 1.0],
            [crop_w - 0.5, crop_h - 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    corners_img = corners_crop @ M_inv.T
    x1 = max(0, int(math.floor(float(corners_img[:, 0].min()))) - 2)
    y1 = max(0, int(math.floor(float(corners_img[:, 1].min()))) - 2)
    x2 = min(2048, int(math.ceil(float(corners_img[:, 0].max()))) + 3)
    y2 = min(2048, int(math.ceil(float(corners_img[:, 1].max()))) + 3)
    if x2 <= x1:
        x2 = min(2048, x1 + 1)
    if y2 <= y1:
        y2 = min(2048, y1 + 1)
    return M_fwd, M_inv, (x1, y1, x2, y2)


def _shift_fwd_for_slice(M_fwd: np.ndarray, x1: int, y1: int) -> np.ndarray:
    M = M_fwd.copy()
    M[:, 2] += M[:, 0] * float(x1) + M[:, 1] * float(y1)
    return M


def _shift_inv_for_roi(M_inv: np.ndarray, x1: int, y1: int) -> np.ndarray:
    M = M_inv.copy()
    M[0, 2] -= float(x1)
    M[1, 2] -= float(y1)
    return M


def load_stage1_prob_2048(stem: str, bundle: dict, is_test: bool) -> np.ndarray:
    """Reconstruct exact 2048x2048 float32 Stage-1 blended probability map for one stem."""
    tile_names = list(bundle.get("tile_names") or [])
    tile_weights = list(bundle.get("tile_weights") or [])
    run_names = list(bundle["run_names"])
    weights = list(bundle["weights"])
    w_2048 = float(bundle.get("w_2048", 0.5))

    p2048 = None
    if tile_names:
        p2048 = run_tree_dp._load_prob_2048(
            stem, [RUNS_DIR / r for r in tile_names], tile_weights, is_test=is_test
        )

    if p2048 is not None and (w_2048 >= 1.0 - 1e-6 or run_names == tile_names):
        return p2048.astype(np.float32, copy=False)

    prob_1024, _, _ = run_tree_dp._load_blended_prob(
        stem, [RUNS_DIR / r for r in run_names], weights, is_test=is_test
    )
    p1024_up = cv2.resize(prob_1024, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    if p2048 is None:
        return p1024_up
    return (1.0 - w_2048) * p1024_up + w_2048 * p2048


def _extract_stem_worker(args) -> dict:
    cv2.setNumThreads(1)
    stem, rles, gt_rles_by_rec, bundle, is_test, crop_h, crop_w = args
    if not rles:
        return {"stem": stem, "inputs": np.empty((0, 6, crop_h, crop_w), dtype=np.uint8), "meta": []}

    img_hwc = cv2.imread(str(CACHE_2048 / "img" / f"{stem}.png"), cv2.IMREAD_COLOR)
    if img_hwc is None:
        raise FileNotFoundError(CACHE_2048 / "img" / f"{stem}.png")
    p_s1 = load_stage1_prob_2048(stem, bundle, is_test=is_test)
    p_u8 = np.clip(p_s1 * 255.0, 0.0, 255.0).round().astype(np.uint8)

    # Disjoint label canvas (0 = bg, k+1 = inst k)
    inst_canvas = np.zeros((2048, 2048), dtype=np.int16)
    pred_masks: List[np.ndarray] = []
    pred_areas: List[int] = []
    pred_pts: List[np.ndarray] = []
    for k, rle in enumerate(rles):
        m = mask_utils.decode(rle).astype(bool)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        idx = len(pred_masks) + 1
        inst_canvas[ys, xs] = idx
        pred_masks.append(m)
        pred_areas.append(len(xs))
        pred_pts.append(np.stack([xs, ys], axis=1))

    # Optional GT matching across annotator records
    gt_rec_masks: List[List[Tuple[np.ndarray, int, Tuple[int, int, int, int]]]] = []
    if gt_rles_by_rec is not None:
        for grles in gt_rles_by_rec:
            rec_list = []
            for gr in grles:
                gm = mask_utils.decode(gr).astype(bool)
                gys, gxs = np.nonzero(gm)
                if len(gxs) == 0:
                    continue
                bbox = (int(gxs.min()), int(gys.min()), int(gxs.max()) + 1, int(gys.max()) + 1)
                rec_list.append((gm, len(gxs), bbox))
            gt_rec_masks.append(rec_list)

    n_inst = len(pred_masks)
    inputs_arr = np.zeros((n_inst, 6, crop_h, crop_w), dtype=np.uint8)
    targets_arr = np.zeros((n_inst, crop_h, crop_w), dtype=np.uint8) if gt_rles_by_rec is not None else None
    meta_list: List[dict] = []

    for k in range(n_inst):
        M_fwd, M_inv, (x1, y1, x2, y2) = _canon_rot_affine(pred_pts[k], crop_h=crop_h, crop_w=crop_w)
        M_slice = _shift_fwd_for_slice(M_fwd, x1, y1)
        M_inv_roi = _shift_inv_for_roi(M_inv, x1, y1)

        img_sl = img_hwc[y1:y2, x1:x2]
        p_sl = p_u8[y1:y2, x1:x2]
        ic_sl = inst_canvas[y1:y2, x1:x2]
        seed_sl = (ic_sl == (k + 1)).astype(np.uint8) * 255
        other_sl = ((ic_sl > 0) & (ic_sl != (k + 1))).astype(np.uint8) * 255

        # Pack 4 continuous channels for one SIMD INTER_LINEAR warp
        cont_sl = np.dstack([img_sl, p_sl])
        cont_w = cv2.warpAffine(
            cont_sl, M_slice, (crop_w, crop_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        # Binary conditioning masks
        mask_sl = np.dstack([seed_sl, other_sl])
        mask_w = cv2.warpAffine(
            mask_sl, M_slice, (crop_w, crop_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        inputs_arr[k, 0:4] = np.transpose(cont_w, (2, 0, 1))
        inputs_arr[k, 4] = mask_w[:, :, 0]
        inputs_arr[k, 5] = mask_w[:, :, 1]

        s1_seed_score = float(p_s1[pred_masks[k]].mean())
        info = {
            "stem": stem,
            "inst_idx": k,
            "roi": (x1, y1, x2, y2),
            "M_inv_roi": M_inv_roi.astype(np.float32),
            "s1_score": s1_seed_score,
            "seed_area": pred_areas[k],
        }

        if gt_rles_by_rec is not None:
            pxs = pred_pts[k][:, 0]
            pys = pred_pts[k][:, 1]
            pbx1, pby1, pbx2, pby2 = int(pxs.min()), int(pys.min()), int(pxs.max()) + 1, int(pys.max()) + 1
            pm_slice = pred_masks[k]

            matched_sum = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            n_matched = 0
            rec_ious: List[float] = []
            for rec_list in gt_rec_masks:
                best_iou = 0.0
                best_gm = None
                for gm, ga, (gx1, gy1, gx2, gy2) in rec_list:
                    if gx2 <= pbx1 or pbx2 <= gx1 or gy2 <= pby1 or pby2 <= gy1:
                        continue
                    ix1, iy1 = max(pbx1, gx1), max(pby1, gy1)
                    ix2, iy2 = min(pbx2, gx2), min(pby2, gy2)
                    inter = int((pm_slice[iy1:iy2, ix1:ix2] & gm[iy1:iy2, ix1:ix2]).sum())
                    if inter == 0:
                        continue
                    iou = float(inter) / float(pred_areas[k] + ga - inter)
                    if iou > best_iou:
                        best_iou = iou
                        best_gm = gm
                rec_ious.append(best_iou)
                if best_iou >= 0.20 and best_gm is not None:
                    matched_sum += best_gm[y1:y2, x1:x2].astype(np.float32)
                    n_matched += 1

            n_recs = max(len(gt_rec_masks), 1)
            keep_tgt = float(np.mean([1.0 if iou >= 0.25 else 0.0 for iou in rec_ious])) if rec_ious else 0.0
            has_match = int(n_matched > 0)
            seg_weight = float(n_matched) / float(n_recs) if has_match else 0.0

            if has_match:
                tgt_sl = np.clip((matched_sum / float(n_matched)) * 255.0, 0.0, 255.0).round().astype(np.uint8)
                tgt_w = cv2.warpAffine(
                    tgt_sl, M_slice, (crop_w, crop_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                targets_arr[k] = tgt_w

            info["has_match"] = has_match
            info["keep_target"] = keep_tgt
            info["seg_weight"] = seg_weight
            info["max_iou"] = max(rec_ious) if rec_ious else 0.0

        meta_list.append(info)

    out = {"stem": stem, "inputs": inputs_arr, "meta": meta_list}
    if targets_arr is not None:
        out["targets"] = targets_arr
    return out


def prepare_crop_cache(
    dp_dir: Path,
    cache_dir: Path,
    splits: Sequence[str] = ("oof", "test"),
    crop_h: int = 128,
    crop_w: int = 384,
    workers: int = 4,
    force: bool = False,
) -> dict:
    """Extract and memory-map rotated 128x384 crops from Stage-1 Tree DP predictions."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = pickle.loads((dp_dir / "tree_dp_bundle.pkl").read_bytes())
    bundle_cfg = {
        "run_names": list(bundle["run_names"]),
        "weights": list(bundle["weights"]),
        "tile_names": list(bundle.get("tile_names") or []),
        "tile_weights": list(bundle.get("tile_weights") or []),
        "w_2048": float(bundle.get("w_2048", 0.5)),
    }
    coco = data.load_coco_grouped() if "oof" in splits else None
    meta_1024 = json.loads((CACHE_1024 / "meta.json").read_text())

    summary = {"crop_h": crop_h, "crop_w": crop_w, "folds": meta_1024["folds"]}
    for split in splits:
        inp_path = cache_dir / f"{split}_inputs.npy"
        meta_path = cache_dir / f"{split}_meta.pkl"
        tgt_path = cache_dir / f"{split}_targets.npy"
        if (
            not force
            and inp_path.exists()
            and meta_path.exists()
            and (split == "test" or tgt_path.exists())
        ):
            meta_items = pickle.loads(meta_path.read_bytes())
            print(f"[Stage2 Cache] Reusing existing {split} cache ({len(meta_items)} crops) at {cache_dir}", flush=True)
            continue

        t0 = time.time()
        if split == "oof":
            rles_map: Dict[str, List[dict]] = bundle["oof_pred_rles"]
            tasks = [
                (
                    s,
                    rles_map[s],
                    [coco["gt_rles_2048"][r] for r in coco["recs_by_stem"][s]],
                    bundle_cfg,
                    False,
                    crop_h,
                    crop_w,
                )
                for s in sorted(rles_map.keys())
            ]
        elif split == "test":
            test_pkl = dp_dir / "test_pred_rles.pkl"
            if not test_pkl.exists():
                raise FileNotFoundError(f"Missing {test_pkl}; re-run run_tree_dp with --export-sub")
            rles_map = pickle.loads(test_pkl.read_bytes())
            tasks = [
                (s, rles_map[s], None, bundle_cfg, True, crop_h, crop_w)
                for s in sorted(rles_map.keys())
            ]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"[Stage2 Cache] Extracting {split} crops across {len(tasks)} stems (workers={workers})...", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_extract_stem_worker, tasks, chunksize=4))

        inputs_list = [r["inputs"] for r in results if len(r["meta"]) > 0]
        meta_items = [m for r in results for m in r["meta"]]
        all_inputs = (
            np.concatenate(inputs_list, axis=0)
            if inputs_list
            else np.empty((0, 6, crop_h, crop_w), dtype=np.uint8)
        )
        np.save(inp_path, all_inputs)
        if split == "oof":
            tgts_list = [r["targets"] for r in results if len(r["meta"]) > 0]
            all_tgts = (
                np.concatenate(tgts_list, axis=0)
                if tgts_list
                else np.empty((0, crop_h, crop_w), dtype=np.uint8)
            )
            np.save(tgt_path, all_tgts)
            matched = sum(m["has_match"] for m in meta_items)
            iou50 = sum(1 for m in meta_items if m["max_iou"] > 0.5)
            print(
                f"[Stage2 Cache] OOF: {len(meta_items)} crops from {len(tasks)} stems | "
                f"matched(>0.20)={matched} ({matched/max(len(meta_items),1):.3f}), "
                f"iou>0.50={iou50} ({iou50/max(len(meta_items),1):.3f}) | {time.time()-t0:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[Stage2 Cache] TEST: {len(meta_items)} crops from {len(tasks)} stems | {time.time()-t0:.1f}s",
                flush=True,
            )
        meta_path.write_bytes(pickle.dumps(meta_items))

    return summary


class Stage2CropDataset(Dataset):
    def __init__(
        self,
        inputs_mmap: np.ndarray,
        indices: Sequence[int],
        meta_items: Sequence[dict],
        targets_mmap: Optional[np.ndarray] = None,
        augment: bool = False,
    ):
        self.inputs = inputs_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.meta = meta_items
        self.targets = targets_mmap
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        arr = np.array(self.inputs[idx], copy=True)  # (6, H, W) uint8
        info = self.meta[idx]

        if self.targets is None:
            return idx, torch.from_numpy(arr)

        tgt = np.array(self.targets[idx], copy=True)  # (H, W) uint8
        if self.augment:
            if np.random.rand() < 0.5:
                arr = arr[:, :, ::-1].copy()
                tgt = tgt[:, ::-1].copy()
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1, :].copy()
                tgt = tgt[::-1, :].copy()

        return (
            torch.from_numpy(arr),
            torch.from_numpy(tgt),
            torch.tensor(info["has_match"], dtype=torch.float32),
            torch.tensor(info["keep_target"], dtype=torch.float32),
            torch.tensor(info["seg_weight"], dtype=torch.float32),
        )


def _normalize_6ch(x: torch.Tensor) -> torch.Tensor:
    """Normalize uint8 or float32 [0, 1] (B, 6, H, W) tensor -> model input."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    img = train_seg.normalize_imagenet(x[:, 0:3])
    cond = x[:, 3:6] * 2.0 - 1.0
    return torch.cat([img, cond], dim=1)


def _gpu_crop_augment(x_01: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spatial + photometric jitter on [0, 1] crops before ImageNet centering."""
    b, _, h, w = x_01.shape
    device = x_01.device
    if torch.rand(1, device=device).item() < 0.80:
        ang = (torch.rand(b, device=device) * 2.0 - 1.0) * (7.0 * math.pi / 180.0)
        sc = torch.exp((torch.rand(b, device=device) * 2.0 - 1.0) * 0.10)
        tx = (torch.rand(b, device=device) * 2.0 - 1.0) * (10.0 / max(w, 1))
        ty = (torch.rand(b, device=device) * 2.0 - 1.0) * (8.0 / max(h, 1))
        ca = torch.cos(ang) / sc
        sa = torch.sin(ang) / sc
        theta = torch.zeros((b, 2, 3), device=device, dtype=torch.float32)
        theta[:, 0, 0] = ca
        theta[:, 0, 1] = sa * (h / max(w, 1))
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = -sa * (w / max(h, 1))
        theta[:, 1, 1] = ca
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, size=(b, 1, h, w), align_corners=False)

        cont = torch.cat([x_01[:, 0:4], tgt.unsqueeze(1)], dim=1)
        cont = F.grid_sample(cont, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        masks = F.grid_sample(x_01[:, 4:6], grid, mode="nearest", padding_mode="zeros", align_corners=False)
        x_01 = torch.cat([cont[:, 0:4], masks], dim=1)
        tgt = cont[:, 4]

    gain = 1.0 + (torch.rand((b, 3, 1, 1), device=device) * 2.0 - 1.0) * 0.20
    bias = (torch.rand((b, 3, 1, 1), device=device) * 2.0 - 1.0) * 0.08
    x_01[:, 0:3] = (x_01[:, 0:3] * gain + bias).clamp(0.0, 1.0)

    # Slightly perturb Stage-1 probability channel so model checks H-alpha visual evidence
    p_scale = 1.0 - torch.rand((b, 1, 1, 1), device=device) * 0.25
    x_01[:, 3:4] = (x_01[:, 3:4] * p_scale).clamp(0.0, 1.0)
    return x_01, tgt


class Stage2RefinerNet(nn.Module):
    """Encoder-decoder crop refiner with instance-masked existence (keep) head."""

    def __init__(self, arch: str = "UnetPlusPlus", encoder: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.base = smp.create_model(
            arch=arch,
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=6,
            classes=1,
        )
        c_last = int(self.base.encoder.out_channels[-1])
        c_dec = 16
        self.keep_head = nn.Sequential(
            nn.Linear(c_last * 3 + c_dec, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def _masked_pool(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        num = (feat * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp_min(1e-3)
        return num / den

    def forward(self, x: torch.Tensor, seed_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.base.encoder(x)
        dec = self.base.decoder(feats)
        mask_logits = self.base.segmentation_head(dec)

        f_last = feats[-1]
        pred_01 = torch.sigmoid(mask_logits.detach())
        gap = f_last.mean(dim=(2, 3))
        seed_pool = self._masked_pool(f_last, seed_01)
        pred_pool = self._masked_pool(f_last, pred_01)
        dec_pool = self._masked_pool(dec, seed_01)
        keep_logits = self.keep_head(torch.cat([gap, seed_pool, pred_pool, dec_pool], dim=1)).squeeze(1)
        return mask_logits, keep_logits


def _stage2_loss(
    mask_logits: torch.Tensor,
    keep_logits: torch.Tensor,
    tgt_01: torch.Tensor,
    has_match: torch.Tensor,
    keep_tgt: torch.Tensor,
    seg_weight: torch.Tensor,
) -> torch.Tensor:
    loss_keep = F.binary_cross_entropy_with_logits(keep_logits, keep_tgt)
    w = seg_weight * has_match
    w_sum = w.sum()
    if w_sum.item() < 1e-4:
        return 0.40 * loss_keep

    logits_1 = mask_logits[:, 0]
    bce_map = F.binary_cross_entropy_with_logits(logits_1, tgt_01, reduction="none").mean(dim=(1, 2))
    loss_bce = (bce_map * w).sum() / w_sum

    prob = torch.sigmoid(logits_1)
    inter = (prob * tgt_01).sum(dim=(1, 2))
    den = (prob + tgt_01).sum(dim=(1, 2))
    dice_per = 1.0 - (2.0 * inter + 1.0) / (den + 1.0)
    loss_dice = (dice_per * w).sum() / w_sum

    return 0.50 * loss_bce + 0.50 * loss_dice + 0.35 * loss_keep


@torch.no_grad()
def _predict_crop_tta(model: nn.Module, x_norm: torch.Tensor, seed_01: torch.Tensor, n_tta: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """4x D2 (crop-aligned flip) TTA on (B, 6, H, W)."""
    p_sum = torch.zeros((x_norm.shape[0], x_norm.shape[2], x_norm.shape[3]), device=x_norm.device, dtype=torch.float32)
    k_sum = torch.zeros((x_norm.shape[0],), device=x_norm.device, dtype=torch.float32)
    flips = [(), (-1,), (-2,), (-2, -1)][:n_tta]
    for dims in flips:
        xi = torch.flip(x_norm, dims=dims) if dims else x_norm
        si = torch.flip(seed_01, dims=dims) if dims else seed_01
        ml, kl = model(xi, si)
        mp = torch.sigmoid(ml[:, 0].float())
        if dims:
            mp = torch.flip(mp, dims=dims)
        p_sum += mp
        k_sum += torch.sigmoid(kl.float())
    n = float(len(flips))
    return p_sum / n, k_sum / n


def train_fold(
    fold: int,
    cache_dir: Path,
    run_dir: Path,
    arch: str,
    encoder: str,
    epochs: int = 16,
    batch_size: int = 32,
    lr: float = 5e-4,
    n_tta: int = 4,
    seed: int = 200,
) -> float:
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    meta_1024 = json.loads((CACHE_1024 / "meta.json").read_text())
    folds_map = meta_1024["folds"]
    oof_inputs = np.load(cache_dir / "oof_inputs.npy", mmap_mode="r")
    oof_targets = np.load(cache_dir / "oof_targets.npy", mmap_mode="r")
    oof_meta = pickle.loads((cache_dir / "oof_meta.pkl").read_bytes())

    tr_idx = [i for i, m in enumerate(oof_meta) if folds_map[m["stem"]] != fold]
    va_idx = [i for i, m in enumerate(oof_meta) if folds_map[m["stem"]] == fold]

    tr_ds = Stage2CropDataset(oof_inputs, tr_idx, oof_meta, oof_targets, augment=True)
    va_ds = Stage2CropDataset(oof_inputs, va_idx, oof_meta, oof_targets, augment=False)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Stage2RefinerNet(arch=arch, encoder=encoder, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(len(tr_loader) * epochs, 1)
    warmup_steps = max(len(tr_loader), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ckpt_path = run_dir / f"fold{fold}_best.pt"
    best_score = -1e9
    best_dice = 0.0

    print(f"[Stage2 {arch}:{encoder}] Fold {fold}: tr={len(tr_idx)} crops, va={len(va_idx)} crops, ep={epochs}", flush=True)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_b = 0
        for arr_u8, tgt_u8, has_m, keep_t, seg_w in tr_loader:
            arr_u8 = arr_u8.to(device, non_blocking=True)
            tgt_01 = tgt_u8.to(device, non_blocking=True).float() / 255.0
            has_m = has_m.to(device, non_blocking=True)
            keep_t = keep_t.to(device, non_blocking=True)
            seg_w = seg_w.to(device, non_blocking=True)

            x_01 = arr_u8.float() / 255.0
            x_01, tgt_01 = _gpu_crop_augment(x_01, tgt_01)
            seed_01 = (x_01[:, 4:5] > 0.5).float()
            x_norm = _normalize_6ch(x_01)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                ml, kl = model(x_norm, seed_01)
                loss = _stage2_loss(ml, kl, tgt_01, has_m, keep_t, seg_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
            n_b += 1

        model.eval()
        dice_sum = 0.0
        n_matched = 0
        keep_bce_sum = 0.0
        n_tot = 0
        with torch.no_grad():
            for arr_u8, tgt_u8, has_m, keep_t, _ in va_loader:
                arr_u8 = arr_u8.to(device, non_blocking=True)
                tgt_01 = tgt_u8.to(device, non_blocking=True).float() / 255.0
                has_m = has_m.to(device, non_blocking=True) > 0.5
                keep_t = keep_t.to(device, non_blocking=True)
                seed_01 = (arr_u8[:, 4:5] > 127).float()
                x_norm = _normalize_6ch(arr_u8.float() / 255.0)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                    ml, kl = model(x_norm, seed_01)
                mp = (torch.sigmoid(ml[:, 0].float()) >= 0.5).float()
                inter = (mp * (tgt_01 >= 0.5).float()).sum(dim=(1, 2))
                den = mp.sum(dim=(1, 2)) + (tgt_01 >= 0.5).float().sum(dim=(1, 2))
                d = (2.0 * inter) / den.clamp_min(1.0)
                if has_m.any():
                    dice_sum += float(d[has_m].sum().item())
                    n_matched += int(has_m.sum().item())
                kb = F.binary_cross_entropy_with_logits(kl.float(), keep_t, reduction="sum")
                keep_bce_sum += float(kb.item())
                n_tot += len(keep_t)

        val_dice = dice_sum / max(n_matched, 1)
        val_kbce = keep_bce_sum / max(n_tot, 1)
        score = val_dice - 0.15 * val_kbce
        msg = (
            f"  ep {ep:02d}/{epochs} | loss={loss_sum/max(n_b,1):.4f} | "
            f"val_dice={val_dice:.4f} | val_kbce={val_kbce:.4f} | {time.time()-t0:.1f}s"
        )
        if score > best_score:
            best_score = score
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": val_dice, "val_kbce": val_kbce}, ckpt_path)
            msg += " *BEST*"
        print(msg, flush=True)

    # Export OOF crop predictions for va_idx
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    oof_p = np.empty((len(va_idx), oof_inputs.shape[2], oof_inputs.shape[3]), dtype=np.uint8)
    oof_k = np.empty((len(va_idx),), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        for arr_u8, _, _, _, _ in va_loader:
            arr_u8 = arr_u8.to(device, non_blocking=True)
            seed_01 = (arr_u8[:, 4:5] > 127).float()
            x_norm = _normalize_6ch(arr_u8)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                mp, kp = _predict_crop_tta(model, x_norm, seed_01, n_tta=n_tta)
            bs = arr_u8.shape[0]
            oof_p[pos : pos + bs] = np.clip(mp.cpu().numpy() * 255.0, 0.0, 255.0).round().astype(np.uint8)
            oof_k[pos : pos + bs] = kp.cpu().numpy()
            pos += bs
    np.savez_compressed(run_dir / f"oof_fold{fold}.npz", idx=np.asarray(va_idx, dtype=np.int64), probs=oof_p, keep=oof_k)
    return best_dice


def _reconstruct_stem_rois(
    stem_items: Sequence[Tuple[int, dict]],
    crop_probs_u8: np.ndarray,
    keep_probs: np.ndarray,
    s2_inputs: np.ndarray,
    crop_h: int,
    crop_w: int,
) -> List[dict]:
    """Warp Stage-2 crop probs & Stage-1 probs into shared ROI coords once per instance for fast sweeping."""
    k_dil = np.ones((11, 11), dtype=np.uint8)
    ones_crop = np.full((crop_h, crop_w), 255, dtype=np.uint8)
    rois = []
    for idx, m in stem_items:
        x1, y1, x2, y2 = m["roi"]
        rw, rh = x2 - x1, y2 - y1
        M_inv_roi = m["M_inv_roi"]

        p_s2_roi = cv2.warpAffine(
            crop_probs_u8[idx], M_inv_roi, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(np.float16) / np.float16(255.0)
        # Warp exact Tree-DP seed mask (preserving per-instance adaptive threshold) for trust-region blending
        seed_f_roi = cv2.warpAffine(
            s2_inputs[idx, 4], M_inv_roi, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(np.float16) / np.float16(255.0)
        seed_roi = seed_f_roi >= np.float16(0.5)
        valid_roi = cv2.warpAffine(
            ones_crop, M_inv_roi, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ) > 0
        seed_dil = cv2.dilate(seed_roi.astype(np.uint8), k_dil) > 0
        rois.append(
            {
                "roi": (x1, y1, x2, y2),
                "p_s2": p_s2_roi,
                "seed_f": seed_f_roi,
                "seed_dil": seed_dil,
                "valid": valid_roi,
                "keep": float(keep_probs[idx]),
                "s1_score": float(m["s1_score"]),
            }
        )
    return rois


def _mask_from_roi_entry(
    entry: dict, alpha: float, tau: float, min_area: int = 120
) -> Optional[Tuple[int, int, int, int, np.ndarray, float]]:
    p_comb = (
        alpha * entry["p_s2"].astype(np.float32) + (1.0 - alpha) * entry["seed_f"].astype(np.float32)
        if alpha < 1.0
        else entry["p_s2"].astype(np.float32)
    )
    fg = (p_comb >= tau) & entry["valid"]
    if not fg.any():
        return None
    # Keep only connected components touching the dilated Stage-1 seed so distant blobs cannot merge in
    n_cc, lbl = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n_cc <= 1:
        return None
    touch = np.unique(lbl[entry["seed_dil"] & (lbl > 0)])
    if len(touch) == 0:
        return None
    kept = np.isin(lbl, touch)
    if int(kept.sum()) < min_area:
        return None
    x1, y1, x2, y2 = entry["roi"]
    score = 0.65 * entry["keep"] + 0.35 * entry["s1_score"]
    return (x1, y1, x2, y2, kept, score)


def evaluate_and_sweep_oof(
    dp_dir: Path,
    cache_dir: Path,
    s2_run_dirs: Sequence[Path],
    s2_weights: Optional[Sequence[float]] = None,
    min_area_2048: int = 120,
    keep_baseline_fallback: bool = True,
) -> dict:
    """Evaluate blended Stage-2 models strictly on OOF PQ and pick optimal (alpha, tau, keep_thr)."""
    oof_meta = pickle.loads((cache_dir / "oof_meta.pkl").read_bytes())
    oof_inputs = np.load(cache_dir / "oof_inputs.npy", mmap_mode="r")
    n_crops, _, crop_h, crop_w = oof_inputs.shape
    w_list = list(s2_weights) if s2_weights else [1.0 / len(s2_run_dirs)] * len(s2_run_dirs)

    # Combine fold OOF arrays across multiple Stage-2 runs
    p_blend = np.zeros((n_crops, crop_h, crop_w), dtype=np.float32)
    k_blend = np.zeros((n_crops,), dtype=np.float32)
    covered = np.zeros((n_crops,), dtype=bool)
    for rdir, w in zip(s2_run_dirs, w_list):
        p_run = np.zeros((n_crops, crop_h, crop_w), dtype=np.float32)
        k_run = np.zeros((n_crops,), dtype=np.float32)
        cov_run = np.zeros((n_crops,), dtype=bool)
        for f in range(5):
            zp = rdir / f"oof_fold{f}.npz"
            if not zp.exists():
                continue
            z = np.load(zp)
            idx = z["idx"]
            p_run[idx] = z["probs"].astype(np.float32)
            k_run[idx] = z["keep"].astype(np.float32)
            cov_run[idx] = True
        p_blend += (w / sum(w_list)) * p_run
        k_blend += (w / sum(w_list)) * k_run
        covered |= cov_run

    val_indices = np.flatnonzero(covered)
    val_stems = sorted({oof_meta[i]["stem"] for i in val_indices})
    coco = data.load_coco_grouped()
    gt_recs = {r: coco["gt_rles_2048"][r] for s in val_stems for r in coco["recs_by_stem"][s]}

    # Baseline Stage-1 PQ on same covered stems
    bundle = pickle.loads((dp_dir / "tree_dp_bundle.pkl").read_bytes())
    s1_pred = {s: bundle["oof_pred_rles"][s] for s in val_stems}
    s1_pq, _ = metrics.evaluate_pq(gt_recs, coco["rec_to_stem"], s1_pred, iou_thr=0.5)
    print(f"[Stage2 OOF] Stems={len(val_stems)} ({len(val_indices)} crops) | Stage-1 PQ = {s1_pq.pq:.4f} (SQ={s1_pq.sq:.4f} x RQ={s1_pq.rq:.4f}, TP={s1_pq.tp}, FP={s1_pq.fp}, FN={s1_pq.fn})", flush=True)

    by_stem: Dict[str, List[Tuple[int, dict]]] = {s: [] for s in val_stems}
    for i in val_indices:
        by_stem[oof_meta[i]["stem"]].append((int(i), oof_meta[i]))

    t0 = time.time()
    p_u8 = np.clip(p_blend, 0.0, 255.0).round().astype(np.uint8)
    stem_rois = {
        s: _reconstruct_stem_rois(by_stem[s], p_u8, k_blend, oof_inputs, crop_h, crop_w)
        for s in val_stems
    }
    print(f"[Stage2 OOF] Pre-warped {len(val_indices)} ROIs in {time.time()-t0:.1f}s", flush=True)

    best = {"pq": -1.0}
    best_pred = s1_pred
    for alpha in (0.40, 0.55, 0.70, 0.85, 1.00):
        for tau in (0.40, 0.45, 0.50, 0.55):
            for keep_thr in (0.0, 0.14, 0.22):
                pred_by_stem: Dict[str, List[dict]] = {}
                for s in val_stems:
                    crops_scores = []
                    for entry in stem_rois[s]:
                        if entry["keep"] < keep_thr:
                            continue
                        item = _mask_from_roi_entry(entry, alpha=alpha, tau=tau)
                        if item is not None:
                            crops_scores.append(item)
                    rles = export.encode_disjoint_crops_2048(crops_scores, min_area=min_area_2048, top_k=50)
                    if not rles and keep_baseline_fallback:
                        rles = s1_pred[s]
                    pred_by_stem[s] = rles

                res, _ = metrics.evaluate_pq(gt_recs, coco["rec_to_stem"], pred_by_stem, iou_thr=0.5)
                if res.pq > best["pq"]:
                    best = {
                        "pq": res.pq,
                        "sq": res.sq,
                        "rq": res.rq,
                        "tp": res.tp,
                        "fp": res.fp,
                        "fn": res.fn,
                        "alpha": alpha,
                        "tau": tau,
                        "keep_thr": keep_thr,
                        "delta": res.pq - s1_pq.pq,
                    }
                    best_pred = pred_by_stem
                    print(
                        f"  alpha={alpha:.2f} tau={tau:.2f} keep>={keep_thr:.2f} -> PQ={res.pq:.4f} "
                        f"({res.pq - s1_pq.pq:+.4f}, SQ={res.sq:.4f} x RQ={res.rq:.4f}, TP={res.tp} FP={res.fp} FN={res.fn}) *BEST*",
                        flush=True,
                    )

    print("\n=== Stage-2 OOF Breakdown ===", flush=True)
    eval_oof_breakdown.analyze_oof(best_pred, gt_recs, coco)
    out_cfg = {
        "best": best,
        "s1_pq": s1_pq.pq,
        "s2_run_dirs": [str(p) for p in s2_run_dirs],
        "s2_weights": w_list,
        "min_area_2048": min_area_2048,
        "val_stems": val_stems,
    }
    return out_cfg


def export_test_predictions(
    dp_dir: Path,
    cache_dir: Path,
    s2_run_dirs: Sequence[Path],
    s2_weights: Sequence[float],
    alpha: float,
    tau: float,
    keep_thr: float,
    out_name: str,
    comb_thr: Optional[float] = None,
    out_csv: Optional[Path] = None,
    min_area_2048: int = 120,
    n_tta: int = 4,
    batch_size: int = 32,
) -> Path:
    """Run all trained folds of Stage-2 models on cached test crops and write verified submission CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    test_inputs = np.load(cache_dir / "test_inputs.npy", mmap_mode="r")
    test_meta = pickle.loads((cache_dir / "test_meta.pkl").read_bytes())
    test_pkl = pickle.loads((dp_dir / "test_pred_rles.pkl").read_bytes())
    test_stems = sorted(test_pkl.keys())
    n_crops, _, crop_h, crop_w = test_inputs.shape

    ds = Stage2CropDataset(test_inputs, list(range(n_crops)), test_meta, targets_mmap=None, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    p_blend = np.zeros((n_crops, crop_h, crop_w), dtype=np.float32)
    k_blend = np.zeros((n_crops,), dtype=np.float32)
    w_sum = 0.0

    for rdir, rw in zip(s2_run_dirs, s2_weights):
        cfg = json.loads((rdir / "config.json").read_text())
        arch, encoder = cfg["arch"], cfg["encoder"]
        ckpts = [rdir / f"fold{f}_best.pt" for f in range(5) if (rdir / f"fold{f}_best.pt").exists()]
        if not ckpts:
            continue
        p_run = np.zeros((n_crops, crop_h, crop_w), dtype=np.float32)
        k_run = np.zeros((n_crops,), dtype=np.float32)
        for ck in ckpts:
            model = Stage2RefinerNet(arch=arch, encoder=encoder, pretrained=False).to(device)
            state = torch.load(ck, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
            model.eval()
            with torch.no_grad():
                for idxs, arr_u8 in loader:
                    arr_u8 = arr_u8.to(device, non_blocking=True)
                    seed_01 = (arr_u8[:, 4:5] > 127).float()
                    x_norm = _normalize_6ch(arr_u8)
                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                        mp, kp = _predict_crop_tta(model, x_norm, seed_01, n_tta=n_tta)
                    np_idx = idxs.numpy()
                    p_run[np_idx] += mp.cpu().numpy() / float(len(ckpts))
                    k_run[np_idx] += kp.cpu().numpy() / float(len(ckpts))
        p_blend += rw * p_run
        k_blend += rw * k_run
        w_sum += rw

    p_blend = p_blend / max(w_sum, 1e-6)
    k_blend = k_blend / max(w_sum, 1e-6)
    p_u8 = np.clip(p_blend * 255.0, 0.0, 255.0).round().astype(np.uint8)

    by_stem: Dict[str, List[Tuple[int, dict]]] = {s: [] for s in test_stems}
    for i, m in enumerate(test_meta):
        by_stem[m["stem"]].append((i, m))

    test_pred_rles: Dict[str, List[dict]] = {}
    for s in test_stems:
        rois = _reconstruct_stem_rois(by_stem[s], p_u8, k_blend, test_inputs, crop_h, crop_w)
        crops_scores = []
        for entry in rois:
            comb = 0.65 * entry["keep"] + 0.35 * entry["s1_score"]
            if comb_thr is not None:
                if comb < comb_thr:
                    continue
            elif entry["keep"] < keep_thr:
                continue
            item = _mask_from_roi_entry(entry, alpha=alpha, tau=tau, min_area=min_area_2048)
            if item is not None:
                crops_scores.append(item)
        rles = export.encode_disjoint_crops_2048(crops_scores, min_area=min_area_2048, top_k=50)
        if not rles and test_pkl[s]:
            rles = test_pkl[s]
        test_pred_rles[s] = rles

    out_dir = RUNS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_pred_rles.pkl").write_bytes(pickle.dumps(test_pred_rles))
    sub_path = Path(out_csv) if out_csv is not None else (SUB_DIR / f"{out_name}.csv")
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    export.write_and_verify_submission(test_pred_rles, test_stems, sub_path)
    return sub_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp-name", required=True, help="Stage-1 Tree DP run name under runs/")
    ap.add_argument("--out-name", default="", help="Stage-2 run dir under runs/ (default: s2_<dp_name>)")
    ap.add_argument("--cache-dir", default="", help="Crop cache dir (default: data/s2_cache/<dp_name>)")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="mit_b2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--n-tta", type=int, default=4)
    ap.add_argument("--crop-h", type=int, default=128)
    ap.add_argument("--crop-w", type=int, default=384)
    ap.add_argument("--min-area-2048", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="Skip training & OOF; run test cache + Stage-2 test inference")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.50)
    ap.add_argument("--keep-thr", type=float, default=0.0)
    ap.add_argument("--comb-thr", type=float, default=0.48)
    ap.add_argument("--out-csv", default="", help="Optional exact path for final submission CSV")
    ap.add_argument("--ensemble-runs", default="", help="Optional comma-sep Stage-2 run names under runs/")
    ap.add_argument("--ensemble-weights", default="")
    ap.add_argument("--export-sub", action="store_true")
    args = ap.parse_args()

    dp_dir = RUNS_DIR / args.dp_name
    out_name = args.out_name or f"s2_{args.dp_name}"
    run_dir = RUNS_DIR / out_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (paths.S2_CACHE_ROOT / args.dp_name)

    if args.test_only:
        prepare_crop_cache(
            dp_dir=dp_dir,
            cache_dir=cache_dir,
            splits=("test",),
            crop_h=args.crop_h,
            crop_w=args.crop_w,
            workers=args.workers,
            force=args.force_cache,
        )
        ens_dirs = [RUNS_DIR / r.strip() for r in args.ensemble_runs.split(",") if r.strip()] if args.ensemble_runs else [run_dir]
        ens_weights = [float(x) for x in args.ensemble_weights.split(",") if x.strip()] if args.ensemble_weights else [1.0 / len(ens_dirs)] * len(ens_dirs)
        export_test_predictions(
            dp_dir=dp_dir,
            cache_dir=cache_dir,
            s2_run_dirs=ens_dirs,
            s2_weights=ens_weights,
            alpha=args.alpha,
            tau=args.tau,
            keep_thr=args.keep_thr,
            comb_thr=args.comb_thr if args.comb_thr >= 0.0 else None,
            out_name=out_name,
            out_csv=Path(args.out_csv) if args.out_csv else None,
            min_area_2048=args.min_area_2048,
            n_tta=args.n_tta,
            batch_size=args.batch_size,
        )
        return

    splits = ("oof",) if (args.prepare_only and not (dp_dir / "test_pred_rles.pkl").exists()) else ("oof", "test")
    prepare_crop_cache(
        dp_dir=dp_dir,
        cache_dir=cache_dir,
        splits=splits,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
        workers=args.workers,
        force=args.force_cache,
    )
    if args.prepare_only:
        return

    if not args.eval_only:
        (run_dir / "config.json").write_text(
            json.dumps({"arch": args.arch, "encoder": args.encoder, "crop_h": args.crop_h, "crop_w": args.crop_w})
        )
        scores = {}
        for f in [int(x) for x in args.folds.split(",") if x.strip()]:
            scores[f] = train_fold(
                fold=f,
                cache_dir=cache_dir,
                run_dir=run_dir,
                arch=args.arch,
                encoder=args.encoder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                n_tta=args.n_tta,
            )
        print(f"[Stage2 {out_name}] Fold val_dice: {scores}", flush=True)

    ens_dirs = [RUNS_DIR / r.strip() for r in args.ensemble_runs.split(",") if r.strip()] if args.ensemble_runs else [run_dir]
    ens_weights = [float(x) for x in args.ensemble_weights.split(",") if x.strip()] if args.ensemble_weights else [1.0 / len(ens_dirs)] * len(ens_dirs)

    res = evaluate_and_sweep_oof(
        dp_dir=dp_dir,
        cache_dir=cache_dir,
        s2_run_dirs=ens_dirs,
        s2_weights=ens_weights,
        min_area_2048=args.min_area_2048,
    )
    (run_dir / "oof_sweep.json").write_text(json.dumps(res, indent=2))

    if args.export_sub:
        b = res["best"]
        sub_name = out_name if not args.ensemble_runs else f"{out_name}_ens"
        export_test_predictions(
            dp_dir=dp_dir,
            cache_dir=cache_dir,
            s2_run_dirs=ens_dirs,
            s2_weights=ens_weights,
            alpha=float(b["alpha"]),
            tau=float(b["tau"]),
            keep_thr=float(b["keep_thr"]),
            out_name=sub_name,
            min_area_2048=args.min_area_2048,
            n_tta=args.n_tta,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
