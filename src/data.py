"""Dataset loading, observation-grouped 5-fold splitting, and soft consensus cache."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from sklearn.model_selection import StratifiedGroupKFold

from . import paths
from .geometry import fit_solar_disk, normalize_disk_and_flatfield

DATA_ROOT = paths.DATA_ROOT
TRAIN_JSON = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES_DIR = DATA_ROOT / "train/train_images"
TEST_IMAGES_DIR = DATA_ROOT / "test/test_images"


def load_coco_grouped(json_path: Path | None = None):
    """Load COCO annotations grouped by physical observation stem."""
    if json_path is None:
        json_path = paths.DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    with open(json_path) as f:
        coco = json.load(f)

    anns_by_rec: Dict[str, List[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_rec[str(a["image_id"])].append(a)

    recs_by_stem: Dict[str, List[str]] = defaultdict(list)
    file_by_stem: Dict[str, str] = {}
    rec_to_stem: Dict[str, str] = {}
    for im in coco["images"]:
        rec_id = str(im["id"])
        fn = im["file_name"]
        stem = fn.rsplit(".", 1)[0]
        recs_by_stem[stem].append(rec_id)
        file_by_stem[stem] = fn
        rec_to_stem[rec_id] = stem

    stems = sorted(recs_by_stem.keys())
    gt_rles_2048: Dict[str, List[dict]] = {}
    for rec_id in rec_to_stem:
        rles = []
        for ann in anns_by_rec[rec_id]:
            rle_objs = mask_utils.frPyObjects(ann["segmentation"], 2048, 2048)
            rle = mask_utils.merge(rle_objs) if isinstance(rle_objs, list) else rle_objs
            rles.append(rle)
        gt_rles_2048[rec_id] = rles

    return {
        "anns_by_rec": dict(anns_by_rec),
        "recs_by_stem": dict(recs_by_stem),
        "file_by_stem": file_by_stem,
        "rec_to_stem": rec_to_stem,
        "stems": stems,
        "gt_rles_2048": gt_rles_2048,
    }


def make_folds(stems: List[str], recs_by_stem: Dict[str, List[str]], n_splits: int = 5, seed: int = 42) -> Dict[str, int]:
    """Stratify stems by station x multiplicity into 5 grouped folds."""
    strata = [f"{s[-2:]}_{min(len(recs_by_stem[s]), 3)}" for s in stems]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: Dict[str, int] = {}
    for fold, (_, val_idx) in enumerate(sgkf.split(stems, strata, groups=stems)):
        for i in val_idx:
            folds[stems[i]] = fold
    return folds


def rasterize_soft_and_spine(
    rec_ids: List[str],
    anns_by_rec: Dict[str, List[dict]],
    out_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (soft_consensus_u8, spine_union_u8) from annotator polygons for one image stem."""
    soft_sum = np.zeros((2048, 2048), dtype=np.float32)
    spine_mask = np.zeros((2048, 2048), dtype=np.uint8)
    recs = rec_ids if rec_ids else []
    for rec_id in recs:
        rec_union = np.zeros((2048, 2048), dtype=np.uint8)
        for ann in anns_by_rec.get(rec_id, []):
            rle_objs = mask_utils.frPyObjects(ann["segmentation"], 2048, 2048)
            rle = mask_utils.merge(rle_objs) if isinstance(rle_objs, list) else rle_objs
            rec_union |= mask_utils.decode(rle).astype(np.uint8)
            sp = ann.get("spine")
            if isinstance(sp, list) and len(sp) >= 4:
                pts = np.asarray(sp, dtype=np.float32).reshape(-1, 2)
                pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(spine_mask, [pts_i], isClosed=False, color=255, thickness=4, lineType=cv2.LINE_AA)
        soft_sum += rec_union.astype(np.float32)

    soft_2048 = soft_sum / max(len(recs), 1)
    if out_size != 2048:
        soft_out = cv2.resize(soft_2048, (out_size, out_size), interpolation=cv2.INTER_AREA)
        spine_out = cv2.resize(spine_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    else:
        soft_out = soft_2048
        spine_out = spine_mask
    return np.clip(soft_out * 255.0, 0, 255).round().astype(np.uint8), spine_out


def build_feature_image(img_2048: np.ndarray, work_size: int = 1024) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Resize grayscale 2048 image if needed, fit solar disk, and return (3xHxW uint8, (cx, cy, r))."""
    if work_size != 2048:
        img = cv2.resize(img_2048, (work_size, work_size), interpolation=cv2.INTER_AREA)
    else:
        img = img_2048
    cx, cy, r = fit_solar_disk(img)
    c0, c1, c2 = normalize_disk_and_flatfield(img, cx, cy, r)
    return np.stack([c0, c1, c2], axis=0), (cx, cy, r)
