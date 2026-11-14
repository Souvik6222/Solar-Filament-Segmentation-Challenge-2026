"""Precompute 1024x1024 normalized 3-ch images, soft targets, spine maps, and metadata."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from src import data, paths


def _process_one(args: Tuple[str, str, int, bool, str]) -> Tuple[str, Tuple[float, float, float]]:
    cv2.setNumThreads(1)
    stem, img_path_str, work_size, _, cache_dir_str = args
    img_2048 = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
    if img_2048 is None:
        raise FileNotFoundError(img_path_str)
    feat, geom = data.build_feature_image(img_2048, work_size=work_size)
    np.save(Path(cache_dir_str) / "img" / f"{stem}.npy", feat)
    return stem, geom


def _process_targets(stem: str, recs: list, anns_by_rec: dict, work_size: int, cache_dir: Path) -> Tuple[str, int]:
    soft, spine = data.rasterize_soft_and_spine(recs, anns_by_rec, out_size=work_size)
    np.save(cache_dir / "soft" / f"{stem}.npy", soft)
    np.save(cache_dir / "spine" / f"{stem}.npy", spine)
    return stem, len(recs)


def build_cache(
    work_size: int = 1024,
    max_workers: int = max(1, min(24, (os.cpu_count() or 8) - 4)),
    test_only: bool = False,
    data_root: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    data_root = Path(data_root) if data_root is not None else paths.DATA_ROOT
    cache_dir = Path(cache_dir) if cache_dir is not None else paths.CACHE_1024

    (cache_dir / "img").mkdir(parents=True, exist_ok=True)
    if not test_only:
        (cache_dir / "soft").mkdir(parents=True, exist_ok=True)
        (cache_dir / "spine").mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "meta.json"
    test_dir = data_root / "test/test_images"
    test_paths = sorted(test_dir.glob("*.jpeg"))
    if not test_paths:
        raise FileNotFoundError(f"No test .jpeg files found in {test_dir}")
    test_stems = [p.stem for p in test_paths]

    if test_only:
        train_stems = []
        folds: Dict[str, int] = {}
        coco = None
    else:
        coco = data.load_coco_grouped(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        folds = data.make_folds(coco["stems"], coco["recs_by_stem"])
        train_stems = sorted(coco["stems"])

    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if test_only:
            if (
                len(existing.get("test_stems", [])) == len(test_stems)
                and all(s in existing.get("geom_1024", {}) for s in test_stems)
                and (cache_dir / "img" / f"{test_stems[-1]}.npy").exists()
            ):
                print(f"Test cache already complete at {cache_dir}")
                return cache_dir
        elif (
            len(existing.get("train_stems", [])) == len(train_stems)
            and len(existing.get("test_stems", [])) == len(test_stems)
            and (cache_dir / "img" / f"{test_stems[-1]}.npy").exists()
            and (cache_dir / "soft" / f"{train_stems[-1]}.npy").exists()
        ):
            print(f"Cache already complete at {cache_dir}")
            return cache_dir

    train_dir = data_root / "train/train_images"
    jobs = []
    if not test_only and coco is not None:
        for s in train_stems:
            if not (cache_dir / "img" / f"{s}.npy").exists():
                jobs.append((s, str(train_dir / coco["file_by_stem"][s]), work_size, True, str(cache_dir)))
    for p in test_paths:
        if not (cache_dir / "img" / f"{p.stem}.npy").exists():
            jobs.append((p.stem, str(p), work_size, False, str(cache_dir)))

    geom_1024: Dict[str, list] = {}
    if meta_path.exists():
        geom_1024.update(json.loads(meta_path.read_text()).get("geom_1024", {}))

    workers = min(max_workers, os.cpu_count() or 8)
    if jobs:
        print(f"Computing {len(jobs)} image caches with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for stem, geom in ex.map(_process_one, jobs, chunksize=8):
                geom_1024[stem] = [float(geom[0]), float(geom[1]), float(geom[2])]

    missing_geom = [s for s in train_stems + test_stems if s not in geom_1024]
    if missing_geom:
        regen = []
        train_set = set(train_stems)
        for s in missing_geom:
            p = (
                train_dir / coco["file_by_stem"][s]
                if (s in train_set and coco is not None)
                else test_dir / f"{s}.jpeg"
            )
            regen.append((s, str(p), work_size, s in train_set, str(cache_dir)))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for stem, geom in ex.map(_process_one, regen, chunksize=8):
                geom_1024[stem] = [float(geom[0]), float(geom[1]), float(geom[2])]

    n_recs: Dict[str, int] = {}
    if not test_only and coco is not None:
        print(f"Rasterizing targets for {len(train_stems)} train stems...")
        for s in train_stems:
            recs = coco["recs_by_stem"][s]
            n_recs[s] = len(recs)
            if not ((cache_dir / "soft" / f"{s}.npy").exists() and (cache_dir / "spine" / f"{s}.npy").exists()):
                _process_targets(s, recs, coco["anns_by_rec"], work_size, cache_dir)

    rs = [geom_1024[s][2] for s in train_stems + test_stems]
    cxs = [geom_1024[s][0] for s in train_stems + test_stems]
    cys = [geom_1024[s][1] for s in train_stems + test_stems]
    print(
        f"Geometry sanity ({len(rs)} imgs): "
        f"cx=[{min(cxs):.1f},{max(cxs):.1f}] "
        f"cy=[{min(cys):.1f},{max(cys):.1f}] "
        f"R=[{min(rs):.1f},{max(rs):.1f}]"
    )

    existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {
        "work_size": work_size,
        "train_stems": train_stems or existing_meta.get("train_stems", []),
        "test_stems": test_stems,
        "folds": folds or existing_meta.get("folds", {}),
        "n_recs": n_recs or existing_meta.get("n_recs", {}),
        "geom_1024": geom_1024,
    }
    meta_path.write_text(json.dumps(meta))
    print(f"Saved cache metadata to {meta_path}")
    return cache_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()
    build_cache(work_size=args.work_size, max_workers=args.workers, test_only=args.test_only)
