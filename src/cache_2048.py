"""Precompute native 2048x2048 normalized 3-ch PNGs and soft/spine target PNGs."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src import data, geometry, paths


def _proc_img(args: Tuple[str, str, Tuple[float, float, float], str]) -> str:
    cv2.setNumThreads(1)
    stem, path_str, geom_1024, cache_2048_str = args
    out_p = Path(cache_2048_str) / "img" / f"{stem}.png"
    if out_p.exists():
        return stem
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path_str)
    cx, cy, r = geom_1024[0] * 2.0, geom_1024[1] * 2.0, geom_1024[2] * 2.0
    d, f, l = geometry.normalize_disk_and_flatfield(img, cx, cy, r)
    hwc = np.stack([d, f, l], axis=-1)
    cv2.imwrite(str(out_p), hwc, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return stem


def _proc_tgt(args: Tuple[str, list, dict, str]) -> str:
    cv2.setNumThreads(1)
    stem, recs, anns_sub, cache_2048_str = args
    cache_2048 = Path(cache_2048_str)
    sp_p = cache_2048 / "spine" / f"{stem}.png"
    sf_p = cache_2048 / "soft" / f"{stem}.png"
    if sf_p.exists() and sp_p.exists():
        return stem
    soft, spine = data.rasterize_soft_and_spine(recs, anns_sub, out_size=2048)
    cv2.imwrite(str(sf_p), soft, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cv2.imwrite(str(sp_p), spine, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return stem


def main(
    max_workers: int = max(1, min(24, (os.cpu_count() or 8) - 4)),
    test_only: bool = False,
    data_root: Path | None = None,
    cache_1024: Path | None = None,
    cache_2048: Path | None = None,
) -> None:
    data_root = Path(data_root) if data_root is not None else paths.DATA_ROOT
    cache_1024 = Path(cache_1024) if cache_1024 is not None else paths.CACHE_1024
    cache_2048 = Path(cache_2048) if cache_2048 is not None else paths.CACHE_2048

    (cache_2048 / "img").mkdir(parents=True, exist_ok=True)
    if not test_only:
        (cache_2048 / "soft").mkdir(parents=True, exist_ok=True)
        (cache_2048 / "spine").mkdir(parents=True, exist_ok=True)

    meta = json.loads((cache_1024 / "meta.json").read_text())
    geom_map = meta["geom_1024"]

    jobs_img = []
    coco = None
    if not test_only:
        coco = data.load_coco_grouped(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        train_dir = data_root / "train/train_images"
        for s in meta["train_stems"]:
            jobs_img.append((s, str(train_dir / coco["file_by_stem"][s]), tuple(geom_map[s]), str(cache_2048)))

    test_dir = data_root / "test/test_images"
    for s in meta["test_stems"]:
        jobs_img.append((s, str(test_dir / f"{s}.jpeg"), tuple(geom_map[s]), str(cache_2048)))

    workers = min(max_workers, os.cpu_count() or 8)
    print(f"Building native-2048 PNG cache for {len(jobs_img)} images ({workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, _ in enumerate(ex.map(_proc_img, jobs_img, chunksize=8)):
            if (idx + 1) % 200 == 0 or (idx + 1) == len(jobs_img):
                print(f"  imgs {idx+1}/{len(jobs_img)}", flush=True)

    if not test_only and coco is not None:
        jobs_tgt = [
            (s, coco["recs_by_stem"][s], {r: coco["anns_by_rec"][r] for r in coco["recs_by_stem"][s]}, str(cache_2048))
            for s in meta["train_stems"]
        ]
        print(f"Rasterizing 2048x2048 targets for {len(jobs_tgt)} train stems...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, _ in enumerate(ex.map(_proc_tgt, jobs_tgt, chunksize=8)):
                if (idx + 1) % 200 == 0 or (idx + 1) == len(jobs_tgt):
                    print(f"  targets {idx+1}/{len(jobs_tgt)}", flush=True)

    print("cache_2048 complete!", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()
    main(max_workers=args.workers, test_only=args.test_only)
