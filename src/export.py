"""Validated submission builder and strict format/disjointness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from src import metrics


def encode_disjoint_crops_2048(
    crops_and_scores: Sequence[Tuple[int, int, int, int, np.ndarray, float]],
    min_area: int = 100,
    top_k: int = 40,
) -> List[dict]:
    """Encode strictly pixel-disjoint 2048x2048 RLEs from tight bbox crops without 4MB per-mask allocs."""
    if not crops_and_scores:
        return []
    ordered = sorted(crops_and_scores, key=lambda x: x[5], reverse=True)[:top_k]
    occupied = np.zeros((2048, 2048), dtype=bool)
    buf = np.zeros((2048, 2048), dtype=np.uint8, order="F")
    out: List[dict] = []
    fallback = None

    for x1, y1, x2, y2, crop_m, _ in ordered:
        clean = crop_m.astype(bool, copy=False) & (~occupied[y1:y2, x1:x2])
        area = int(clean.sum())
        if fallback is None and area > 0:
            fallback = (x1, y1, x2, y2, clean.copy())
        if area < min_area:
            continue
        occupied[y1:y2, x1:x2] |= clean
        buf[y1:y2, x1:x2] = clean
        out.append(mask_utils.encode(buf))
        buf[y1:y2, x1:x2] = 0

    if not out and fallback is not None:
        x1, y1, x2, y2, clean = fallback
        buf[y1:y2, x1:x2] = clean
        out.append(mask_utils.encode(buf))
        buf[y1:y2, x1:x2] = 0
    return out


def enforce_disjoint_2048(
    masks_and_scores: Sequence[Tuple[np.ndarray, float]],
    min_area: int = 100,
) -> List[dict]:
    """Ensure strictly pixel-disjoint 2048x2048 binary RLEs ordered by descending score."""
    if not masks_and_scores:
        return []
    ordered = sorted(masks_and_scores, key=lambda x: x[1], reverse=True)
    occupied = np.zeros((2048, 2048), dtype=bool)
    out: List[dict] = []
    for mask, _ in ordered:
        clean = mask.astype(bool, copy=False) & (~occupied)
        if int(clean.sum()) < min_area:
            continue
        occupied |= clean
        out.append(metrics.encode_rle(clean.astype(np.uint8)))
    return out


def write_and_verify_submission(
    pred_rles_by_stem: Dict[str, List[dict]],
    test_stems: Sequence[str],
    out_path: Path,
) -> pd.DataFrame:
    """Write Kaggle CSV and verify disjointness, schema, roundtrip RLE, and coverage."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    counts_per_img = []
    areas = []

    empty_stems: List[str] = []
    for stem in sorted(test_stems):
        rles = pred_rles_by_stem.get(stem, [])
        if not rles:
            empty_stems.append(stem)
            continue
        canvas = np.zeros((2048, 2048), dtype=np.uint8)
        kept = 0
        for rle in rles:
            m = mask_utils.decode(rle).astype(np.uint8)
            if m.shape != (2048, 2048):
                raise ValueError(f"Invalid mask shape {m.shape} for {stem}")
            area = int(m.sum())
            if area == 0:
                continue
            if np.any((canvas > 0) & (m > 0)):
                raise ValueError(f"Pixel overlap detected inside {stem}")
            canvas |= m
            kept += 1
            areas.append(area)
            s = metrics.counts_str(rle)
            # Roundtrip verify
            dec = mask_utils.decode(metrics.from_counts(s, 2048, 2048))
            if int(dec.sum()) != area:
                raise ValueError(f"RLE roundtrip mismatch for {stem}")
            rows.append((f"{stem}_{kept}", s))
        if kept == 0:
            empty_stems.append(stem)
            continue
        counts_per_img.append(kept)

    if len(counts_per_img) < max(1, len(test_stems) - 15) or len(areas) == 0:
        raise ValueError(
            f"Too many empty images: non-empty={len(counts_per_img)}/{len(test_stems)}, "
            f"empty={len(empty_stems)}"
        )

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if df.isna().any().any():
        raise ValueError("Submission DataFrame contains NaNs")
    df.to_csv(out_path, index=False)

    ca = np.array(counts_per_img)
    aa = np.array(areas)
    print(
        f"Verified {out_path.name}: rows={len(df)}, imgs={len(counts_per_img)}/{len(test_stems)}, "
        f"empty={len(empty_stems)}, "
        f"inst/img min={ca.min()} med={np.median(ca):.1f} mean={ca.mean():.2f} max={ca.max()}, "
        f"area q01/10/50/90={np.percentile(aa, [1, 10, 50, 90]).astype(int).tolist()}"
    )
    if empty_stems:
        print(f"  Empty stems omitted ({len(empty_stems)}): {empty_stems}")
    return df
