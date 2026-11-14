"""Error analysis & OOF PQ breakdown by filament size, annotator multiplicity, station, and disk radius."""

from __future__ import annotations

import json
from typing import Dict, List

from pycocotools import mask as mask_utils

from src import metrics, paths


def analyze_oof(pred_by_stem: Dict[str, List[dict]], gt_recs: Dict[str, List[dict]], coco: dict) -> None:
    res, hits_list = metrics.evaluate_pq(gt_recs, coco["rec_to_stem"], pred_by_stem)
    print(f"OVERALL: {res}")
    gt_deg, pr_deg = metrics.relation_counts(hits_list)
    print("GT match degrees:", dict(sorted(gt_deg.items())), "Pred match degrees:", dict(sorted(pr_deg.items())))

    _ = json.loads((paths.CACHE_1024 / "meta.json").read_text())
    stems = sorted(pred_by_stem.keys())
    for label, cond in [
        ("mult=1", lambda s: len(coco["recs_by_stem"][s]) == 1),
        ("mult>=2", lambda s: len(coco["recs_by_stem"][s]) >= 2),
    ]:
        sub_recs = {r: gt_recs[r] for s in stems if cond(s) for r in coco["recs_by_stem"][s] if r in gt_recs}
        if sub_recs:
            sub_res, _ = metrics.evaluate_pq(sub_recs, coco["rec_to_stem"], pred_by_stem)
            print(f"  [{label}] ({len(sub_recs)} recs): {sub_res}")

    small_tot = small_hit = mid_tot = mid_hit = large_tot = large_hit = 0
    for rec_id, hits in hits_list:
        rles = gt_recs[rec_id]
        matched = hits.sum(axis=1) > 0
        for rle, ok in zip(rles, matched):
            area = int(mask_utils.area(rle))
            if area < 450:
                small_tot += 1
                small_hit += int(ok)
            elif area < 1600:
                mid_tot += 1
                mid_hit += int(ok)
            else:
                large_tot += 1
                large_hit += int(ok)
    print(
        f"  GT Recall by 2048px area: "
        f"<450: {small_hit}/{small_tot} ({small_hit/max(small_tot,1):.3f}), "
        f"450..1600: {mid_hit}/{mid_tot} ({mid_hit/max(mid_tot,1):.3f}), "
        f">=1600: {large_hit}/{large_tot} ({large_hit/max(large_tot,1):.3f})"
    )

    p_s_tot = p_s_hit = p_m_tot = p_m_hit = p_l_tot = p_l_hit = 0
    for rec_id, hits in hits_list:
        stem = coco["rec_to_stem"][rec_id]
        rles = pred_by_stem.get(stem, [])
        matched = hits.sum(axis=0) > 0
        for rle, ok in zip(rles, matched):
            area = int(mask_utils.area(rle))
            if area < 450:
                p_s_tot += 1
                p_s_hit += int(ok)
            elif area < 1600:
                p_m_tot += 1
                p_m_hit += int(ok)
            else:
                p_l_tot += 1
                p_l_hit += int(ok)
    print(
        f"  Pred Precision by 2048px area: "
        f"<450: {p_s_hit}/{p_s_tot} ({p_s_hit/max(p_s_tot,1):.3f}), "
        f"450..1600: {p_m_hit}/{p_m_tot} ({p_m_hit/max(p_m_tot,1):.3f}), "
        f">=1600: {p_l_hit}/{p_l_tot} ({p_l_hit/max(p_l_tot,1):.3f})"
    )
