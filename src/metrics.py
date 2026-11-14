"""Official-exact Panoptic Quality (PQ) scorer for Solar Filament Segmentation 2026."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float
    n_pred: int
    n_gt: int

    def __str__(self) -> str:
        return (
            f"PQ={self.pq:.4f} (SQ={self.sq:.4f} x RQ={self.rq:.4f}) "
            f"TP={self.tp} FP={self.fp} FN={self.fn} (pred={self.n_pred}, gt={self.n_gt})"
        )


def encode_rle(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def counts_str(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def from_counts(counts: str, h: int = 2048, w: int = 2048) -> dict:
    return {"size": [h, w], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def evaluate_pq(
    gt_by_record: Mapping[str, Sequence[dict]],
    record_to_stem: Mapping[str, str],
    pred_by_stem: Mapping[str, Sequence[dict]],
    iou_thr: float = 0.5,
) -> Tuple[PQResult, List[Tuple[str, np.ndarray]]]:
    """Host-faithful pooled PQ across all annotator records.

    Args:
        gt_by_record: record_id -> list of pycocotools RLE dicts
        record_to_stem: record_id -> physical image stem
        pred_by_stem: physical image stem -> list of pycocotools RLE dicts (disjoint)
        iou_thr: strict IoU threshold (> 0.5)
    """
    tp_ious: List[float] = []
    fp = 0
    fn = 0
    n_pred_tot = 0
    n_gt_tot = 0
    hits_list: List[Tuple[str, np.ndarray]] = []

    for rec_id, gt_rles in gt_by_record.items():
        stem = record_to_stem[rec_id]
        pred_rles = pred_by_stem.get(stem, [])
        n_gt = len(gt_rles)
        n_pred = len(pred_rles)
        n_gt_tot += n_gt
        n_pred_tot += n_pred
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        iou = np.asarray(mask_utils.iou(list(gt_rles), list(pred_rles), [0] * n_pred), dtype=np.float64)
        cand_r, cand_c = np.nonzero(iou > iou_thr)
        hits = np.zeros_like(iou, dtype=bool)
        if len(cand_r) > 0:
            order = np.argsort(-iou[cand_r, cand_c])
            used_gt = set()
            used_pr = set()
            for idx in order.tolist():
                r_i, c_i = int(cand_r[idx]), int(cand_c[idx])
                if r_i in used_gt or c_i in used_pr:
                    continue
                used_gt.add(r_i)
                used_pr.add(c_i)
                hits[r_i, c_i] = True
                tp_ious.append(float(iou[r_i, c_i]))
        fp += n_pred - int(hits.sum())
        fn += n_gt - int(hits.sum())
        hits_list.append((rec_id, hits))

    tp = len(tp_ious)
    sum_iou = float(sum(tp_ious))
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denom if denom > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    rq = tp / denom if denom > 0 else 0.0
    return PQResult(pq=pq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn, sum_iou=sum_iou, n_pred=n_pred_tot, n_gt=n_gt_tot), hits_list


def relation_counts(hits_list: Sequence[Tuple[str, np.ndarray]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    gt_deg: Counter = Counter()
    pred_deg: Counter = Counter()
    for _, hits in hits_list:
        for c in hits.sum(axis=1).tolist():
            gt_deg[int(c)] += 1
        for c in hits.sum(axis=0).tolist():
            pred_deg[int(c)] += 1
    return dict(sorted(gt_deg.items())), dict(sorted(pred_deg.items()))
