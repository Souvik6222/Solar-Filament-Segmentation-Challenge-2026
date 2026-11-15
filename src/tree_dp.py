"""Multi-threshold Component Tree, Cross-Fitted Hurdle LightGBM, and Antichain DP for PQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from src import metrics

THRESHOLDS: Tuple[float, ...] = (
    0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42,
    0.46, 0.50, 0.55, 0.60, 0.65, 0.72, 0.80,
)

FEATURE_COLS: Tuple[str, ...] = (
    "thr",
    "area_2048",
    "log_area",
    "prob_mean",
    "prob_std",
    "prob_max",
    "prob_q10",
    "prob_q25",
    "prob_q50",
    "prob_q75",
    "prob_q90",
    "frac_gt_50",
    "frac_gt_70",
    "frac_gt_85",
    "rim_prob_mean",
    "halo_prob_mean",
    "halo_prob_ratio",
    "prob_drop",
    "spine_mean",
    "spine_max",
    "spine_density",
    "spine_frac_25",
    "n_spine_30",
    "dt_max",
    "dt_mean",
    "dt_std",
    "width_cv",
    "bbox_w",
    "bbox_h",
    "fill_ratio",
    "major_axis",
    "minor_axis",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "perimeter",
    "compactness",
    "r_cent",
    "r_min",
    "r_max",
    "flat_in_mean",
    "flat_ring_mean",
    "flat_contrast",
    "flat_in_q10",
    "diskz_in_mean",
    "diskz_contrast",
    "loc_contrast",
    "n_children",
    "child_area_ratio",
    "parent_area_ratio",
    "depth",
    "img_prob_mass",
    "n_roots_img",
    "inter_std_mean",
    "inter_agree_40",
)


@dataclass
class TreeImage:
    stem: str
    h: int
    w: int
    nodes: List[dict]
    children: List[List[int]]
    roots: List[int]
    df: pd.DataFrame


def _bbox_moments_and_solidity(sub_mask_u8: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    area = float(sub_mask_u8.sum())
    if area <= 0:
        return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
    m = cv2.moments(sub_mask_u8, binaryImage=True)
    m00 = max(m["m00"], 1.0)
    mu20 = m["mu20"] / m00
    mu02 = m["mu02"] / m00
    mu11 = m["mu11"] / m00
    disc = float(np.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * mu11 * mu11)))
    lam1 = max(0.0, 0.5 * (mu20 + mu02 + disc))
    lam2 = max(0.0, 0.5 * (mu20 + mu02 - disc))
    major = float(4.0 * np.sqrt(lam1)) if lam1 > 0 else 1.0
    minor = float(4.0 * np.sqrt(lam2)) if lam2 > 0 else 1.0
    aspect = major / max(minor, 0.5)
    ecc = float(np.sqrt(max(0.0, 1.0 - lam2 / max(lam1, 1e-6))))

    contours, _ = cv2.findContours(sub_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else 1.0
    if contours:
        all_pts = np.vstack(contours)
        hull_area = float(cv2.contourArea(cv2.convexHull(all_pts))) if len(all_pts) >= 3 else area
        solidity = area / max(hull_area, area, 1.0)
    else:
        solidity = 1.0
    return major, minor, aspect, ecc, solidity, perim


def build_tree_for_image(
    stem: str,
    prob: np.ndarray,
    img_3ch: np.ndarray,
    geom: Tuple[float, float, float],
    spine: Optional[np.ndarray] = None,
    model_probs: Optional[Sequence[np.ndarray]] = None,
    thresholds: Sequence[float] = THRESHOLDS,
    min_area: int = 25,
    max_area: int = 45000,
) -> TreeImage:
    """Build hierarchical component tree across probability thresholds with physics features."""
    h, w = prob.shape
    scale2 = (2048.0 / h) ** 2
    cx, cy, r_disk = geom
    r_disk = max(float(r_disk), 1.0)

    if spine is None:
        spine = np.zeros_like(prob, dtype=np.float32)

    if model_probs and len(model_probs) > 1:
        stacked = np.stack(model_probs, axis=0)
        inter_std_map = np.std(stacked, axis=0).astype(np.float32)
        inter_agree_map = np.mean(stacked >= 0.40, axis=0).astype(np.float32)
    else:
        inter_std_map = np.zeros_like(prob, dtype=np.float32)
        inter_agree_map = (prob >= 0.40).astype(np.float32)

    diskz = img_3ch[0].astype(np.float32)
    flat = img_3ch[1].astype(np.float32)
    loc = img_3ch[2].astype(np.float32)
    img_prob_mass = float((prob >= 0.30).sum()) / (np.pi * r_disk * r_disk)

    nodes: List[dict] = []
    # Keep track of active parent node indices per pixel from previous threshold level
    prev_node_map = np.full((h, w), -1, dtype=np.int32)
    k3 = np.ones((3, 3), dtype=np.uint8)
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for thr_idx, thr in enumerate(thresholds):
        bw = (prob >= thr).astype(np.uint8)
        n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
        cur_node_map = np.full((h, w), -1, dtype=np.int32)
        valid_ids = [
            k for k in range(1, n_lbl)
            if min_area <= stats[k, cv2.CC_STAT_AREA] <= max_area
        ]
        # Pass through parent pointers if intermediate component was above max_area
        if len(valid_ids) == 0 and n_lbl > 1:
            pass

        for k in valid_ids:
            x1 = int(stats[k, cv2.CC_STAT_LEFT])
            y1 = int(stats[k, cv2.CC_STAT_TOP])
            bw_k = int(stats[k, cv2.CC_STAT_WIDTH])
            bh_k = int(stats[k, cv2.CC_STAT_HEIGHT])
            x2 = x1 + bw_k
            y2 = y1 + bh_k
            area = int(stats[k, cv2.CC_STAT_AREA])

            sub_lbl = lbl[y1:y2, x1:x2]
            sub_mask = sub_lbl == k
            ys_loc, xs_loc = np.nonzero(sub_mask)
            ay = y1 + int(ys_loc[0])
            ax = x1 + int(xs_loc[0])
            parent_id = int(prev_node_map[ay, ax])

            # Skip exact duplicate node (same pixels as parent)
            if parent_id >= 0 and nodes[parent_id]["area"] == area:
                cur_node_map[y1:y2, x1:x2][sub_mask] = parent_id
                continue

            nid = len(nodes)
            cur_node_map[y1:y2, x1:x2][sub_mask] = nid

            # Padded crop for boundary & ring contrast
            px1 = max(0, x1 - 4)
            py1 = max(0, y1 - 4)
            px2 = min(w, x2 + 4)
            py2 = min(h, y2 + 4)
            pmask_u8 = (lbl[py1:py2, px1:px2] == k).astype(np.uint8)
            sub_u8 = sub_mask.astype(np.uint8)

            eroded = cv2.erode(pmask_u8, k3, iterations=1)
            rim_mask = (pmask_u8 > 0) & (eroded == 0)
            dilated_out = cv2.dilate(pmask_u8, k3, iterations=2)
            halo_mask = (dilated_out > 0) & (pmask_u8 == 0)
            ring_dil = cv2.dilate(pmask_u8, k_ring, iterations=1)
            ring_mask = (ring_dil > 0) & (pmask_u8 == 0)

            p_crop = prob[py1:py2, px1:px2]
            p_in = p_crop[pmask_u8 > 0]
            p_rim = p_crop[rim_mask] if rim_mask.any() else p_in
            p_halo = p_crop[halo_mask] if halo_mask.any() else np.array([0.0], dtype=np.float32)

            sp_crop = spine[y1:y2, x1:x2]
            sp_in = sp_crop[sub_mask]
            n_sp_lbl, _, sp_stats, _ = cv2.connectedComponentsWithStats(
                ((sp_crop >= 0.30) & sub_mask).astype(np.uint8), connectivity=8
            )
            n_spine_30 = float(sum(1 for j in range(1, n_sp_lbl) if sp_stats[j, cv2.CC_STAT_AREA] >= 3))
            std_in = inter_std_map[y1:y2, x1:x2][sub_mask]
            agr_in = inter_agree_map[y1:y2, x1:x2][sub_mask]

            # Distance transform thickness on tight mask with 1px zero border
            pad_u8 = cv2.copyMakeBorder(sub_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            dt = cv2.distanceTransform(pad_u8, cv2.DIST_L2, 5)[1:-1, 1:-1][sub_mask]
            dt_mean = float(dt.mean())
            dt_std = float(dt.std())

            major, minor, aspect, ecc, solidity, perim = _bbox_moments_and_solidity(sub_u8)
            compactness = float(4.0 * np.pi * area / max(perim * perim, 1.0))

            # Radial geometry normalized by solar disk radius
            mcx, mcy = float(cents[k, 0]), float(cents[k, 1])
            r_cent = float(np.hypot(mcx - cx, mcy - cy) / r_disk)
            ys_abs = y1 + ys_loc
            xs_abs = x1 + xs_loc
            r_pix = np.hypot(xs_abs - cx, ys_abs - cy) / r_disk
            r_min = float(r_pix.min())
            r_max = float(r_pix.max())

            # Appearance intensity contrast
            f_crop = flat[py1:py2, px1:px2]
            d_crop = diskz[py1:py2, px1:px2]
            l_crop = loc[py1:py2, px1:px2]
            f_in = f_crop[pmask_u8 > 0]
            f_ring = f_crop[ring_mask] if ring_mask.any() else f_in
            d_in = d_crop[pmask_u8 > 0]
            d_ring = d_crop[ring_mask] if ring_mask.any() else d_in
            l_in = l_crop[pmask_u8 > 0]
            l_ring = l_crop[ring_mask] if ring_mask.any() else l_in

            q10, q25, q50, q75, q90 = np.percentile(p_in, [10, 25, 50, 75, 90])
            p_mean = float(p_in.mean())
            rim_mean = float(p_rim.mean())
            halo_mean = float(p_halo.mean())

            nodes.append(
                {
                    "nid": nid,
                    "parent": parent_id,
                    "thr": float(thr),
                    "thr_idx": thr_idx,
                    "ax": ax,
                    "ay": ay,
                    "bbox": (x1, y1, x2, y2),
                    "area": area,
                    "feat": {
                        "thr": float(thr),
                        "area_2048": float(area * scale2),
                        "log_area": float(np.log1p(area * scale2)),
                        "prob_mean": p_mean,
                        "prob_std": float(p_in.std()),
                        "prob_max": float(p_in.max()),
                        "prob_q10": float(q10),
                        "prob_q25": float(q25),
                        "prob_q50": float(q50),
                        "prob_q75": float(q75),
                        "prob_q90": float(q90),
                        "frac_gt_50": float((p_in >= 0.50).mean()),
                        "frac_gt_70": float((p_in >= 0.70).mean()),
                        "frac_gt_85": float((p_in >= 0.85).mean()),
                        "rim_prob_mean": rim_mean,
                        "halo_prob_mean": halo_mean,
                        "halo_prob_ratio": halo_mean / max(p_mean, 1e-4),
                        "prob_drop": rim_mean - halo_mean,
                        "spine_mean": float(sp_in.mean()),
                        "spine_max": float(sp_in.max()),
                        "spine_density": float(sp_in.sum() / np.sqrt(area + 1.0)),
                        "spine_frac_25": float((sp_in >= 0.25).mean()),
                        "n_spine_30": n_spine_30,
                        "dt_max": float(dt.max()),
                        "dt_mean": dt_mean,
                        "dt_std": dt_std,
                        "width_cv": dt_std / max(dt_mean, 1e-4),
                        "bbox_w": float(bw_k),
                        "bbox_h": float(bh_k),
                        "fill_ratio": float(area / max(bw_k * bh_k, 1)),
                        "major_axis": major,
                        "minor_axis": minor,
                        "aspect_ratio": aspect,
                        "eccentricity": ecc,
                        "solidity": solidity,
                        "perimeter": perim,
                        "compactness": compactness,
                        "r_cent": r_cent,
                        "r_min": r_min,
                        "r_max": r_max,
                        "flat_in_mean": float(f_in.mean()),
                        "flat_ring_mean": float(f_ring.mean()),
                        "flat_contrast": float(f_ring.mean() - f_in.mean()),
                        "flat_in_q10": float(np.percentile(f_in, 10)),
                        "diskz_in_mean": float(d_in.mean()),
                        "diskz_contrast": float(d_ring.mean() - d_in.mean()),
                        "loc_contrast": float(l_ring.mean() - l_in.mean()),
                        "n_children": 0.0,
                        "child_area_ratio": 0.0,
                        "parent_area_ratio": 1.0,
                        "depth": 0.0,
                        "img_prob_mass": img_prob_mass,
                        "n_roots_img": 0.0,
                        "inter_std_mean": float(std_in.mean()),
                        "inter_agree_40": float(agr_in.mean()),
                    },
                }
            )
        prev_node_map = cur_node_map

    n_nodes = len(nodes)
    children: List[List[int]] = [[] for _ in range(n_nodes)]
    roots: List[int] = []
    for node in nodes:
        p = node["parent"]
        if p >= 0:
            children[p].append(node["nid"])
        else:
            roots.append(node["nid"])

    # Fill structural tree context features
    for nid, node in enumerate(nodes):
        ch = children[nid]
        node["feat"]["n_children"] = float(len(ch))
        if ch:
            node["feat"]["child_area_ratio"] = float(max(nodes[c]["area"] for c in ch)) / node["area"]
        p = node["parent"]
        if p >= 0:
            node["feat"]["parent_area_ratio"] = float(node["area"]) / nodes[p]["area"]
            node["feat"]["depth"] = nodes[p]["feat"]["depth"] + 1.0
        node["feat"]["n_roots_img"] = float(len(roots))

    rows = [node["feat"] for node in nodes]
    df = pd.DataFrame(rows, columns=list(FEATURE_COLS)) if rows else pd.DataFrame(columns=list(FEATURE_COLS))
    return TreeImage(stem=stem, h=h, w=w, nodes=nodes, children=children, roots=roots, df=df)


def match_tree_targets(
    tree: TreeImage,
    prob: np.ndarray,
    gt_rles_2048_list: Sequence[Sequence[dict]],
    iou_thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean over annotator records of (hit, u=IoU*hit) for each tree node."""
    n_nodes = len(tree.nodes)
    if n_nodes == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    h, w = prob.shape
    scale = 2048 // h
    # Decode and downsample GT masks once for fast node matching
    rec_gts = []
    for rles in gt_rles_2048_list:
        gts = []
        for rle in rles:
            m2048 = mask_utils.decode(rle)
            if scale > 1:
                m = cv2.resize(m2048, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                m = m2048 > 0
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                continue
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            gts.append((x1, y1, x2, y2, m[y1:y2, x1:x2], int(m[y1:y2, x1:x2].sum())))
        rec_gts.append(gts)

    # Reconstruct candidate masks once per threshold level
    node_masks: List[Tuple[int, int, int, int, np.ndarray, int]] = [None] * n_nodes  # type: ignore
    by_thr: Dict[float, List[int]] = {}
    for node in tree.nodes:
        by_thr.setdefault(node["thr"], []).append(node["nid"])
    for thr, nids in by_thr.items():
        bw = (prob >= thr).astype(np.uint8)
        _, lbl = cv2.connectedComponents(bw, connectivity=8)
        for nid in nids:
            node = tree.nodes[nid]
            x1, y1, x2, y2 = node["bbox"]
            k = int(lbl[node["ay"], node["ax"]])
            sub = lbl[y1:y2, x1:x2] == k
            node_masks[nid] = (x1, y1, x2, y2, sub, int(sub.sum()))

    n_rec = max(len(rec_gts), 1)
    y_hit = np.zeros(n_nodes, dtype=np.float32)
    y_u = np.zeros(n_nodes, dtype=np.float32)

    for nid in range(n_nodes):
        cx1, cy1, cx2, cy2, cmask, carea = node_masks[nid]
        hits = 0.0
        us = 0.0
        for gts in rec_gts:
            best_iou = 0.0
            for gx1, gy1, gx2, gy2, gmask, garea in gts:
                ix1 = max(cx1, gx1)
                iy1 = max(cy1, gy1)
                ix2 = min(cx2, gx2)
                iy2 = min(cy2, gy2)
                if ix1 >= ix2 or iy1 >= iy2:
                    continue
                inter = int(
                    (
                        cmask[iy1 - cy1 : iy2 - cy1, ix1 - cx1 : ix2 - cx1]
                        & gmask[iy1 - gy1 : iy2 - gy1, ix1 - gx1 : ix2 - gx1]
                    ).sum()
                )
                if inter == 0:
                    continue
                iou = inter / float(carea + garea - inter)
                if iou > best_iou:
                    best_iou = iou
            if best_iou > iou_thr:
                hits += 1.0
                us += best_iou
        y_hit[nid] = hits / n_rec
        y_u[nid] = us / n_rec
    return y_hit, y_u


def solve_antichain_dp(tree: TreeImage, u_hat: np.ndarray, tau: float) -> List[int]:
    """Exact linear-time maximum weight antichain over component tree with empty-image guard."""
    if len(tree.nodes) == 0:
        return []
    memo_val: Dict[int, float] = {}
    memo_take: Dict[int, bool] = {}

    # Iterative post-order traversal to avoid recursion depth limits
    order: List[int] = []
    stack = list(tree.roots)
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(tree.children[u])

    for u in reversed(order):
        w_self = float(u_hat[u]) - tau
        ch_sum = sum(max(0.0, memo_val[c]) for c in tree.children[u])
        if w_self >= ch_sum and w_self > 0.0:
            memo_val[u] = w_self
            memo_take[u] = True
        else:
            memo_val[u] = ch_sum
            memo_take[u] = False

    selected: List[int] = []
    collect_stack = [r for r in tree.roots if memo_val[r] > 0.0]
    while collect_stack:
        u = collect_stack.pop()
        if memo_take[u]:
            selected.append(u)
        else:
            for c in tree.children[u]:
                if memo_val[c] > 0.0:
                    collect_stack.append(c)

    if not selected:
        selected = [int(np.argmax(u_hat))]
    return selected


def reconstruct_2048_rles(
    tree: TreeImage,
    prob: np.ndarray,
    selected_nids: Sequence[int],
    u_hat: Optional[np.ndarray] = None,
    upsample_prob: bool = True,
    min_area_2048: int = 100,
    prob_2048: Optional[np.ndarray] = None,
    w_2048: float = 0.5,
) -> List[dict]:
    """Convert selected tree node indices into strictly pixel-disjoint 2048x2048 COCO RLEs."""
    if not selected_nids:
        return []
    h, w = prob.shape
    scale = 2048 // h

    ordered = (
        sorted(selected_nids, key=lambda n: float(u_hat[n]), reverse=True)
        if u_hat is not None
        else list(selected_nids)
    )

    by_thr: Dict[float, List[int]] = {}
    for nid in ordered:
        by_thr.setdefault(tree.nodes[nid]["thr"], []).append(nid)

    lbl_by_thr: Dict[float, np.ndarray] = {}
    for thr in by_thr:
        _, lbl = cv2.connectedComponents((prob >= thr).astype(np.uint8), connectivity=8)
        lbl_by_thr[thr] = lbl

    occupied = np.zeros((2048, 2048), dtype=bool)
    buf = np.zeros((2048, 2048), dtype=np.uint8, order="F")
    out_rles: List[dict] = []
    fallback_candidate = None
    k3 = np.ones((3, 3), dtype=np.uint8)

    for nid in ordered:
        node = tree.nodes[nid]
        thr = node["thr"]
        lbl = lbl_by_thr[thr]
        k = int(lbl[node["ay"], node["ax"]])
        x1, y1, x2, y2 = node["bbox"]
        sub_mask = lbl[y1:y2, x1:x2] == k

        if scale == 1:
            m2048_crop = sub_mask
            X1, Y1, X2, Y2 = x1, y1, x2, y2
        elif upsample_prob:
            # Smooth bilinear probability boundary inside padded crop, anchored to the node's component
            px1, py1 = max(0, x1 - 2), max(0, y1 - 2)
            px2, py2 = min(w, x2 + 2), min(h, y2 + 2)
            anchor = cv2.dilate((lbl[py1:py2, px1:px2] == k).astype(np.uint8), k3)
            p_hi = cv2.resize(prob[py1:py2, px1:px2], ((px2 - px1) * scale, (py2 - py1) * scale), interpolation=cv2.INTER_LINEAR)
            a_hi = cv2.resize(anchor, ((px2 - px1) * scale, (py2 - py1) * scale), interpolation=cv2.INTER_NEAREST) > 0
            X1, Y1, X2, Y2 = px1 * scale, py1 * scale, px2 * scale, py2 * scale
            if prob_2048 is not None:
                p_hi = (1.0 - w_2048) * p_hi + w_2048 * prob_2048[Y1:Y2, X1:X2]
            m2048_crop = (p_hi >= thr) & a_hi
        else:
            m2048_crop = cv2.resize(sub_mask.astype(np.uint8), ((x2 - x1) * scale, (y2 - y1) * scale), interpolation=cv2.INTER_NEAREST) > 0
            X1, Y1, X2, Y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

        clean_crop = m2048_crop & (~occupied[Y1:Y2, X1:X2])
        if fallback_candidate is None and clean_crop.any():
            fallback_candidate = (X1, Y1, X2, Y2, clean_crop.copy())
        if int(clean_crop.sum()) < min_area_2048:
            continue
        occupied[Y1:Y2, X1:X2] |= clean_crop
        buf[Y1:Y2, X1:X2] = clean_crop
        out_rles.append(mask_utils.encode(buf))
        buf[Y1:Y2, X1:X2] = 0

    if not out_rles and fallback_candidate is not None:
        X1, Y1, X2, Y2, clean_crop = fallback_candidate
        buf[Y1:Y2, X1:X2] = clean_crop
        out_rles.append(mask_utils.encode(buf))
        buf[Y1:Y2, X1:X2] = 0

    return out_rles


TOPO_COLS: Tuple[str, ...] = (
    "p_diff_parent",
    "sum_child_area_ratio",
    "n_siblings",
    "sibling_area_frac",
    "chain_up",
    "chain_down",
    "subtree_max_thr_gap",
    "subtree_nodes",
    "aspect_diff_parent",
    "spine_diff_parent",
)


def augment_tree_features(tree: TreeImage) -> pd.DataFrame:
    """Append persistent-homology lifetime and parent/sibling delta features to node table."""
    df = tree.df.copy()
    n = len(df)
    if n == 0:
        for c in TOPO_COLS:
            df[c] = np.zeros(0, dtype=np.float32)
        return df

    nodes, ch = tree.nodes, tree.children
    p_diff = np.zeros(n, dtype=np.float32)
    sc_ratio = np.zeros(n, dtype=np.float32)
    n_sib = np.zeros(n, dtype=np.float32)
    sib_frac = np.ones(n, dtype=np.float32)
    c_up = np.zeros(n, dtype=np.float32)
    c_dn = np.zeros(n, dtype=np.float32)
    st_cnt = np.ones(n, dtype=np.float32)
    max_thr = np.array([nd["thr"] for nd in nodes], dtype=np.float32)
    asp_diff = np.zeros(n, dtype=np.float32)
    sp_diff = np.zeros(n, dtype=np.float32)

    for u in range(n - 1, -1, -1):
        for c in ch[u]:
            st_cnt[u] += st_cnt[c]
            if max_thr[c] > max_thr[u]:
                max_thr[u] = max_thr[c]
    st_gap = max_thr - np.array([nd["thr"] for nd in nodes], dtype=np.float32)

    for u, nd in enumerate(nodes):
        p = nd["parent"]
        if p >= 0:
            p_diff[u] = float(nd["feat"]["prob_mean"] - nodes[p]["feat"]["prob_mean"])
            asp_diff[u] = float(nd["feat"]["aspect_ratio"] - nodes[p]["feat"]["aspect_ratio"])
            sp_diff[u] = float(nd["feat"]["spine_density"] - nodes[p]["feat"]["spine_density"])
            sibs = ch[p]
            n_sib[u] = float(len(sibs) - 1)
            sib_frac[u] = float(nd["area"]) / max(sum(nodes[s]["area"] for s in sibs), 1.0)
        if ch[u]:
            sc_ratio[u] = sum(nodes[c]["area"] for c in ch[u]) / max(float(nd["area"]), 1.0)
        cur, k = u, 0
        while nodes[cur]["parent"] >= 0 and len(ch[nodes[cur]["parent"]]) == 1:
            k += 1
            cur = nodes[cur]["parent"]
        c_up[u] = float(k)
        cur, k = u, 0
        while len(ch[cur]) == 1:
            k += 1
            cur = ch[cur][0]
        c_dn[u] = float(k)

    for c, v in zip(
        TOPO_COLS, [p_diff, sc_ratio, n_sib, sib_frac, c_up, c_dn, st_gap, st_cnt, asp_diff, sp_diff]
    ):
        df[c] = v
    return df


def cross_fit_hurdle_lgbm(
    trees: Sequence[TreeImage],
    y_hit_list: Sequence[np.ndarray],
    y_u_list: Sequence[np.ndarray],
    folds: Dict[str, int],
    sample_weight_by_stem: Optional[Dict[str, float]] = None,
    seed: int = 42,
    w_hurdle: float = 1.0,
) -> Tuple[List[np.ndarray], List[Tuple]]:
    """Strict 5-fold cross-fitted Hurdle (P(hit) * E[IoU|hit]) with GBDT+ExtraTrees bagging."""
    all_dfs = []
    for i, tr in enumerate(trees):
        if len(tr.df) == 0:
            continue
        df_i = augment_tree_features(tr)
        df_i["stem"] = tr.stem
        df_i["fold"] = folds[tr.stem]
        df_i["y_hit"] = y_hit_list[i]
        df_i["y_u"] = y_u_list[i]
        df_i["y_iou_cond"] = np.where(y_hit_list[i] > 0, y_u_list[i] / np.maximum(y_hit_list[i], 1e-6), np.nan)
        df_i["weight"] = sample_weight_by_stem[tr.stem] if sample_weight_by_stem else 1.0
        df_i["tree_idx"] = i
        df_i["nid"] = np.arange(len(tr.df))
        all_dfs.append(df_i)

    full_df = pd.concat(all_dfs, ignore_index=True)
    feat_cols = list(FEATURE_COLS) + list(TOPO_COLS)
    oof_u = [np.zeros(len(tr.nodes), dtype=np.float32) for tr in trees]
    models = []

    params_cls = dict(
        objective="binary",
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_estimators=500,
        random_state=seed,
        n_jobs=8,
        verbose=-1,
    )
    params_cls_et = dict(
        params_cls,
        extra_trees=True,
        num_leaves=63,
        colsample_bytree=0.7,
        random_state=seed + 100,
    )
    params_reg = dict(
        objective="regression",
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_estimators=500,
        random_state=seed + 1,
        n_jobs=8,
        verbose=-1,
    )

    for f in sorted(set(folds.values())):
        tr_m = full_df["fold"] != f
        va_m = full_df["fold"] == f

        X_tr, w_tr = full_df.loc[tr_m, feat_cols], full_df.loc[tr_m, "weight"]
        X_va, w_va = full_df.loc[va_m, feat_cols], full_df.loc[va_m, "weight"]
        pos_w = full_df.loc[tr_m, "y_hit"].values
        X_h = pd.concat([X_tr, X_tr], ignore_index=True)
        y_h = np.concatenate([np.ones(len(X_tr), dtype=int), np.zeros(len(X_tr), dtype=int)])
        w_h = np.concatenate([pos_w * w_tr.values, (1.0 - pos_w) * w_tr.values])
        keep_h = w_h > 1e-6

        pos_w_va = full_df.loc[va_m, "y_hit"].values
        X_h_va = pd.concat([X_va, X_va], ignore_index=True)
        y_h_va_bin = np.concatenate([np.ones(len(X_va), dtype=int), np.zeros(len(X_va), dtype=int)])
        w_h_va = np.concatenate([pos_w_va * w_va.values, (1.0 - pos_w_va) * w_va.values])
        keep_h_va = w_h_va > 1e-6

        m_hit = lgb.LGBMClassifier(**params_cls)
        m_hit.fit(
            X_h.iloc[keep_h], y_h[keep_h], sample_weight=w_h[keep_h],
            eval_set=[(X_h_va.iloc[keep_h_va], y_h_va_bin[keep_h_va])], eval_sample_weight=[w_h_va[keep_h_va]],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        m_hit_et = lgb.LGBMClassifier(**params_cls_et)
        m_hit_et.fit(
            X_h.iloc[keep_h], y_h[keep_h], sample_weight=w_h[keep_h],
            eval_set=[(X_h_va.iloc[keep_h_va], y_h_va_bin[keep_h_va])], eval_sample_weight=[w_h_va[keep_h_va]],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )

        tr_pos = tr_m & (full_df["y_hit"] > 0)
        va_pos = va_m & (full_df["y_hit"] > 0)
        m_iou = lgb.LGBMRegressor(**params_reg)
        m_iou.fit(
            full_df.loc[tr_pos, feat_cols],
            full_df.loc[tr_pos, "y_iou_cond"],
            sample_weight=(full_df.loc[tr_pos, "y_hit"] * full_df.loc[tr_pos, "weight"]).values,
            eval_set=[(full_df.loc[va_pos, feat_cols], full_df.loc[va_pos, "y_iou_cond"])],
            eval_sample_weight=[(full_df.loc[va_pos, "y_hit"] * full_df.loc[va_pos, "weight"]).values],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )

        p_hit = 0.5 * m_hit.predict_proba(X_va)[:, 1] + 0.5 * m_hit_et.predict_proba(X_va)[:, 1]
        p_iou = np.clip(m_iou.predict(X_va), 0.5, 1.0)
        if w_hurdle < 1.0:
            m_dir = lgb.LGBMRegressor(**params_reg)
            m_dir.fit(
                X_tr, full_df.loc[tr_m, "y_u"], sample_weight=w_tr.values,
                eval_set=[(X_va, full_df.loc[va_m, "y_u"])], eval_sample_weight=[w_va.values],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            p_dir = np.clip(m_dir.predict(X_va), 0.0, 1.0)
            u_pred = w_hurdle * (p_hit * p_iou) + (1.0 - w_hurdle) * p_dir
            b_dir = m_dir.booster_
        else:
            u_pred = p_hit * p_iou
            b_dir = None

        va_rows = full_df.loc[va_m, ["tree_idx", "nid"]].values
        for idx_row, (t_idx, nid) in enumerate(va_rows):
            oof_u[int(t_idx)][int(nid)] = float(u_pred[idx_row])

        models.append((m_hit.booster_, m_hit_et.booster_, m_iou.booster_, b_dir, w_hurdle))

    return oof_u, models


def predict_hurdle_lgbm(
    trees: Sequence[TreeImage],
    models: Sequence[Tuple],
) -> List[np.ndarray]:
    feat_cols = list(FEATURE_COLS) + list(TOPO_COLS)
    preds = []
    for tr in trees:
        if len(tr.df) == 0:
            preds.append(np.zeros(0, dtype=np.float32))
            continue
        X = augment_tree_features(tr)[feat_cols]
        u_acc = np.zeros(len(X), dtype=np.float64)
        for mod in models:
            b_hit, b_hit_et, b_iou, b_dir, wh = mod
            p_hit = 0.5 * b_hit.predict(X) + 0.5 * b_hit_et.predict(X)
            p_iou = np.clip(b_iou.predict(X), 0.5, 1.0)
            if b_dir is not None and wh < 1.0:
                p_dir = np.clip(b_dir.predict(X), 0.0, 1.0)
                u_acc += wh * (p_hit * p_iou) + (1.0 - wh) * p_dir
            else:
                u_acc += p_hit * p_iou
        preds.append((u_acc / len(models)).astype(np.float32))
    return preds
