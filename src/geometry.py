"""Solar disk fitting and radial limb-darkening flat-field normalization."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def fit_solar_disk(img_u8: np.ndarray, iters: int = 3) -> Tuple[float, float, float]:
    """Fit solar disk circle (cx, cy, R) in pixel coordinates of img_u8 (H, W)."""
    h, w = img_u8.shape
    blur = cv2.GaussianBlur(img_u8, (0, 0), 3.0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((thresh > 0).astype(np.uint8), connectivity=8)
    if n < 2:
        return (w - 1) / 2.0, (h - 1) / 2.0, min(h, w) * 0.455
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return (w - 1) / 2.0, (h - 1) / 2.0, min(h, w) * 0.455
    pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    keep = np.ones(len(pts), dtype=bool)
    cx, cy, r = (w - 1) / 2.0, (h - 1) / 2.0, min(h, w) * 0.455
    for _ in range(iters):
        p = pts[keep]
        if len(p) < 12:
            break
        A = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))])
        b = p[:, 0] ** 2 + p[:, 1] ** 2
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1.0)))
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        res = np.abs(dist - r)
        mad = np.median(res)
        keep = res <= max(3.0 * 1.4826 * mad, 4.0)
    if not (0.35 * min(h, w) <= r <= 0.52 * min(h, w)):
        (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    return float(cx), float(cy), float(r)


def radius_map(shape: Tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return np.hypot(xx - cx, yy - cy) / max(r, 1.0)


def normalize_disk_and_flatfield(
    img_u8: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    disk_frac: float = 0.985,
    n_annuli: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (disk_z_u8, flat_field_u8, local_contrast_u8) at work resolution.

    - disk_z: (img - disk_mean) / disk_std mapped to [0, 255] via clip([-4, 4])
    - flat_field: img / radial_median_profile(rho), background-leveled, mapped [0.5, 1.5] -> [0, 255]
    - local_contrast: (flat - local_mean) / local_std inside disk, mapped [-4, 4] -> [0, 255]
    """
    h, w = img_u8.shape
    img = img_u8.astype(np.float32)
    rho = radius_map((h, w), cx, cy, r)
    valid = rho < disk_frac

    # 1. Disk z-score
    v_vals = img[valid]
    d_mean = float(v_vals.mean()) if v_vals.size else 128.0
    d_std = max(float(v_vals.std()), 1.0) if v_vals.size else 30.0
    disk_z = np.where(valid, (img - d_mean) / d_std, 0.0)

    # 2. Radial median limb-darkening profile
    bins = np.clip((rho * (n_annuli / disk_frac)).astype(np.int32), 0, n_annuli - 1)
    prof = np.ones(n_annuli, dtype=np.float32) * d_mean
    vb = bins[valid]
    vi = img[valid]
    order = np.argsort(vb)
    vb_s = vb[order]
    vi_s = vi[order]
    splits = np.flatnonzero(np.diff(vb_s)) + 1
    starts = np.concatenate(([0], splits))
    ends = np.concatenate((splits, [len(vb_s)]))
    for s, e in zip(starts, ends):
        if e > s:
            prof[vb_s[s]] = float(np.median(vi_s[s:e]))
    # Smooth radial profile slightly
    prof_s = cv2.GaussianBlur(prof.reshape(1, -1), (1, 9), 2.0).ravel()
    prof_map = prof_s[bins]
    flat = np.where(valid, img / np.maximum(prof_map, 8.0), 1.0)

    # Smooth 2D background correction (clouds/gradients) ignoring dark filaments
    sig = max(0.08 * r, 16.0)
    ksize = int(round(sig * 4)) | 1
    bg_w = valid.astype(np.float32)
    # First pass
    num = cv2.GaussianBlur(flat * bg_w, (0, 0), sig)
    den = cv2.GaussianBlur(bg_w, (0, 0), sig)
    bg1 = num / np.maximum(den, 1e-4)
    res1 = flat - bg1
    mad = float(np.median(np.abs(res1[valid]))) * 1.4826 if valid.any() else 0.05
    clean_w = (valid & (np.abs(res1) < max(2.5 * mad, 0.04))).astype(np.float32)
    num2 = cv2.GaussianBlur(flat * clean_w, (0, 0), sig)
    den2 = cv2.GaussianBlur(clean_w, (0, 0), sig)
    bg2 = np.where(den2 > 1e-3, num2 / np.maximum(den2, 1e-4), bg1)
    flat_corr = np.where(valid, flat / np.maximum(bg2, 0.2), 1.0)

    # 3. Local contrast z-score
    mu_sig = max(r * 0.032, 8.0)
    sd_sig = max(r * 0.064, 16.0)
    w_f = valid.astype(np.float32)
    l_den = np.maximum(cv2.GaussianBlur(w_f, (0, 0), mu_sig), 1e-4)
    l_mu = cv2.GaussianBlur(flat_corr * w_f, (0, 0), mu_sig) / l_den
    s_den = np.maximum(cv2.GaussianBlur(w_f, (0, 0), sd_sig), 1e-4)
    l_var = cv2.GaussianBlur(((flat_corr - l_mu) ** 2) * w_f, (0, 0), sd_sig) / s_den
    l_sd = np.maximum(np.sqrt(np.maximum(l_var, 0.0)), 0.015)
    loc_z = np.where(valid, (flat_corr - l_mu) / l_sd, 0.0)

    disk_z_u8 = np.clip((disk_z + 4.0) * (255.0 / 8.0), 0, 255).round().astype(np.uint8)
    flat_u8 = np.clip((flat_corr - 0.5) * 255.0, 0, 255).round().astype(np.uint8)
    loc_u8 = np.clip((loc_z + 4.0) * (255.0 / 8.0), 0, 255).round().astype(np.uint8)
    return disk_z_u8, flat_u8, loc_u8
