"""Parallel runner for Component Tree + Cross-Fitted Hurdle LightGBM + Antichain DP."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src import data, eval_oof_breakdown, export, metrics, paths, tree_dp

CACHE_DIR = paths.CACHE_1024


def _load_blended_prob(
    stem: str, run_dirs: Sequence[Path], weights: Sequence[float], is_test: bool, n_folds: int = 5
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    p_acc = np.zeros((1024, 1024), dtype=np.float32)
    s_acc = np.zeros((1024, 1024), dtype=np.float32)
    w_tot = 0.0
    model_probs: List[np.ndarray] = []

    for rdir, w in zip(run_dirs, weights):
        if is_test:
            fold_ps = []
            fold_ss = []
            for f in range(n_folds):
                fp = rdir / f"fold{f}" / "test_prob" / f"{stem}.npy"
                fs = rdir / f"fold{f}" / "test_spine" / f"{stem}.npy"
                if fp.exists():
                    fold_ps.append(np.load(fp).astype(np.float32))
                if fs.exists():
                    fold_ss.append(np.load(fs).astype(np.float32))
            if not fold_ps:
                continue
            pm = np.mean(fold_ps, axis=0)
            sm = np.mean(fold_ss, axis=0) if fold_ss else np.zeros((1024, 1024), dtype=np.float32)
        else:
            fp = rdir / "oof_prob" / f"{stem}.npy"
            fs = rdir / "oof_spine" / f"{stem}.npy"
            if not fp.exists():
                continue
            pm = np.load(fp).astype(np.float32)
            sm = np.load(fs).astype(np.float32) if fs.exists() else np.zeros((1024, 1024), dtype=np.float32)

        p_acc += w * pm
        s_acc += w * sm
        w_tot += w
        model_probs.append(pm)

    if w_tot == 0:
        raise FileNotFoundError(f"No predictions found for {stem}")
    return p_acc / w_tot, s_acc / w_tot, (model_probs if len(model_probs) > 1 else None)


def _load_prob_2048(
    stem: str, tile_runs: Sequence[Path], tile_weights: Sequence[float], is_test: bool, n_folds: int = 5
) -> Optional[np.ndarray]:
    if not tile_runs:
        return None
    acc_total = None
    w_tot = 0.0
    for trun, tw in zip(tile_runs, tile_weights):
        if not is_test:
            p = trun / "oof_prob_2048" / f"{stem}.png"
            if not p.exists():
                continue
            pm = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        else:
            acc = None
            cnt = 0
            for f in range(n_folds):
                p = trun / f"fold{f}" / "test_prob_2048" / f"{stem}.png"
                if p.exists():
                    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
                    acc = arr if acc is None else (acc + arr)
                    cnt += 1
            if acc is None:
                continue
            pm = acc / cnt
        acc_total = (tw * pm) if acc_total is None else (acc_total + tw * pm)
        w_tot += tw
    return (acc_total / w_tot) if (acc_total is not None and w_tot > 0) else None


def _build_train_worker(args):
    cv2.setNumThreads(1)
    stem, run_dir_strs, weights, geom, gt_list = args
    run_dirs = [Path(s) for s in run_dir_strs]
    prob, spine, mprobs = _load_blended_prob(stem, run_dirs, weights, is_test=False)
    img_3ch = np.load(CACHE_DIR / "img" / f"{stem}.npy")
    tr = tree_dp.build_tree_for_image(stem, prob, img_3ch, tuple(geom), spine=spine, model_probs=mprobs)
    yh, yu = tree_dp.match_tree_targets(tr, prob, gt_list)
    return tr, yh, yu, prob


def _build_test_worker(args):
    cv2.setNumThreads(1)
    stem, run_dir_strs, weights, geom = args
    run_dirs = [Path(s) for s in run_dir_strs]
    prob, spine, mprobs = _load_blended_prob(stem, run_dirs, weights, is_test=True)
    img_3ch = np.load(CACHE_DIR / "img" / f"{stem}.npy")
    tr = tree_dp.build_tree_for_image(stem, prob, img_3ch, tuple(geom), spine=spine, model_probs=mprobs)
    return tr, prob


def run_test_only(
    bundle_path: Path,
    out_name: str,
    workers: int = max(1, min(24, (os.cpu_count() or 8) - 4)),
    min_area_2048: int = 120,
    upsample_prob: bool = True,
    export_csv: bool = False,
) -> Path:
    """Run Tree-DP test inference using pre-fitted LightGBM hurdle models from a saved bundle."""
    bundle = pickle.loads(bundle_path.read_bytes())
    lgb_models = bundle["lgb_models"]
    best_tau = float(bundle["best_tau"])
    run_names = list(bundle["run_names"])
    weights = [float(x) for x in bundle["weights"]]
    tile_names = list(bundle.get("tile_names") or [])
    tile_weights = [float(x) for x in (bundle.get("tile_weights") or [])]
    w_2048 = float(bundle.get("w_2048", 0.92))

    run_dirs = [paths.RUNS_DIR / n for n in run_names]
    run_dir_strs = [str(p) for p in run_dirs]
    tile_runs = [paths.RUNS_DIR / n for n in tile_names]

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    geom_map: Dict[str, List[float]] = meta["geom_1024"]
    test_stems = sorted(meta["test_stems"])
    t_jobs = [(s, run_dir_strs, weights, geom_map[s]) for s in test_stems]

    n_workers = min(workers, os.cpu_count() or 8)
    test_trees = []
    test_probs: Dict[str, np.ndarray] = {}
    t0 = time.time()
    print(f"[TreeDP test-only] Building Component Trees for {len(test_stems)} test stems...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        for s, (tr, prob) in zip(test_stems, ex.map(_build_test_worker, t_jobs, chunksize=4)):
            test_trees.append(tr)
            test_probs[s] = prob

    test_u = tree_dp.predict_hurdle_lgbm(test_trees, lgb_models)
    pred_test: Dict[str, List[dict]] = {}
    for tr, u_hat in zip(test_trees, test_u):
        sel = tree_dp.solve_antichain_dp(tr, u_hat, tau=best_tau)
        p2048 = _load_prob_2048(tr.stem, tile_runs, tile_weights, is_test=True)
        pred_test[tr.stem] = tree_dp.reconstruct_2048_rles(
            tr,
            test_probs[tr.stem],
            sel,
            u_hat=u_hat,
            upsample_prob=upsample_prob,
            min_area_2048=min_area_2048,
            prob_2048=p2048,
            w_2048=w_2048,
        )

    out_dir = paths.RUNS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "test_pred_rles.pkl"
    pkl_path.write_bytes(pickle.dumps(pred_test))
    print(f"[TreeDP test-only] Saved {pkl_path} (tau={best_tau:.2f}) in {time.time()-t0:.1f}s", flush=True)
    if export_csv:
        paths.SUB_DIR.mkdir(parents=True, exist_ok=True)
        export.write_and_verify_submission(pred_test, test_stems, paths.SUB_DIR / f"{out_name}.csv")
    return pkl_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="Comma-separated run dirs under runs/")
    ap.add_argument("--weights", default="", help="Comma-separated blend weights (default equal)")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--tile2048-run", default="", help="Optional comma-separated 2048px tile runs for boundary refinement")
    ap.add_argument("--tile2048-weights", default="", help="Comma-separated blend weights for --tile2048-run")
    ap.add_argument("--w-2048", type=float, default=0.5)
    ap.add_argument("--w-hurdle", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 8) - 4)))
    ap.add_argument("--export-sub", action="store_true")
    ap.add_argument("--weighted-records", action="store_true")
    ap.add_argument("--upsample-prob", action="store_true", default=True)
    ap.add_argument("--min-area-2048", type=int, default=120)
    ap.add_argument("--test-only", action="store_true", help="Run test-only inference using saved tree_dp_bundle.pkl")
    ap.add_argument("--bundle", default="", help="Optional path to tree_dp_bundle.pkl for --test-only")
    args = ap.parse_args()

    if args.test_only:
        bundle_p = Path(args.bundle) if args.bundle else (paths.RUNS_DIR / args.out_name / "tree_dp_bundle.pkl")
        run_test_only(
            bundle_path=bundle_p,
            out_name=args.out_name,
            workers=args.workers,
            min_area_2048=args.min_area_2048,
            upsample_prob=args.upsample_prob,
            export_csv=args.export_sub,
        )
        return

    if not args.runs:
        raise ValueError("--runs is required unless --test-only is specified")

    t0 = time.time()
    run_names = [x.strip() for x in args.runs.split(",") if x.strip()]
    run_dirs = [paths.RUNS_DIR / n for n in run_names]
    run_dir_strs = [str(p) for p in run_dirs]
    weights = [float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(run_dirs)
    tile_names = [x.strip() for x in args.tile2048_run.split(",") if x.strip()]
    tile_runs = [paths.RUNS_DIR / n for n in tile_names]
    tile_weights = (
        [float(x) for x in args.tile2048_weights.split(",")] if args.tile2048_weights else [1.0] * len(tile_runs)
    )

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds_map: Dict[str, int] = meta["folds"]
    geom_map: Dict[str, List[float]] = meta["geom_1024"]
    coco = data.load_coco_grouped()

    oof_stems = sorted(
        s for s in meta["train_stems"]
        if all((rd / "oof_prob" / f"{s}.npy").exists() for rd in run_dirs)
    )
    print(f"Building Component Trees in parallel for {len(oof_stems)} OOF stems from {run_names}...", flush=True)

    jobs = [
        (s, run_dir_strs, weights, geom_map[s], [coco["gt_rles_2048"][r] for r in coco["recs_by_stem"][s]])
        for s in oof_stems
    ]
    trees: List[tree_dp.TreeImage] = []
    probs_oof: Dict[str, np.ndarray] = {}
    y_hit_list: List[np.ndarray] = []
    y_u_list: List[np.ndarray] = []
    sample_weights: Dict[str, float] = {}

    workers = min(args.workers, os.cpu_count() or 8)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, (tr, yh, yu, prob) in zip(oof_stems, ex.map(_build_train_worker, jobs, chunksize=4)):
            trees.append(tr)
            probs_oof[s] = prob
            y_hit_list.append(yh)
            y_u_list.append(yu)
            sample_weights[s] = float(len(coco["recs_by_stem"][s])) if args.weighted_records else 1.0

    print(f"Trees built ({sum(len(t.nodes) for t in trees)} nodes) in {time.time()-t0:.1f}s", flush=True)

    active_folds = {s: folds_map[s] for s in oof_stems}
    oof_gt_recs = {r: coco["gt_rles_2048"][r] for s in oof_stems for r in coco["recs_by_stem"][s]}
    if len(set(active_folds.values())) < 2:
        print("Only 1 outer fold detected — using internal 5-fold split for Hurdle LGBM validation.", flush=True)
        active_folds = {s: i % 5 for i, s in enumerate(oof_stems)}

    oof_u, lgb_models = tree_dp.cross_fit_hurdle_lgbm(
        trees, y_hit_list, y_u_list, active_folds, sample_weight_by_stem=sample_weights, w_hurdle=args.w_hurdle
    )

    p2048_oof = {tr.stem: _load_prob_2048(tr.stem, tile_runs, tile_weights, is_test=False) for tr in trees}
    best_tau = 0.24
    best_res = None
    best_pred = None
    for tau in (0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36):
        pred_by_stem = {}
        for tr, u_hat in zip(trees, oof_u):
            sel = tree_dp.solve_antichain_dp(tr, u_hat, tau=tau)
            pred_by_stem[tr.stem] = tree_dp.reconstruct_2048_rles(
                tr, probs_oof[tr.stem], sel, u_hat=u_hat, upsample_prob=args.upsample_prob,
                min_area_2048=args.min_area_2048, prob_2048=p2048_oof[tr.stem], w_2048=args.w_2048,
            )
        res, _ = metrics.evaluate_pq(oof_gt_recs, coco["rec_to_stem"], pred_by_stem)
        print(f"  tau={tau:.2f} -> {res}", flush=True)
        if best_res is None or res.pq > best_res.pq:
            best_res = res
            best_tau = tau
            best_pred = pred_by_stem

    print(f"\nBEST Tree DP OOF PQ = {best_res.pq:.4f} @ tau={best_tau:.2f}", flush=True)
    if best_pred is not None:
        eval_oof_breakdown.analyze_oof(best_pred, oof_gt_recs, coco)

    out_dir = paths.RUNS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tree_dp_bundle.pkl", "wb") as f:
        pickle.dump(
            {
                "lgb_models": lgb_models,
                "best_tau": best_tau,
                "run_names": run_names,
                "weights": weights,
                "tile_names": tile_names,
                "tile_weights": tile_weights,
                "w_2048": args.w_2048,
                "oof_pred_rles": best_pred,
            },
            f,
        )

    if args.export_sub:
        test_stems = sorted(meta["test_stems"])
        t_jobs = [(s, run_dir_strs, weights, geom_map[s]) for s in test_stems]
        test_trees = []
        test_probs = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for s, (tr, prob) in zip(test_stems, ex.map(_build_test_worker, t_jobs, chunksize=4)):
                test_trees.append(tr)
                test_probs[s] = prob

        test_u = tree_dp.predict_hurdle_lgbm(test_trees, lgb_models)
        pred_test: Dict[str, List[dict]] = {}
        for tr, u_hat in zip(test_trees, test_u):
            sel = tree_dp.solve_antichain_dp(tr, u_hat, tau=best_tau)
            p2048 = _load_prob_2048(tr.stem, tile_runs, tile_weights, is_test=True)
            pred_test[tr.stem] = tree_dp.reconstruct_2048_rles(
                tr, test_probs[tr.stem], sel, u_hat=u_hat, upsample_prob=args.upsample_prob,
                min_area_2048=args.min_area_2048, prob_2048=p2048, w_2048=args.w_2048,
            )
        with open(out_dir / "test_pred_rles.pkl", "wb") as f:
            pickle.dump(pred_test, f)
        paths.SUB_DIR.mkdir(parents=True, exist_ok=True)
        sub_path = paths.SUB_DIR / f"{args.out_name}.csv"
        export.write_and_verify_submission(pred_test, test_stems, sub_path)


if __name__ == "__main__":
    main()
