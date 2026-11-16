"""End-to-end reproduction script for the final competition submission (58424451 / s2_hd_mitb2_d3_2)."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import download_weights
from src import paths


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    print(f"    completed in {time.time() - t0:.1f}s", flush=True)


def _has_all_test_outputs(run_dir: Path, test_stems: list[str], need_2048: bool) -> bool:
    for f in range(5):
        p1024 = run_dir / f"fold{f}" / "test_prob"
        if not p1024.is_dir() or len(list(p1024.glob("*.npy"))) < len(test_stems):
            return False
        if need_2048:
            p2048 = run_dir / f"fold{f}" / "test_prob_2048"
            if not p2048.is_dir() or len(list(p2048.glob("*.png"))) < len(test_stems):
                return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce final MAGFiLO 2026 submission CSV.")
    ap.add_argument("--data-root", default=str(paths.DATA_ROOT), help="Path to MAGFiLO_1.0_Kaggle_2026 dataset root")
    ap.add_argument("--runs-dir", default=str(paths.RUNS_DIR), help="Directory storing model checkpoints and predictions")
    ap.add_argument("--out-csv", default=str(ROOT / "submission.csv"), help="Output path for submission CSV")
    default_workers = max(1, min(24, (os.cpu_count() or 8) - 4))
    ap.add_argument("--workers", type=int, default=default_workers, help="Parallel CPU workers for geometry & tree DP")
    ap.add_argument("--dfine-python", default=os.environ.get("SOLAR_DFINE_PYTHON", sys.executable), help="Python executable with dfine-seg installed")
    ap.add_argument("--skip-download", action="store_true", help="Do not download checkpoints if missing")
    ap.add_argument("--force", action="store_true", help="Re-run inference even if intermediate fold outputs already exist")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    out_csv = Path(args.out_csv).resolve()

    test_img_dir = data_root / "test/test_images"
    test_imgs = sorted(test_img_dir.glob("*.jpeg"))
    if not test_imgs:
        raise FileNotFoundError(f"Expected test .jpeg files under {test_img_dir}")
    test_stems = [p.stem for p in test_imgs]
    print(f"Found {len(test_stems)} test images at {test_img_dir}", flush=True)

    env = os.environ.copy()
    env["SOLAR_REPO_ROOT"] = str(ROOT)
    env["SOLAR_DATA_ROOT"] = str(data_root)
    env["SOLAR_RUNS_DIR"] = str(runs_dir)

    if not args.skip_download:
        download_weights.ensure_weights(runs_dir=runs_dir)

    py = sys.executable
    dfine_py = args.dfine_python

    # 1. Solar disk fitting & 3-channel radial limb-darkening normalization (1024 & 2048 test cache)
    _run([py, "-u", "-m", "src.cache", "--test-only", "--workers", str(args.workers)], env)
    _run([py, "-u", "-m", "src.cache_2048", "--test-only", "--workers", str(args.workers)], env)

    # 2. Stage-1 5-fold ensemble test inference (4 heterogeneous architectures)
    if args.force or not _has_all_test_outputs(runs_dir / "unet_cnx_t", test_stems, need_2048=False):
        _run(
            [
                py, "-u", "-m", "src.train_seg",
                "--arch", "Unet", "--encoder", "tu-convnext_tiny",
                "--run-name", "unet_cnx_t", "--folds", "0",
                "--n-tta", "8", "--channels-last", "--infer-test",
            ],
            env,
        )
        _run(
            [
                py, "-u", "-m", "src.train_seg",
                "--arch", "Unet", "--encoder", "tu-convnext_tiny",
                "--run-name", "unet_cnx_t", "--folds", "1,2,3,4",
                "--n-tta", "4", "--channels-last", "--infer-test",
            ],
            env,
        )
    else:
        print("[Skip] unet_cnx_t test outputs present.", flush=True)

    if args.force or not _has_all_test_outputs(runs_dir / "tile2048_unet_r34", test_stems, need_2048=True):
        _run(
            [
                py, "-u", "-m", "src.train_tile2048",
                "--arch", "Unet", "--encoder", "resnet34",
                "--run-name", "tile2048_unet_r34", "--folds", "0,1,2,3,4",
                "--n-tta", "2", "--infer-test",
            ],
            env,
        )
    else:
        print("[Skip] tile2048_unet_r34 test outputs present.", flush=True)

    if args.force or not _has_all_test_outputs(runs_dir / "tile512_hrnet32", test_stems, need_2048=True):
        _run(
            [
                py, "-u", "-m", "src.train_tile512",
                "--arch", "Unet", "--encoder", "tu-hrnet_w32",
                "--run-name", "tile512_hrnet32", "--folds", "0,1,2,3,4",
                "--crop-size", "512", "--zoom", "1", "--overlap", "0.25",
                "--pos-weight", "2.0", "--n-tta", "4", "--channels-last", "--infer-test",
            ],
            env,
        )
    else:
        print("[Skip] tile512_hrnet32 test outputs present.", flush=True)

    if args.force or not _has_all_test_outputs(runs_dir / "dfine_s_2048", test_stems, need_2048=True):
        _run(
            [
                dfine_py, "-u", "-m", "src.train_dfine",
                "--mode", "infer-test", "--run-name", "dfine_s_2048",
                "--folds", "0,1,2,3,4", "--tta", "4",
            ],
            env,
        )
    else:
        print("[Skip] dfine_s_2048 test outputs present.", flush=True)

    # 3. Component Tree DAG + Hurdle LightGBM scoring + Antichain DP (dp_hd_sharp)
    dp_pkl = runs_dir / "dp_hd_sharp" / "test_pred_rles.pkl"
    if args.force or not dp_pkl.is_file():
        _run(
            [
                py, "-u", "-m", "src.run_tree_dp",
                "--test-only", "--out-name", "dp_hd_sharp",
                "--bundle", str(runs_dir / "dp_hd_sharp" / "tree_dp_bundle.pkl"),
                "--workers", str(args.workers), "--min-area-2048", "120",
            ],
            env,
        )
    else:
        print("[Skip] dp_hd_sharp test_pred_rles.pkl present.", flush=True)

    # 4. Stage-2 rotated 128x384 crop refiner (Unet:mit_b2) + combined confidence filtering
    s2_cmd = [
        py, "-u", "-m", "src.stage2_refine",
        "--test-only",
        "--dp-name", "dp_hd_sharp",
        "--out-name", "s2_hd_mitb2",
        "--arch", "Unet", "--encoder", "mit_b2",
        "--batch-size", "24", "--n-tta", "4",
        "--alpha", "1.0", "--tau", "0.50", "--comb-thr", "0.48",
        "--min-area-2048", "120", "--workers", str(args.workers),
        "--out-csv", str(out_csv),
    ]
    if args.force:
        s2_cmd.append("--force-cache")
    _run(s2_cmd, env)

    print(f"\nDone! Wrote {out_csv} (sha256={_sha256(out_csv)})", flush=True)


if __name__ == "__main__":
    main()
