"""5-fold D-FINE-seg-S 2048x2048 dataset preparation, training, and 4x D4 TTA inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from src import data, paths

VIEWS_4 = [(0, False), (2, False), (0, True), (2, True)]
VIEWS_8 = [(k, f) for f in (False, True) for k in range(4)]


def dihedral(x: torch.Tensor, k: int, flip: bool, inverse: bool = False) -> torch.Tensor:
    if not inverse:
        x = x.flip(-1) if flip else x
        return torch.rot90(x, k, (-2, -1))
    x = torch.rot90(x, -k, (-2, -1))
    return x.flip(-1) if flip else x


@torch.no_grad()
def infer_probs_u8(net, img_bgr: np.ndarray, views: Sequence[tuple[int, bool]], amp_dtype: torch.dtype = torch.bfloat16) -> np.ndarray:
    x, _, (orig,) = net._prepare_inputs(img_bgr)
    dev_type = x.device.type if hasattr(x, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    amp_ok = dev_type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    prob = 0
    for k, f in views:
        with torch.amp.autocast(dev_type, dtype=amp_dtype, enabled=amp_ok):
            logits = net._predict(dihedral(x, k, f))["sem_seg_logits"].float()
        prob = prob + dihedral(logits.softmax(1)[:, 1:2], k, f, inverse=True)
    prob = torch.nn.functional.interpolate(prob / len(views), size=orig, mode="bilinear", align_corners=False)
    return (prob[0, 0] * 255.0).clamp(0, 255).round().byte().cpu().numpy()


def prepare_dfine_dataset(size: int = 2048) -> None:
    """Create 5-fold D-FINE sem_seg directory tree matching canonical stratified grouped folds."""
    coco_raw = json.loads((paths.DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    meta = json.loads((paths.CACHE_1024 / "meta.json").read_text())
    folds = meta["folds"]
    train_img_dir = paths.DATA_ROOT / "train/train_images"

    segs = defaultdict(list)
    for a in coco_raw["annotations"]:
        segs[a["image_id"]].extend(a["segmentation"])

    out = paths.DFINE_WORK / "data_sem_5f"
    sh_img = out / "shared_images"
    sh_lbl = out / "shared_labels"
    sh_img.mkdir(parents=True, exist_ok=True)
    sh_lbl.mkdir(parents=True, exist_ok=True)

    view_to_fold = {}
    for im in coco_raw["images"]:
        view_id = str(im["id"])
        src_img = (train_img_dir / im["file_name"]).resolve()
        stem = Path(im["file_name"]).stem
        view_to_fold[view_id] = int(folds[stem])

        dst_img = sh_img / f"{view_id}.jpeg"
        if not dst_img.exists():
            if dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(src_img)

        dst_lbl = sh_lbl / f"{view_id}.png"
        if not dst_lbl.exists():
            poly_list = segs.get(im["id"], [])
            if poly_list:
                rle = mask_utils.merge(mask_utils.frPyObjects(poly_list, size, size))
                mask = mask_utils.decode(rle).astype(np.uint8)
            else:
                mask = np.zeros((size, size), dtype=np.uint8)
            cv2.imwrite(str(dst_lbl), mask)

    for f in range(5):
        fdir = out / f"fold{f}"
        fdir.mkdir(parents=True, exist_ok=True)
        for sub, target in (("images", sh_img), ("labels", sh_lbl)):
            p = fdir / sub
            if not p.exists():
                if p.is_symlink():
                    p.unlink()
                p.symlink_to(target)
        tr_files = sorted(f"{vid}.jpeg" for vid, vf in view_to_fold.items() if vf != f)
        va_files = sorted(f"{vid}.jpeg" for vid, vf in view_to_fold.items() if vf == f)
        (fdir / "train.csv").write_text("\n".join(tr_files) + "\n")
        (fdir / "val.csv").write_text("\n".join(va_files) + "\n")

    cfg_dst = paths.DFINE_WORK / "config.yaml"
    cfg_dst.write_text((paths.CONFIGS_DIR / "dfine_sem_seg.yaml").read_text())
    print(f"[D-FINE] Prepared 5-fold dataset ({len(view_to_fold)} views) under {out}")


def _export_fold_outputs(
    fold: int,
    ckpt: Path,
    out_run_dir: Path,
    views: Sequence[tuple[int, bool]],
    export_oof: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> None:
    from dfine_seg import load_model

    test_dir = paths.DATA_ROOT / "test/test_images"
    test_imgs = sorted(test_dir.glob("*.jpeg"))
    if not test_imgs:
        raise FileNotFoundError(f"No test images in {test_dir}")

    f_prob_2048 = out_run_dir / f"fold{fold}" / "test_prob_2048"
    f_prob_1024 = out_run_dir / f"fold{fold}" / "test_prob"
    f_prob_2048.mkdir(parents=True, exist_ok=True)
    f_prob_1024.mkdir(parents=True, exist_ok=True)

    net = load_model(str(ckpt), input_height=2048, input_width=2048)
    t0 = time.time()
    for p in test_imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        p_u8 = infer_probs_u8(net, img, views, amp_dtype=amp_dtype)
        p_u16 = p_u8.astype(np.uint16) * 257
        cv2.imwrite(str(f_prob_2048 / f"{p.stem}.png"), p_u16)
        p_1024 = cv2.resize(p_u8.astype(np.float32) / 255.0, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16)
        np.save(f_prob_1024 / f"{p.stem}.npy", p_1024)

    if export_oof:
        meta = json.loads((paths.CACHE_1024 / "meta.json").read_text())
        folds_map = meta["folds"]
        train_dir = paths.DATA_ROOT / "train/train_images"
        oof_2048 = out_run_dir / "oof_prob_2048"
        oof_1024 = out_run_dir / "oof_prob"
        oof_2048.mkdir(parents=True, exist_ok=True)
        oof_1024.mkdir(parents=True, exist_ok=True)
        val_stems = sorted(k for k, v in folds_map.items() if int(v) == fold)
        for s in val_stems:
            img = cv2.imread(str(train_dir / f"{s}.jpeg"), cv2.IMREAD_COLOR)
            p_u8 = infer_probs_u8(net, img, views, amp_dtype=amp_dtype)
            cv2.imwrite(str(oof_2048 / f"{s}.png"), p_u8.astype(np.uint16) * 257)
            p_1024 = cv2.resize(p_u8.astype(np.float32) / 255.0, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16)
            np.save(oof_1024 / f"{s}.npy", p_1024)

    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[dfine_s_2048] Fold {fold} exported {len(test_imgs)} test maps ({len(views)}x-TTA) in {time.time()-t0:.1f}s", flush=True)


def infer_test_folds(folds: Sequence[int], run_name: str = "dfine_s_2048", tta: int = 4, amp_dtype_str: str = "auto") -> None:
    """Run D-FINE-seg-S inference across folds from saved fold{f}/model.pt checkpoints."""
    views = VIEWS_8 if tta == 8 else (VIEWS_4 if tta == 4 else VIEWS_4[:1])
    out_run_dir = paths.RUNS_DIR / run_name
    for f in folds:
        ckpt = out_run_dir / f"fold{f}" / "model.pt"
        if not ckpt.is_file():
            ckpt = paths.DFINE_WORK / "runs" / f"sem_s_f{f}" / "model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing D-FINE checkpoint for fold {f} under {out_run_dir}")
        if amp_dtype_str == "auto":
            dt = torch.float16 if f in (2, 3) else torch.bfloat16
        elif amp_dtype_str == "fp16":
            dt = torch.float16
        elif amp_dtype_str == "fp32":
            dt = torch.float32
        else:
            dt = torch.bfloat16
        _export_fold_outputs(f, ckpt, out_run_dir, views=views, export_oof=False, amp_dtype=dt)


def train_folds(
    folds: Sequence[int],
    run_name: str = "dfine_s_2048",
    epochs: int = 30,
    max_walltime_min: int = 315,
    tta: int = 4,
    dfine_bin: str = "",
) -> None:
    prepare_dfine_dataset(size=2048)
    views = VIEWS_8 if tta == 8 else (VIEWS_4 if tta == 4 else VIEWS_4[:1])
    bin_path = dfine_bin or shutil.which("dfine") or str(Path(sys.executable).parent / "dfine")
    out_run_dir = paths.RUNS_DIR / run_name
    out_run_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir = paths.DATA_ROOT / "test/test_images"

    env = os.environ.copy()
    env["DFINE_SEG_CONFIG_DIR"] = str(paths.DFINE_WORK)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for f in folds:
        work_run = paths.DFINE_WORK / "runs" / f"sem_s_f{f}"
        cmd = [
            bin_path, "train",
            f"exp_name=sem_s_f{f}",
            f"train.root={paths.DFINE_WORK}",
            f"train.data_path={paths.DFINE_WORK}/data_sem_5f/fold{f}",
            f"train.path_to_test_data={test_img_dir}",
            f"train.path_to_save={work_run}",
            "train.to_visualize_eval=False",
            "train.debug_img_processing=False",
            "train.batch_size=1",
            "train.b_accum_steps=2",
            "train.num_workers=4",
            "train.amp_dtype=bfloat16",
            f"train.epochs={epochs}",
            f"train.max_walltime_min={max_walltime_min}",
        ]
        subprocess.check_call(cmd, cwd=str(paths.DFINE_WORK), env=env)
        ckpt = work_run / "model.pt"
        dst_fold = out_run_dir / f"fold{f}"
        dst_fold.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, dst_fold / "model.pt")
        _export_fold_outputs(f, dst_fold / "model.pt", out_run_dir, views=views, export_oof=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("prep", "train", "infer-test"), default="infer-test")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--run-name", default="dfine_s_2048")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-walltime-min", type=int, default=315)
    ap.add_argument("--tta", type=int, default=4, choices=(1, 4, 8))
    ap.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    ap.add_argument("--dfine-bin", default="")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if args.mode == "prep":
        prepare_dfine_dataset()
    elif args.mode == "train":
        train_folds(
            folds=folds,
            run_name=args.run_name,
            epochs=args.epochs,
            max_walltime_min=args.max_walltime_min,
            tta=args.tta,
            dfine_bin=args.dfine_bin,
        )
    else:
        infer_test_folds(folds=folds, run_name=args.run_name, tta=args.tta, amp_dtype_str=args.amp_dtype)


if __name__ == "__main__":
    main()
