"""Download and verify pre-trained Stage-1, Tree-DP, and Stage-2 checkpoints from GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import paths

DEFAULT_BASE_URL = "https://github.com/Souvik6222/Solar-Filament-Segmentation-Challenge-2026/releases/download/v1.0"
MANIFEST_PATH = ROOT / "configs" / "weights_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"Downloading {url} -> {dst.name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "solar-filament-repro/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -1
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total > 0:
                pct = int(done * 100 / total)
                if pct >= last_pct + 10:
                    last_pct = pct
                    print(f"  {dst.name}: {pct}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", flush=True)
    tmp.replace(dst)


def ensure_weights(
    base_url: str = DEFAULT_BASE_URL,
    manifest_path: Path = MANIFEST_PATH,
    runs_dir: Path | None = None,
    force: bool = False,
    keep_archives: bool = False,
) -> None:
    runs_dir = Path(runs_dir) if runs_dir is not None else paths.RUNS_DIR
    manifest = json.loads(manifest_path.read_text())

    required_markers = [
        runs_dir / "unet_cnx_t/fold4/best.pt",
        runs_dir / "tile2048_unet_r34/fold4/best.pt",
        runs_dir / "tile512_hrnet32/fold4/best.pt",
        runs_dir / "dfine_s_2048/fold4/model.pt",
        runs_dir / "dp_hd_sharp/tree_dp_bundle.pkl",
        runs_dir / "s2_hd_mitb2/fold4_best.pt",
        runs_dir / "s2_hd_mitb2/config.json",
    ]
    if not force and all(p.is_file() for p in required_markers):
        print(f"All required checkpoints already present under {runs_dir}", flush=True)
        return

    cache_dl = runs_dir / "_archives"
    cache_dl.mkdir(parents=True, exist_ok=True)
    for item in manifest["archives"]:
        fname = item["filename"]
        expected_sha = item["sha256"]
        arc = cache_dl / fname
        if force or not arc.is_file() or (expected_sha and _sha256(arc) != expected_sha):
            url = f"{base_url.rstrip('/')}/{fname}"
            _download_with_progress(url, arc)
            actual_sha = _sha256(arc)
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(f"Checksum mismatch for {fname}: got {actual_sha}, expected {expected_sha}")
        print(f"Extracting {fname} into {runs_dir}...", flush=True)
        with tarfile.open(arc, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise RuntimeError(f"Unsafe tar member path: {member.name}")
                rel = Path(member.name)
                if rel.parts and rel.parts[0] == "runs":
                    rel = Path(*rel.parts[1:])
                if not str(rel) or str(rel) == ".":
                    continue
                target = runs_dir / rel
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isreg():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src_f, open(target, "wb") as dst_f:
                        dst_f.write(src_f.read())
        if not keep_archives and arc.exists():
            arc.unlink()

    missing = [str(p) for p in required_markers if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoints still missing after extraction: {missing}")
    print("Weights download and verification complete.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()
    ensure_weights(base_url=args.base_url, force=args.force, keep_archives=args.keep_archives)


if __name__ == "__main__":
    main()
