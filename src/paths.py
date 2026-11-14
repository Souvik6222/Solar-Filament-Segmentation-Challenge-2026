"""Repository-relative and environment-configurable filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SOLAR_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("SOLAR_DATA_ROOT", ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"))
CACHE_1024 = Path(os.environ.get("SOLAR_CACHE_1024", ROOT / "data" / "cache_1024"))
CACHE_2048 = Path(os.environ.get("SOLAR_CACHE_2048", ROOT / "data" / "cache_2048"))
RUNS_DIR = Path(os.environ.get("SOLAR_RUNS_DIR", ROOT / "runs"))
SUB_DIR = Path(os.environ.get("SOLAR_SUB_DIR", ROOT / "sub"))
S2_CACHE_ROOT = Path(os.environ.get("SOLAR_S2_CACHE", ROOT / "data" / "s2_cache"))
DFINE_WORK = Path(os.environ.get("SOLAR_DFINE_WORK", RUNS_DIR / "dfine_work"))
CONFIGS_DIR = ROOT / "configs"
