"""Shared path resolution for Stage-A collect/build scripts.

Override with env vars (recommended):

  NAVIGATION_ROOT   root that holds mp3d_scenes / R2R_VLNCE_* / RxR_VLNCE_*
  TRAIN_DATA_ROOT   where frames + JSON are written (default: <repo>/train_data)
  MP3D_ROOT         Matterport3D scenes root (default: $NAVIGATION_ROOT/mp3d_scenes)
  R2RCE_ROOT        R2R-CE dataset root (default: $NAVIGATION_ROOT/R2R_VLNCE_v1-3)
  RXRCE_ROOT        RxR-CE dataset root (default: $NAVIGATION_ROOT/RxR_VLNCE_v0)
"""

from __future__ import annotations

import os
from pathlib import Path

STAGE_A_DIR = Path(__file__).resolve().parent
SNAV_TRAINING_DIR = STAGE_A_DIR.parent
REPO_ROOT = SNAV_TRAINING_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default.expanduser().resolve()


def resolve_navigation_root() -> Path:
    env = os.environ.get("NAVIGATION_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    for candidate in (REPO_ROOT.parent, REPO_ROOT):
        if (candidate / "mp3d_scenes").is_dir() or (candidate / "R2R_VLNCE_v1-3").is_dir():
            return candidate.resolve()
    return REPO_ROOT.resolve()


NAVIGATION_ROOT = resolve_navigation_root()
# Prefer <repo>/train_data so configs/*.yaml relative json_path entries resolve.
TRAIN_DATA_ROOT = _env_path("TRAIN_DATA_ROOT", REPO_ROOT / "train_data")
MP3D_ROOT = _env_path("MP3D_ROOT", NAVIGATION_ROOT / "mp3d_scenes")
R2R_ROOT = _env_path(
    "R2RCE_ROOT",
    Path(os.environ.get("R2R_ROOT", str(NAVIGATION_ROOT / "R2R_VLNCE_v1-3"))),
)
RXR_ROOT = _env_path(
    "RXRCE_ROOT",
    Path(os.environ.get("RXR_ROOT", str(NAVIGATION_ROOT / "RxR_VLNCE_v0"))),
)
