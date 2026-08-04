#!/usr/bin/env python3
"""Deprecated wrapper — use ``precise_movement/run.py`` instead."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.precise_movement.run import main  # noqa: E402

if __name__ == "__main__":
    warnings.warn(
        "2_render.py is deprecated; use: python data_augmentation/precise_movement/run.py",
        DeprecationWarning,
        stacklevel=1,
    )
    print("[deprecated] Forwarding to precise_movement/run.py", file=sys.stderr)
    raise SystemExit(main())
