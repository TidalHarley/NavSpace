#!/usr/bin/env python3
"""Deprecated — spatial relationship now uses regex filter only (1_filter.py).

Paraphrase is no longer part of this pipeline.  Run::

    python data_augmentation/spatial_relationship/1_filter.py

"""

from __future__ import annotations

import sys
import warnings

if __name__ == "__main__":
    warnings.warn(
        "2_paraphrase.py is deprecated; spatial_relationship is filter-only. "
        "Use: python data_augmentation/spatial_relationship/1_filter.py",
        DeprecationWarning,
        stacklevel=1,
    )
    print(__doc__, file=sys.stderr)
    raise SystemExit(0)
