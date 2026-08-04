#!/usr/bin/env python3
"""Filter R2R episodes with large start/goal height difference."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    height_diff,
    load_config,
    load_r2r_episodes,
    pipeline_output_dir,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--min-height-diff", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["vertical_perception"]
    min_diff = float(args.min_height_diff or vcfg["min_height_diff"])
    max_eps = args.max_episodes or int(vcfg["max_episodes"])

    episodes = load_r2r_episodes(cfg["paths"]["r2r_train"])
    candidates = []
    for ep in episodes:
        diff = height_diff(ep)
        if diff < min_diff:
            continue
        candidates.append({**ep, "height_diff": diff})

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:max_eps]

    out_dir = pipeline_output_dir(cfg, "vertical_perception")
    save_json(
        {"pipeline": "vertical_perception", "count": len(selected), "episodes": selected},
        out_dir / "candidates.json",
    )
    print(f"Saved {len(selected)} / {len(candidates)} candidates -> {out_dir / 'candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
