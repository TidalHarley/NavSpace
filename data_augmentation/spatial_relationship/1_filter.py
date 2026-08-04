#!/usr/bin/env python3
"""Filter R2R train instructions with spatial-relationship keywords (regex only).

No paraphrase / Qwen rewrite — the original R2R instruction is kept as-is.
Outputs ``candidates.json``, ``output.json``, and ``custom_instructions.json``
for merge + render.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    load_config,
    load_r2r_episodes,
    match_spatial_instruction,
    pipeline_output_dir,
    save_json,
    spatial_match_keyword,
    to_custom_instructions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scfg = cfg["spatial_relationship"]
    max_eps = args.max_episodes or int(scfg["max_episodes"])

    episodes = load_r2r_episodes(cfg["paths"]["r2r_train"])
    candidates = []
    for ep in episodes:
        keyword = spatial_match_keyword(ep["instruction_text"])
        if keyword is None:
            continue
        candidates.append({**ep, "matched_keyword": keyword})

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:max_eps]

    records = [
        {
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "original_instruction": ep["instruction_text"],
            "instruction": ep["instruction_text"],
            "matched_keyword": ep["matched_keyword"],
            "pipeline": "spatial_relationship",
        }
        for ep in selected
    ]

    out_dir = pipeline_output_dir(cfg, "spatial_relationship")
    save_json(
        {
            "pipeline": "spatial_relationship",
            "count": len(selected),
            "total_matched": len(candidates),
            "episodes": selected,
        },
        out_dir / "candidates.json",
    )
    save_json(
        {"pipeline": "spatial_relationship", "count": len(records), "records": records},
        out_dir / "output.json",
    )
    custom = to_custom_instructions(records)
    save_json(custom, out_dir / "custom_instructions.json")

    print(
        f"Saved {len(selected)} / {len(candidates)} spatial episodes "
        f"(from {len(episodes)} R2R train) -> {out_dir / 'custom_instructions.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
