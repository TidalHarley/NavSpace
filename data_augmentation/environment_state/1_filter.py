#!/usr/bin/env python3
"""Step 1 — Sample R2R train episodes for environment-state augmentation.

Filters R2R episodes by geodesic distance and samples ``max_episodes``
trajectories.  The output schema matches ``vertical_perception/verified.json``
(records with ``episode_id`` / ``original_instruction`` / ``scene_id`` …) so
that ``scripts/build_passthrough_instructions.py`` and the renderer can
consume it directly without further adaptation.

Outputs:
    data_augmentation/outputs/environment_state/candidates.json
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
    pipeline_output_dir,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Override config.environment_state.max_episodes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["environment_state"]
    max_eps = args.max_episodes or int(ecfg["max_episodes"])
    min_geo = float(ecfg["min_geodesic"])
    max_geo = float(ecfg["max_geodesic"])

    episodes = load_r2r_episodes(cfg["paths"]["r2r_train"])
    print(f"Loaded {len(episodes)} R2R train episodes; "
          f"filter geodesic in [{min_geo:.1f}, {max_geo:.1f}]")

    candidates = []
    for ep in episodes:
        geo = float(ep.get("info", {}).get("geodesic_distance", 0.0))
        if geo < min_geo or geo > max_geo:
            continue
        candidates.append({
            "episode_id": int(ep["episode_id"]),
            "trajectory_id": ep.get("trajectory_id"),
            "scene_id": ep["scene_id"],
            "start_position": ep["start_position"],
            "start_rotation": ep["start_rotation"],
            "goals": ep["goals"],
            "goal_position": ep["goal_position"],
            "original_instruction": ep["instruction_text"],
            "geodesic_distance": geo,
        })

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:max_eps]

    out_dir = pipeline_output_dir(cfg, "environment_state")
    out_path = out_dir / "candidates.json"
    save_json({
        "pipeline": "environment_state",
        "count": len(selected),
        "records": selected,
    }, out_path)
    print(f"Saved {len(selected)} candidates -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
