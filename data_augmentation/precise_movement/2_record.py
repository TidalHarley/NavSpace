#!/usr/bin/env python3
"""Habitat rollout and action-segment recording for precise movement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    build_sim,
    count_turns,
    load_config,
    load_json,
    merge_action_segments,
    pipeline_output_dir,
    resolve_mp3d_scene,
    rollout_episode,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--input", default=None, help="candidates.json from 1_sample.py")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pcfg = cfg["precise_movement"]
    out_dir = pipeline_output_dir(cfg, "precise_movement")
    input_path = Path(args.input) if args.input else out_dir / "candidates.json"
    payload = load_json(input_path)
    episodes = payload["episodes"]

    by_scene: dict[str, list[dict]] = {}
    for ep in episodes:
        by_scene.setdefault(ep["scene_id"], []).append(ep)

    records = []
    min_turns = int(pcfg["min_turns"])
    mp3d_root = cfg["paths"]["mp3d_root"]

    for scene_id, scene_eps in by_scene.items():
        try:
            scene_path = resolve_mp3d_scene(scene_id, mp3d_root)
        except FileNotFoundError as exc:
            print(f"Skip scene {scene_id}: {exc}")
            continue

        sim = build_sim(scene_path, cfg)
        try:
            for ep in scene_eps:
                result = rollout_episode(sim, ep, cfg, capture_rgb=False)
                if not result["success"]:
                    continue
                actions = result["actions"]
                if count_turns(actions) < min_turns:
                    continue
                segments = merge_action_segments(
                    actions,
                    forward_step=float(cfg["habitat"]["forward_step"]),
                    turn_angle=float(cfg["habitat"]["turn_angle"]),
                )
                if len(segments) < 2:
                    continue
                records.append(
                    {
                        "episode_id": ep["episode_id"],
                        "scene_id": ep["scene_id"],
                        "start_position": ep["start_position"],
                        "start_rotation": ep["start_rotation"],
                        "goal_position": ep["goal_position"],
                        "original_instruction": ep["instruction_text"],
                        "actions": actions,
                        "segments": segments,
                        "num_steps": result["num_steps"],
                    }
                )
        finally:
            sim.close()

    out_path = out_dir / "actions.json"
    save_json({"pipeline": "precise_movement", "count": len(records), "records": records}, out_path)
    print(f"Recorded {len(records)} episodes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
