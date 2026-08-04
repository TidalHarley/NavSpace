#!/usr/bin/env python3
"""Verify cross-floor R2R episodes via Habitat semantic API.

Pipeline (per scene, single Habitat sim instance):

  1. Load scene → query ``sim.semantic_scene`` for the total number of
     semantic ``levels``.  If the scene has only one level, drop all
     candidates from that scene.
  2. For each candidate episode:
     * resolve the semantic ``level_index`` of the start and goal
       positions through ``sim.semantic_scene.get_regions_for_point``
       (with a nearest ``floor_height`` fallback);
     * keep only episodes whose start / end levels differ;
     * run the GreedyGeodesicFollower along
       the shortest path and ask Qwen-VL whether stairs are visible in
       at least N sampled frames — useful when you suspect the semantic
       regions miss a hallway transit.


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    annotate_vertical_floors_from_sim,
    build_sim,
    checkpoint_append,
    checkpoint_iter,
    checkpoint_load_ids,
    detect_stairs_in_frames,
    get_scene_total_levels,
    load_config,
    load_json,
    pipeline_output_dir,
    resolve_mp3d_scene,
    rollout_episode_with_frames,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--input", default=None,
                        help="Override candidates.json path.")
    parser.add_argument("--enable-stairs-check", action="store_true",
                        help="Also run Qwen-VL stair detection on rollout frames "
                             "(default OFF — semantic API already gives a "
                             "trustworthy floor signal).")
    parser.add_argument("--min-stairs-frames", type=int, default=None,
                        help="Only used when --enable-stairs-check is set.")
    parser.add_argument("--require-rollout", action="store_true",
                        help="Also require the Greedy follower to reach the "
                             "goal (otherwise we trust the floor labels alone).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Stop after this many verified episodes total.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing checkpoint and start over.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["vertical_perception"]
    min_stairs = args.min_stairs_frames or int(vcfg.get("min_stairs_frames", 3))
    out_dir = pipeline_output_dir(cfg, "vertical_perception")
    input_path = Path(args.input) if args.input else out_dir / "candidates.json"
    payload = load_json(input_path)
    episodes = payload["episodes"]

    checkpoint_path = out_dir / "verified.jsonl"
    final_path = out_dir / "verified.json"
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()

    done_ids = checkpoint_load_ids(checkpoint_path)
    target = args.max_episodes if args.max_episodes else None
    print(f"Resuming with {len(done_ids)} episodes already verified."
          + (f" target={target}" if target else ""))

    by_scene: dict[str, list[dict]] = {}
    for ep in episodes:
        if ep["episode_id"] in done_ids:
            continue
        by_scene.setdefault(ep["scene_id"], []).append(ep)

    mp3d_root = cfg["paths"]["mp3d_root"]
    stats = {"verified": len(done_ids), "skip_same_level": 0,
             "skip_no_level": 0, "skip_single_level": 0,
             "skip_rollout": 0, "skip_stairs": 0}

    for scene_id, scene_eps in by_scene.items():
        if target is not None and stats["verified"] >= target:
            break
        try:
            scene_path = resolve_mp3d_scene(scene_id, mp3d_root)
        except FileNotFoundError as exc:
            print(f"Skip scene {scene_id}: {exc}")
            continue

        need_sensors = bool(args.enable_stairs_check or args.require_rollout)
        try:
            sim = build_sim(scene_path, cfg, with_sensors=need_sensors)
        except Exception as exc:  # noqa: BLE001
            print(f"Skip scene {scene_id} (sim init failed): {exc}")
            continue

        try:
            total_levels = get_scene_total_levels(sim)
            if total_levels <= 1:
                stats["skip_single_level"] += len(scene_eps)
                print(f"Skip scene {scene_id} ({total_levels} semantic level)")
                continue
            print(f"[{scene_id}] semantic levels = {total_levels}, "
                  f"candidates = {len(scene_eps)}")

            for ep in scene_eps:
                if target is not None and stats["verified"] >= target:
                    break

                floor_meta = annotate_vertical_floors_from_sim(
                    sim, ep["start_position"], ep["goal_position"],
                )
                if floor_meta is None:
                    stats["skip_no_level"] += 1
                    continue
                if floor_meta["direction"] == "same":
                    stats["skip_same_level"] += 1
                    continue

                stairs_info = None
                rollout_ok = True
                if args.require_rollout or args.enable_stairs_check:
                    result = rollout_episode_with_frames(
                        sim,
                        ep,
                        cfg,
                        frame_stride=int(vcfg.get("frame_stride", 5)),
                        max_frames=int(vcfg.get("max_frames", 12)),
                    )
                    rollout_ok = bool(result.get("success", False))
                    if args.require_rollout and not rollout_ok:
                        stats["skip_rollout"] += 1
                        continue

                    if args.enable_stairs_check and rollout_ok:
                        try:
                            stairs_info = detect_stairs_in_frames(
                                cfg, result.get("frames", []),
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(f"Stair detect failed for ep {ep['episode_id']}: {exc}")
                            stats["skip_stairs"] += 1
                            continue
                        if stairs_info["stairs_frame_count"] < min_stairs:
                            stats["skip_stairs"] += 1
                            continue

                record = {
                    "episode_id": ep["episode_id"],
                    "scene_id": ep["scene_id"],
                    "start_position": ep["start_position"],
                    "start_rotation": ep["start_rotation"],
                    "goal_position": ep["goal_position"],
                    "original_instruction": ep["instruction_text"],
                    "height_diff": ep.get("height_diff", floor_meta["height_diff"]),
                    "floor_meta": floor_meta,
                }
                if stairs_info is not None:
                    record["stairs_info"] = stairs_info
                checkpoint_append(checkpoint_path, record)
                done_ids.add(ep["episode_id"])
                stats["verified"] += 1
                print(
                    f"  ep {ep['episode_id']}: "
                    f"floors {floor_meta['start_level_index']+1} → "
                    f"{floor_meta['end_level_index']+1}/{floor_meta['total_levels']} "
                    f"({floor_meta['direction']}, "
                    f"Δh={floor_meta['height_diff']:.2f} m) "
                    f"[total={stats['verified']}]"
                )
        finally:
            sim.close()

    records = list(checkpoint_iter(checkpoint_path))
    save_json(
        {"pipeline": "vertical_perception", "count": len(records), "records": records},
        final_path,
    )
    print(
        f"Done. verified={stats['verified']} "
        f"(same_level={stats['skip_same_level']}, "
        f"no_level={stats['skip_no_level']}, "
        f"single_level={stats['skip_single_level']}, "
        f"rollout={stats['skip_rollout']}, "
        f"stairs={stats['skip_stairs']}) "
        f"-> {final_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
