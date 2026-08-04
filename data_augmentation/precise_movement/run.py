#!/usr/bin/env python3
"""End-to-end Precise Movement: sample, rollout, filter, instruct, render.

Per MP3D scene Habitat-Sim is started once.  Inside the sim we repeatedly:

  1. sample a random navigable (start, goal) pair (geodesic + height checks)
  2. rollout along the shortest path while writing per-step JPGs
  3. keep only trajectories that pass step/turn/segment filters
  4. build a rule-based precise-movement instruction + LLaVA annotations

Resumable via existing ``llava_annotations.json`` / ``output.json``.

Run after ``conda activate navspace39`` and ``source data_augmentation/env_shim.sh``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    ACTION_NAMES,
    build_precise_instruction,
    build_sim,
    count_turns,
    load_config,
    load_json,
    merge_action_segments,
    pipeline_output_dir,
    resolve_mp3d_scene,
    sample_random_navigable_pair,
    save_json,
    validate_precise_instruction,
)

PRECISE_EPISODE_ID_BASE = 9_000_000
GT_STEP = 6
_ACT_ID_TO_LANG = {1: "Move forward", 2: "Turn left", 3: "Turn right", 0: "Stop"}


def _list_mp3d_scenes(mp3d_root: Path) -> list[str]:
    scans = []
    for child in sorted(mp3d_root.iterdir()):
        if not child.is_dir():
            continue
        glb = child / f"{child.name}.glb"
        navmesh = child / f"{child.name}.navmesh"
        if glb.exists() and navmesh.exists():
            scans.append(f"mp3d/{child.name}/{child.name}.glb")
    return scans


def _multi_step_gt(gt_actions: list[int], step_idx: int, gt_step: int = GT_STEP) -> str:
    lang = []
    for i in range(gt_step):
        idx = step_idx + i
        if idx >= len(gt_actions):
            lang.append("Stop")
        else:
            lang.append(_ACT_ID_TO_LANG.get(gt_actions[idx], "Stop"))
    return ",".join(lang)


def _build_prompt(instruction: str, gt_step: int = GT_STEP) -> str:
    return (
        f" You are navigating in an indoor environment given the instruction: "
        f"{instruction};\n"
        f"            You are given the observation history of previous steps "
        f"you have taken;\n"
        f"            You should:\n"
        f"            1) evaluate the history to decide which step of "
        f"instruction you are at.\n"
        f"            2) Predict actions for the next {gt_step} steps to "
        f"follow up the given instruction until you reach the goal;\n"
        f"            Notice that:\n"
        f"            1) You can only choose from the following four actions: "
        f"Move forward, Turn left, Turn right, Stop;\n"
        f"            2) Move forward means to move 0.25 meters straight "
        f"ahead, and turning left or right is a 30-degree turn.\n"
        f"            3) If you believe you have reached the target or caught "
        f"in obstacles, you should choose the stop action.\n"
        f"            ----\n"
        f"            Starting below, you should strictly follow this format:\n"
        f"            Final Answer: Your predicted actions for the next "
        f"{gt_step} steps"
    )


def _rollout_and_save(
    sim,
    episode: dict,
    out_episode_dir: Path,
    *,
    max_steps: int,
    goal_radius: float,
) -> tuple[list[int], int]:
    """Greedy follower rollout that writes per-step JPGs and returns actions."""
    import habitat_sim
    from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
    from habitat_sim.utils.common import quat_from_coeffs

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode["start_position"], dtype=np.float32)
    if episode.get("start_rotation"):
        agent_state.rotation = quat_from_coeffs(episode["start_rotation"])
    sim.get_agent(0).set_state(agent_state)

    follower = GreedyGeodesicFollower(
        sim.pathfinder, sim.get_agent(0), goal_radius=goal_radius,
    )
    goal = episode["goal_position"]

    actions: list[int] = [-1]
    num_frames = 0
    out_episode_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    aid = 0
    while step < max_steps:
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(out_episode_dir / f"{num_frames:03d}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        num_frames += 1

        try:
            raw_action = follower.next_action_along(goal)
        except Exception:
            return actions, num_frames

        if raw_action is None:
            aid = 0
        elif isinstance(raw_action, str):
            aid = {"move_forward": 1, "turn_left": 2, "turn_right": 3, "stop": 0}.get(raw_action, 0)
        else:
            aid = int(raw_action)
        if aid == 0:
            break
        action_name = ACTION_NAMES.get(aid)
        if action_name is None:
            break
        sim.step(action_name)
        actions.append(aid)
        step += 1

    return actions, num_frames


def _load_resume_state(
    llava_anno_path: Path,
    summary_path: Path,
) -> tuple[list[dict], set[str], list[dict], int]:
    existing_entries: list[dict] = []
    existing_eptags: set[str] = set()
    summary_records: list[dict] = []
    next_eid = PRECISE_EPISODE_ID_BASE

    if llava_anno_path.exists():
        with llava_anno_path.open("r", encoding="utf-8") as file:
            existing_entries = json.load(file)
        for entry in existing_entries:
            vid = entry.get("video", "")
            if "/" in vid:
                existing_eptags.add(vid.split("/", 1)[1])

    if summary_path.exists():
        payload = load_json(summary_path)
        summary_records = list(payload.get("records", []))
        if summary_records:
            next_eid = max(int(r["episode_id"]) for r in summary_records) + 1

    return existing_entries, existing_eptags, summary_records, next_eid


def _save_checkpoint(
    llava_anno_path: Path,
    summary_path: Path,
    existing_entries: list[dict],
    new_entries: list[dict],
    summary_records: list[dict],
) -> None:
    combined = existing_entries + new_entries
    with llava_anno_path.open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=2, ensure_ascii=False)
    save_json(
        {
            "pipeline": "precise_movement",
            "count": len(summary_records),
            "records": summary_records,
        },
        summary_path,
    )


def run_precise_movement(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pcfg = cfg["precise_movement"]
    hcfg = cfg["habitat"]

    max_eps = args.max_episodes or int(pcfg["max_episodes"])
    min_geo = float(pcfg["min_geodesic"])
    max_geo = float(pcfg["max_geodesic"])
    max_hdiff = float(pcfg["max_height_diff"])
    min_turns = int(pcfg["min_turns"])
    min_steps = int(args.min_steps if args.min_steps is not None else pcfg.get("min_steps", 16))
    max_steps_filter = int(args.max_steps if args.max_steps is not None else pcfg.get("max_steps", 45))
    forward_step = float(hcfg["forward_step"])
    turn_angle = float(hcfg["turn_angle"])
    rollout_max_steps = int(hcfg.get("max_steps", 500))
    goal_radius = float(hcfg.get("goal_radius", 0.5))

    mp3d_root = Path(cfg["paths"]["mp3d_root"])
    scans = _list_mp3d_scenes(mp3d_root)
    if not scans:
        print(f"No MP3D scans found under {mp3d_root}", file=sys.stderr)
        return 1

    per_scene = args.per_scene
    if per_scene is None:
        per_scene = max(1, (max_eps + len(scans) - 1) // len(scans) + 2)

    pipeline_dir = pipeline_output_dir(cfg, "precise_movement")
    output_root = Path(args.output_root) if args.output_root else ROOT / "snav_data/aug_mix/precise_movement"
    output_root.mkdir(parents=True, exist_ok=True)
    llava_anno_path = output_root / "llava_annotations.json"
    summary_path = pipeline_dir / "output.json"

    if args.restart:
        if llava_anno_path.exists():
            llava_anno_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    existing_entries, existing_eptags, summary_records, next_eid = _load_resume_state(
        llava_anno_path, summary_path,
    )
    if existing_eptags:
        print(
            f"Resuming: {len(existing_entries)} LLaVA entries / "
            f"{len(existing_eptags)} episodes; next episode_id={next_eid}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(scans)

    new_entries: list[dict] = []
    rendered = len(existing_eptags)
    skipped_short = 0
    skipped_long = 0
    skipped_turns = 0
    skipped_other = 0
    sample_failures = 0

    for scene_id in scans:
        if rendered >= max_eps:
            break

        try:
            scene_path = resolve_mp3d_scene(scene_id, mp3d_root)
        except FileNotFoundError as exc:
            print(f"Skip scene {scene_id}: {exc}")
            continue

        try:
            sim = build_sim(scene_path, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"Skip scene {scene_id} (sim init failed): {exc}")
            continue

        scan = Path(scene_id).parent.name or scene_id.split("/")[0]
        success_for_scene = 0
        attempts = max(per_scene * 15, 30)

        try:
            while (
                success_for_scene < per_scene
                and rendered < max_eps
                and attempts > 0
            ):
                attempts -= 1
                pair = sample_random_navigable_pair(
                    sim,
                    min_geodesic=min_geo,
                    max_geodesic=max_geo,
                    max_height_diff=max_hdiff,
                    rng=rng,
                )
                if pair is None:
                    sample_failures += 1
                    continue

                ep_id = next_eid
                ep_tag = f"{scan}_precise_{ep_id:06d}"
                if ep_tag in existing_eptags:
                    next_eid += 1
                    continue

                episode = {
                    "episode_id": ep_id,
                    "scene_id": scene_id,
                    "start_position": pair["start_position"],
                    "start_rotation": pair["start_rotation"],
                    "goal_position": pair["goal_position"],
                    "geodesic_distance": pair["geodesic_distance"],
                }
                ep_frame_dir = output_root / ep_tag

                actions, num_frames = _rollout_and_save(
                    sim,
                    episode,
                    ep_frame_dir,
                    max_steps=rollout_max_steps,
                    goal_radius=goal_radius,
                )

                if num_frames != len(actions):
                    shutil.rmtree(ep_frame_dir, ignore_errors=True)
                    skipped_other += 1
                    continue

                steps_done = len(actions) - 1
                if steps_done < min_steps:
                    shutil.rmtree(ep_frame_dir, ignore_errors=True)
                    skipped_short += 1
                    continue
                if steps_done > max_steps_filter:
                    shutil.rmtree(ep_frame_dir, ignore_errors=True)
                    skipped_long += 1
                    continue
                if count_turns(actions) < min_turns:
                    shutil.rmtree(ep_frame_dir, ignore_errors=True)
                    skipped_turns += 1
                    continue

                segments = merge_action_segments(
                    actions,
                    forward_step=forward_step,
                    turn_angle=turn_angle,
                )
                if len(segments) < 2:
                    shutil.rmtree(ep_frame_dir, ignore_errors=True)
                    skipped_other += 1
                    continue

                instruction = build_precise_instruction(segments, rng=rng)
                if not validate_precise_instruction(instruction, segments):
                    shutil.rmtree(ep_frame_dir, ignore_errors=True)
                    skipped_other += 1
                    continue

                gt_actions = actions[1:] + [0]
                video_rel = f"{args.video_subdir}/{ep_tag}"
                for step_idx in range(num_frames):
                    gt_act_str = _multi_step_gt(gt_actions, step_idx)
                    prompt = _build_prompt(instruction)
                    new_entries.append({
                        "id": f"{args.video_subdir}_{ep_id:06d}_step_{step_idx}",
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{prompt}"},
                            {"from": "gpt", "value": f"Final Answer: {gt_act_str}"},
                        ],
                        "video": video_rel,
                        "video_nframes": step_idx + 1,
                    })

                summary_records.append({
                    "episode_id": ep_id,
                    "scene_id": scene_id,
                    "instruction": instruction,
                    "segments": segments,
                    "num_steps": steps_done,
                    "geodesic_distance": pair["geodesic_distance"],
                    "ep_tag": ep_tag,
                })
                existing_eptags.add(ep_tag)
                next_eid += 1
                success_for_scene += 1
                rendered += 1
                print(
                    f"[{scan}] ep {ep_id}: {steps_done} steps, {len(segments)} segs "
                    f"(scene {success_for_scene}/{per_scene}, total {rendered}/{max_eps})"
                )
                print(f"  -> {instruction}")
        finally:
            sim.close()

        _save_checkpoint(
            llava_anno_path,
            summary_path,
            existing_entries,
            new_entries,
            summary_records,
        )
        print(f"[{scene_id}] scene done: {success_for_scene} kept")

    print(
        f"Done. rendered={rendered}, "
        f"skip(short={skipped_short},long={skipped_long},"
        f"turns={skipped_turns},other={skipped_other},"
        f"sample_fail={sample_failures}). "
        f"LLaVA annotations -> {llava_anno_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--output-root", default=None,
                        help="Per-step JPGs + llava_annotations.json "
                             "(default: <repo>/snav_data/aug_mix/precise_movement)")
    parser.add_argument("--video-subdir", default="precise_movement",
                        help="Subdir used inside the LLaVA 'video' field.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Target number of successful trajectories.")
    parser.add_argument("--per-scene", type=int, default=None,
                        help="Max successes per scene before moving on.")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="Drop trajectories shorter than this (default: config precise_movement.min_steps).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Drop trajectories longer than this (default: config precise_movement.max_steps).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true",
                        help="Delete existing output and start fresh.")
    args = parser.parse_args()
    return run_precise_movement(args)


if __name__ == "__main__":
    raise SystemExit(main())
