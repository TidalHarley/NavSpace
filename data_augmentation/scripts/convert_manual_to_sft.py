#!/usr/bin/env python3
"""Convert human-annotated `trajectories.json` (produced by
``annotation_pipeline/websocket_annotation_server.py``) into the
``aug_mix`` layout consumed by ``snav_training/dataset_snav.py``:

    <out_dir>/
        annotations.json
        <ep_tag>/rgb/001.jpg ... NNN.jpg
        <ep_tag>/...

For each episode we replay (start_position, start_rotation) +
action_sequence in habitat-sim and save the RGB observation BEFORE every
action. The bookkeeping matches ``render_episode_streamvln`` so that the
SFT dataset code can mix human + augmented data without special-casing.

Conventions:
    actions[0] = -1                 (init, no action taken)
    actions[i] ∈ {1=fwd, 2=left, 3=right}  for i in 1..len(rgb)-1
    Implicit terminal STOP is added by SNavVideoDataset; we strip the
    trailing 'stop' string from action_sequence.
    len(actions) == number of frames on disk.

Usage:
    conda activate navspace
    python data_augmentation/scripts/convert_manual_to_sft.py \
        --input  snav_data/manual/trajectories.json \
        --output snav_data/manual_sft \
        --width 384 --height 384 --hfov 120
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.agent.agent import ActionSpec
try:
    from habitat_sim.utils.common import quat_from_coeffs
except ImportError:
    from habitat_sim.utils import quat_from_coeffs  # type: ignore


ACTION_STR_TO_ID = {"forward": 1, "left": 2, "right": 3, "backward": 4, "stop": 0}
ACTION_STR_TO_SIM = {
    "forward": "move_forward",
    "left": "turn_left",
    "right": "turn_right",
    "backward": "move_backward",
}


def build_sim(
    scene_path: str,
    width: int,
    height: int,
    hfov: float,
    forward_step: float,
    turn_angle: float,
    camera_height: float,
    gpu_device_id: int,
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = gpu_device_id

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.hfov = hfov
    rgb_spec.position = [0.0, camera_height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]
    agent_cfg.action_space = {
        "stop": ActionSpec("stop"),
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=forward_step)),
        "move_backward": ActionSpec("move_backward", ActuationSpec(amount=forward_step)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=turn_angle)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=turn_angle)),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh = os.path.splitext(scene_path)[0] + ".navmesh"
    if os.path.exists(navmesh):
        sim.pathfinder.load_nav_mesh(navmesh)
    if not sim.pathfinder.is_loaded:
        raise RuntimeError(f"NavMesh missing: {navmesh}")
    return sim


def render_one(sim: habitat_sim.Simulator, episode: dict, out_root: Path,
               ep_tag: str, jpg_quality: int) -> dict | None:
    rgb_dir = out_root / ep_tag / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode["start_position"], dtype=np.float32)
    rot = episode.get("start_rotation")
    if rot:
        agent_state.rotation = quat_from_coeffs(rot)
    sim.get_agent(0).set_state(agent_state)

    actions: list[int] = [-1]
    frame_count = 0
    bad = False
    for a_str in episode["action_sequence"]:
        a_str = (a_str or "").lower()
        if a_str not in ACTION_STR_TO_ID:
            logging.warning("ep %s: skipping unknown action %r",
                            episode.get("episode_id"), a_str)
            bad = True
            break

        # Save frame BEFORE taking action (matches render_streamvln).
        obs = sim.get_sensor_observations()
        frame_count += 1
        Image.fromarray(obs["rgb"]).convert("RGB").save(
            rgb_dir / f"{frame_count:03d}.jpg", quality=jpg_quality)

        if a_str == "stop":
            break  # don't append; matches render_streamvln convention
        sim.step(ACTION_STR_TO_SIM[a_str])
        actions.append(ACTION_STR_TO_ID[a_str])

    if bad or len(actions) != frame_count:
        shutil.rmtree(rgb_dir.parent, ignore_errors=True)
        return None
    if len(actions) < 2:
        shutil.rmtree(rgb_dir.parent, ignore_errors=True)
        return None

    instr_text = ((episode.get("instruction") or {}).get("instruction_text", "")
                  or "").strip()
    if not instr_text:
        shutil.rmtree(rgb_dir.parent, ignore_errors=True)
        logging.warning("ep %s: empty instruction, dropped",
                        episode.get("episode_id"))
        return None

    return {
        "id": ep_tag,
        "video": ep_tag,
        "instructions": [instr_text],
        "actions": actions,
        "render_params": {},
        "source": "human_manual",
        "source_episode": episode.get("episode_id"),
        "source_trajectory": episode.get("trajectory_id"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="trajectories.json from the annotation server")
    ap.add_argument("--output", required=True,
                    help="output dir (will contain annotations.json + per-clip rgb/)")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--forward-step", type=float, default=0.25)
    ap.add_argument("--turn-angle", type=float, default=30.0)
    ap.add_argument("--camera-height", type=float, default=1.5)
    ap.add_argument("--gpu-device-id", type=int, default=0)
    ap.add_argument("--jpg-quality", type=int, default=90)
    ap.add_argument("--task-tag", default="manual",
                    help="appears in the clip folder name and ep id")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    data = json.loads(Path(args.input).read_text())
    episodes = data.get("episodes", data) if isinstance(data, dict) else data
    if not episodes:
        logging.error("no episodes found in %s", args.input)
        return 1

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # Group by scene so we only need to build sim once per scene.
    by_scene: dict[str, list[dict]] = {}
    for ep in episodes:
        scene = ep.get("scene_id")
        if not scene or not os.path.isfile(scene):
            logging.warning("ep %s: scene missing on disk: %r — skipped",
                            ep.get("episode_id"), scene)
            continue
        by_scene.setdefault(scene, []).append(ep)

    logging.info("input episodes: %d  (across %d scenes)",
                 sum(len(v) for v in by_scene.values()), len(by_scene))

    anno: list[dict] = []
    n_ok = n_skip = 0
    for scene, eps in by_scene.items():
        scan = Path(scene).stem
        logging.info("--- scene %s  (%d episodes) ---", scan, len(eps))
        sim = build_sim(scene, args.width, args.height, args.hfov,
                        args.forward_step, args.turn_angle,
                        args.camera_height, args.gpu_device_id)
        try:
            for ep in eps:
                ep_id = int(ep.get("episode_id", n_ok + n_skip))
                ep_tag = f"{scan}_{args.task_tag}_{ep_id:06d}"
                row = render_one(sim, ep, out_root, ep_tag, args.jpg_quality)
                if row is None:
                    n_skip += 1
                    continue
                anno.append(row)
                n_ok += 1
                if n_ok % 10 == 0:
                    logging.info("  rendered %d/%d (skipped %d)",
                                 n_ok, n_ok + n_skip, n_skip)
        finally:
            sim.close()

    anno_path = out_root / "annotations.json"
    anno_path.write_text(json.dumps(anno, indent=2, ensure_ascii=False))
    logging.info("== done == wrote %d annotations to %s (skipped %d)",
                 n_ok, anno_path, n_skip)
    if n_ok == 0:
        return 1

    n_frames = sum(len(a["actions"]) for a in anno)
    logging.info("total frames on disk: %d (avg %.1f / clip)",
                 n_frames, n_frames / n_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
