#!/usr/bin/env python3
"""
Render trajectories using Habitat-Sim.

Supports R2R-CE, EnvDrop, and RxR-CE data formats.
Uses GreedyGeodesicFollower to navigate to goal_position via the navmesh,
which VLN-CE confirms produces trajectories equivalent to reference_path.

Three output modes (--output_mode):

  frames  (default, StreamVLN-compatible):
    {output_dir}/images/{scan}_{tag}_{id}/rgb/001.jpg ...
    {output_dir}/annotations.json

  video  (SNav-compatible, cumulative MP4):
    {output_dir}/{scan}_{tag}_{id}/step_0_video.mp4 ...  (cumulative videos)
    {output_dir}/llava_annotations.json   (LLaVA SFT format, 1 sample per step)

  snav_frames  (SNav-compatible, per-step JPGs — O(N) storage):
    {output_dir}/{scan}_{tag}_{id}/000.jpg, 001.jpg ...
    {output_dir}/llava_annotations.json   (with video_nframes field)
"""

import argparse
import gzip
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
from habitat_sim.utils.common import quat_from_coeffs

GT_STEP = 6  # SNav predicts 6 future actions per step


@dataclass
class Episode:
    episode_id: int
    scene_id: str
    instructions: List[str]
    start_position: List[float]
    start_rotation: List[float]
    goal_position: List[float]
    trajectory_id: Optional[int] = None


def _to_action_id(action) -> int:
    if action is None:
        return 0
    if isinstance(action, str):
        return {"move_forward": 1, "turn_left": 2, "turn_right": 3}.get(action, 0)
    return int(action)


_ACTION_ID_TO_NAME = {1: "move_forward", 2: "turn_left", 3: "turn_right"}


def _load_json(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_r2r_episodes(json_path: str) -> List[Episode]:
    data = _load_json(json_path)
    episodes = []
    for ep in data["episodes"]:
        instr = ep["instruction"]["instruction_text"].strip()
        if not instr:
            continue
        goals = ep.get("goals", [])
        if not goals:
            continue
        goal_pos = goals[0]["position"]
        episodes.append(Episode(
            episode_id=int(ep["episode_id"]),
            scene_id=ep["scene_id"],
            instructions=[instr],
            start_position=ep["start_position"],
            start_rotation=ep["start_rotation"],
            goal_position=goal_pos,
            trajectory_id=ep.get("trajectory_id"),
        ))
    return episodes


def load_rxr_episodes(json_path: str, lang_filter: str = "en") -> List[Episode]:
    data = _load_json(json_path)
    episodes = []
    for ep in data["episodes"]:
        instr_obj = ep.get("instruction", {})
        if not isinstance(instr_obj, dict):
            continue
        lang = instr_obj.get("language", "")
        text = instr_obj.get("instruction_text", "").strip()
        if lang_filter and not lang.lower().startswith(lang_filter):
            continue
        if not text:
            continue

        goals = ep.get("goals", [])
        ref_path = ep.get("reference_path", [])
        if goals:
            goal_pos = goals[0].get("position") if isinstance(goals[0], dict) else None
        else:
            goal_pos = None
        if goal_pos is None and ref_path:
            goal_pos = ref_path[-1]
        if goal_pos is None:
            continue

        start_rot = ep.get("start_rotation", [])
        if isinstance(start_rot, (list, tuple)) and len(start_rot) == 4:
            start_rot = [float(x) for x in start_rot]
        else:
            start_rot = []
        episodes.append(Episode(
            episode_id=int(ep["episode_id"]),
            scene_id=str(ep["scene_id"]),
            instructions=[text],
            start_position=[float(x) for x in ep["start_position"]],
            start_rotation=start_rot,
            goal_position=[float(x) for x in goal_pos],
            trajectory_id=ep.get("trajectory_id"),
        ))
    return episodes


def build_sim(
    scene_path: str,
    width: int = 640,
    height: int = 480,
    hfov: float = 79.0,
    forward_step: float = 0.25,
    turn_angle: float = 15.0,
    camera_height: float = 0.88,
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

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
        "move_forward": ActionSpec(
            "move_forward", ActuationSpec(amount=forward_step)),
        "turn_left": ActionSpec(
            "turn_left", ActuationSpec(amount=turn_angle)),
        "turn_right": ActionSpec(
            "turn_right", ActuationSpec(amount=turn_angle)),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    navmesh_path = os.path.splitext(scene_path)[0] + ".navmesh"
    if os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
    if not sim.pathfinder.is_loaded:
        raise RuntimeError(f"NavMesh not loaded: {navmesh_path}")
    return sim


def render_episode_streamvln(
    sim: habitat_sim.Simulator,
    episode: Episode,
    output_dir: str,
    dataset_tag: str,
    max_steps: int = 500,
    goal_radius: float = 0.5,
) -> Optional[Dict]:
    """
    Render one episode using GreedyGeodesicFollower to navigate to
    goal_position via the navmesh (equivalent to reference_path per VLN-CE).

    Frame i is the observation BEFORE action i is taken.
    actions[0] = -1 (initial, no action); len(actions) == number of frames.

    Returns annotation dict on success, None if the episode should be skipped.
    """
    scan = episode.scene_id.split("/")[-2]
    ep_id = episode.episode_id

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode.start_position, dtype=np.float32)
    if episode.start_rotation:
        agent_state.rotation = quat_from_coeffs(episode.start_rotation)
    sim.get_agent(0).set_state(agent_state)

    video_rel = os.path.join("images", f"{scan}_{dataset_tag}_{ep_id:06d}")
    rgb_dir = os.path.join(output_dir, video_rel, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    actions: List[int] = [-1]
    frame_count = 0

    follower = GreedyGeodesicFollower(
        sim.pathfinder, sim.get_agent(0), goal_radius=goal_radius)
    goal = episode.goal_position

    step = 0
    while step < max_steps:
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"]
        frame_count += 1
        Image.fromarray(rgb).convert("RGB").save(
            os.path.join(rgb_dir, f"{frame_count:03d}.jpg"), quality=90)

        try:
            raw_action = follower.next_action_along(goal)
        except Exception as exc:
            logging.warning("Episode %s follower error: %s", ep_id, exc)
            shutil.rmtree(os.path.join(output_dir, video_rel),
                          ignore_errors=True)
            return None

        aid = _to_action_id(raw_action)
        if aid == 0:
            break

        action_name = _ACTION_ID_TO_NAME.get(aid)
        if action_name is None:
            break
        sim.step(action_name)
        actions.append(aid)
        step += 1

    if len(actions) > 498:
        shutil.rmtree(os.path.join(output_dir, video_rel), ignore_errors=True)
        return None

    if len(actions) < 4:
        shutil.rmtree(os.path.join(output_dir, video_rel), ignore_errors=True)
        return None

    if len(actions) != frame_count:
        logging.warning(
            "Episode %s: actions(%d) != frames(%d), skipping",
            ep_id, len(actions), frame_count)
        shutil.rmtree(os.path.join(output_dir, video_rel), ignore_errors=True)
        return None

    return {
        "id": ep_id,
        "video": video_rel,
        "instructions": episode.instructions,
        "actions": actions,
    }


# ═══════════════════════════════════════════════════════════════════════
# Video output mode (SNav-compatible, cumulative MP4 — legacy)
# ═══════════════════════════════════════════════════════════════════════

_ACT_ID_TO_LANG = {
    1: "Move forward",
    2: "Turn left",
    3: "Turn right",
    0: "Stop",
}


def _multi_step_gt(gt_actions: List[int], step_idx: int, gt_step: int = GT_STEP) -> str:
    """Return comma-separated natural-language action string for ``gt_step``
    future actions starting at ``step_idx`` (pad with Stop)."""
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


# ═══════════════════════════════════════════════════════════════════════
# SNav frames mode (per-step JPG — O(N) storage)
# ═══════════════════════════════════════════════════════════════════════

def render_episode_snav_frames(
    sim: habitat_sim.Simulator,
    episode: Episode,
    output_dir: str,
    dataset_tag: str,
    video_subdir: str,
    max_steps: int = 500,
    goal_radius: float = 0.5,
) -> Optional[List[Dict]]:
    """Render one episode, save per-step observation JPGs (O(N) storage),
    and produce LLaVA-format annotation dicts with ``video_nframes`` field.

    The frame directory is referenced via ``"video"`` so the dataloader can
    list JPGs and load the first ``video_nframes`` for each step.
    """
    scan = episode.scene_id.split("/")[-2]
    ep_id = episode.episode_id
    ep_tag = f"{scan}_{dataset_tag}_{ep_id:06d}"

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode.start_position, dtype=np.float32)
    if episode.start_rotation:
        agent_state.rotation = quat_from_coeffs(episode.start_rotation)
    sim.get_agent(0).set_state(agent_state)

    ep_frame_dir = os.path.join(output_dir, ep_tag)
    os.makedirs(ep_frame_dir, exist_ok=True)

    follower = GreedyGeodesicFollower(
        sim.pathfinder, sim.get_agent(0), goal_radius=goal_radius)
    goal = episode.goal_position

    actions_raw: List[int] = [-1]
    num_frames = 0
    step = 0

    while step < max_steps:
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(ep_frame_dir, f"{num_frames:03d}.jpg")
        cv2.imwrite(frame_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        num_frames += 1

        try:
            raw_action = follower.next_action_along(goal)
        except Exception as exc:
            logging.warning("Episode %s follower error: %s", ep_id, exc)
            shutil.rmtree(ep_frame_dir, ignore_errors=True)
            return None

        aid = _to_action_id(raw_action)
        if aid == 0:
            break

        action_name = _ACTION_ID_TO_NAME.get(aid)
        if action_name is None:
            break
        sim.step(action_name)
        actions_raw.append(aid)
        step += 1

    if num_frames != len(actions_raw):
        shutil.rmtree(ep_frame_dir, ignore_errors=True)
        return None
    if num_frames < 4 or num_frames > 498:
        shutil.rmtree(ep_frame_dir, ignore_errors=True)
        return None

    gt_actions = actions_raw[1:] + [0]
    video_rel = os.path.join(video_subdir, ep_tag)

    llava_entries: List[Dict] = []
    for step_idx in range(num_frames):
        gt_act_str = _multi_step_gt(gt_actions, step_idx)
        for instr in episode.instructions:
            if not instr or not instr.strip():
                continue
            prompt = _build_prompt(instr.strip())
            entry = {
                "id": f"{video_subdir}_{ep_id:06d}_step_{step_idx}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": f"Final Answer: {gt_act_str}"},
                ],
                "video": video_rel,
                "video_nframes": step_idx + 1,
            }
            llava_entries.append(entry)

    return llava_entries


def render_episode_video(
    sim: habitat_sim.Simulator,
    episode: Episode,
    output_dir: str,
    dataset_tag: str,
    video_subdir: str,
    max_steps: int = 500,
    goal_radius: float = 0.5,
) -> Optional[List[Dict]]:
    """Render one episode and directly produce cumulative MP4 videos +
    LLaVA-format annotation dicts (one per step per instruction).

    Returns a list of LLaVA annotation dicts, or None on failure.
    """
    scan = episode.scene_id.split("/")[-2]
    ep_id = episode.episode_id
    ep_tag = f"{scan}_{dataset_tag}_{ep_id:06d}"

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode.start_position, dtype=np.float32)
    if episode.start_rotation:
        agent_state.rotation = quat_from_coeffs(episode.start_rotation)
    sim.get_agent(0).set_state(agent_state)

    ep_video_dir = os.path.join(output_dir, ep_tag)
    os.makedirs(ep_video_dir, exist_ok=True)

    follower = GreedyGeodesicFollower(
        sim.pathfinder, sim.get_agent(0), goal_radius=goal_radius)
    goal = episode.goal_position

    actions_raw: List[int] = [-1]
    frames_bgr: List[np.ndarray] = []
    step = 0

    while step < max_steps:
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames_bgr.append(bgr)

        try:
            raw_action = follower.next_action_along(goal)
        except Exception as exc:
            logging.warning("Episode %s follower error: %s", ep_id, exc)
            shutil.rmtree(ep_video_dir, ignore_errors=True)
            return None

        aid = _to_action_id(raw_action)
        if aid == 0:
            break

        action_name = _ACTION_ID_TO_NAME.get(aid)
        if action_name is None:
            break
        sim.step(action_name)
        actions_raw.append(aid)
        step += 1

    num_frames = len(frames_bgr)
    if num_frames != len(actions_raw):
        shutil.rmtree(ep_video_dir, ignore_errors=True)
        return None
    if num_frames < 4 or num_frames > 498:
        shutil.rmtree(ep_video_dir, ignore_errors=True)
        return None

    # effective ground-truth action sequence (remove leading -1, append stop)
    gt_actions = actions_raw[1:] + [0]

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Write cumulative videos and build annotation entries
    llava_entries: List[Dict] = []
    prompt_text = None  # lazily built per instruction

    for step_idx in range(num_frames):
        video_filename = f"step_{step_idx}_video.mp4"
        video_abs = os.path.join(ep_video_dir, video_filename)
        writer = cv2.VideoWriter(video_abs, fourcc, 1.0, (w, h))
        for fi in range(step_idx + 1):
            writer.write(frames_bgr[fi])
        writer.release()

        gt_act_str = _multi_step_gt(gt_actions, step_idx)
        video_rel = os.path.join(ep_tag, video_filename)

        for instr in episode.instructions:
            if not instr or not instr.strip():
                continue
            prompt = _build_prompt(instr.strip())
            entry = {
                "id": f"{video_subdir}_{ep_id:06d}_step_{step_idx}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": f"Final Answer: {gt_act_str}"},
                ],
                "video": os.path.join(video_subdir, video_rel),
            }
            llava_entries.append(entry)

    return llava_entries


def main():
    parser = argparse.ArgumentParser(
        description="Render trajectories using Habitat-Sim")
    parser.add_argument("--data_json", required=True,
                        help="Episode JSON path (.json or .json.gz)")
    parser.add_argument("--data_format", choices=["r2r", "rxr"], default="r2r",
                        help="r2r for R2R-CE/EnvDrop, rxr for RxR-CE")
    parser.add_argument("--dataset_tag", default="r2r",
                        help="Tag in output paths (r2r / rxr / envdrop)")
    parser.add_argument("--scenes_root",
                        default="/mnt/longyuxing/yhl/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="0 = all episodes")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--goal_radius", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--forward_step", type=float, default=0.25)
    parser.add_argument("--turn_angle", type=float, default=15.0)
    parser.add_argument("--camera_height", type=float, default=0.88)
    parser.add_argument("--lang_filter", default="en",
                        help="Language filter for RxR data")
    parser.add_argument("--output_mode",
                        choices=["frames", "video", "snav_frames"],
                        default="frames",
                        help="frames: StreamVLN format (rgb JPGs). "
                             "video: SNav format (cumulative MP4 + "
                             "LLaVA JSON). "
                             "snav_frames: SNav format with "
                             "per-step JPGs (O(N) storage) + LLaVA JSON.")
    parser.add_argument("--video_subdir", default="",
                        help="Subdirectory name used in LLaVA JSON 'video' "
                             "field (e.g. 'r2rce'). Video/snav_frames "
                             "mode only.")
    parser.add_argument("--custom_instructions_json", default=None,
                        help="JSON file with custom instructions to override "
                             "episode instructions. Expected format: "
                             "[{\"id\": <episode_id>, \"instructions\": "
                             "[...]}]. Episodes without a match are skipped.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_handlers = [logging.StreamHandler()]
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        log_handlers.append(
            logging.FileHandler(args.log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers,
    )

    if args.data_format == "r2r":
        episodes = load_r2r_episodes(args.data_json)
    else:
        episodes = load_rxr_episodes(args.data_json,
                                     lang_filter=args.lang_filter)

    if args.custom_instructions_json:
        with open(args.custom_instructions_json, "r", encoding="utf-8") as f:
            custom_data = json.load(f)
        custom_map = {int(e["id"]): e["instructions"] for e in custom_data}
        logging.info("Loaded %d custom instructions from %s",
                     len(custom_map), args.custom_instructions_json)
        filtered = []
        for ep in episodes:
            if ep.episode_id in custom_map:
                ep.instructions = custom_map[ep.episode_id]
                filtered.append(ep)
        logging.info("Filtered episodes with custom instructions: %d -> %d",
                     len(episodes), len(filtered))
        episodes = filtered

    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    logging.info("Loaded %d episodes from %s", len(episodes), args.data_json)
    logging.info("Dataset tag: %s, format: %s, output_mode: %s",
                 args.dataset_tag, args.data_format, args.output_mode)
    logging.info("Camera: %dx%d HFOV=%s forward=%.2fm turn=%.1f° "
                 "height=%.2fm",
                 args.width, args.height, args.hfov, args.forward_step,
                 args.turn_angle, args.camera_height)

    by_scene: Dict[str, List[Episode]] = defaultdict(list)
    for ep in episodes:
        by_scene[ep.scene_id].append(ep)

    # ── Frames mode state ──
    anno_path = os.path.join(args.output_dir, "annotations.json")
    # ── Video / snav_frames mode state ──
    llava_anno_path = os.path.join(args.output_dir, "llava_annotations.json")

    annotations: List[Dict] = []
    llava_annotations: List[Dict] = []
    existing_ids: set = set()

    if args.output_mode == "frames":
        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                annotations = json.load(f)
            existing_ids = {a["id"] for a in annotations}
            logging.info("Resuming (frames): %d existing annotations",
                         len(existing_ids))
    else:
        if os.path.exists(llava_anno_path):
            with open(llava_anno_path, "r") as f:
                llava_annotations = json.load(f)
            _seen = set()
            for a in llava_annotations:
                vid = a.get("video", "")
                ep_tag = vid.split("/")[1] if "/" in vid else ""
                _seen.add(ep_tag)
            existing_ids = _seen
            logging.info("Resuming (%s): %d episode dirs done, "
                         "%d LLaVA entries",
                         args.output_mode, len(existing_ids),
                         len(llava_annotations))

    total_eps = len(episodes)
    skipped = 0
    rendered = 0

    progress = tqdm(total=total_eps, desc="Rendering", unit="ep")
    for scene_id, eps in sorted(by_scene.items()):
        scene_path = os.path.join(args.scenes_root, scene_id)
        if not os.path.exists(scene_path):
            logging.warning("Scene not found: %s", scene_path)
            progress.update(len(eps))
            skipped += len(eps)
            continue

        logging.info("Loading scene: %s (%d eps)", scene_path, len(eps))
        sim = build_sim(
            scene_path, args.width, args.height, args.hfov,
            args.forward_step, args.turn_angle, args.camera_height)

        for ep in eps:
            scan = ep.scene_id.split("/")[-2]
            ep_tag = f"{scan}_{args.dataset_tag}_{ep.episode_id:06d}"

            if args.output_mode == "frames":
                if ep.episode_id in existing_ids:
                    progress.update(1)
                    continue
                anno = render_episode_streamvln(
                    sim, ep, args.output_dir, args.dataset_tag,
                    max_steps=args.max_steps,
                    goal_radius=args.goal_radius)
                if anno:
                    annotations.append(anno)
                    existing_ids.add(ep.episode_id)
                    rendered += 1
                    logging.info("Episode %d: %d actions",
                                 ep.episode_id, len(anno["actions"]) - 1)
                else:
                    skipped += 1
                    logging.warning("Skipped episode %d", ep.episode_id)

            elif args.output_mode == "video":
                if ep_tag in existing_ids:
                    progress.update(1)
                    continue
                entries = render_episode_video(
                    sim, ep, args.output_dir, args.dataset_tag,
                    video_subdir=args.video_subdir or args.dataset_tag,
                    max_steps=args.max_steps,
                    goal_radius=args.goal_radius)
                if entries:
                    llava_annotations.extend(entries)
                    existing_ids.add(ep_tag)
                    rendered += 1
                    logging.info("Episode %d: %d videos, %d LLaVA entries",
                                 ep.episode_id,
                                 entries[0]["video"].count("step_") if entries else 0,
                                 len(entries))
                else:
                    skipped += 1
                    logging.warning("Skipped episode %d", ep.episode_id)

            elif args.output_mode == "snav_frames":
                if ep_tag in existing_ids:
                    progress.update(1)
                    continue
                entries = render_episode_snav_frames(
                    sim, ep, args.output_dir, args.dataset_tag,
                    video_subdir=args.video_subdir or args.dataset_tag,
                    max_steps=args.max_steps,
                    goal_radius=args.goal_radius)
                if entries:
                    llava_annotations.extend(entries)
                    existing_ids.add(ep_tag)
                    rendered += 1
                    logging.info("Episode %d: %d frames, %d LLaVA entries",
                                 ep.episode_id,
                                 entries[-1].get("video_nframes", 0),
                                 len(entries))
                else:
                    skipped += 1
                    logging.warning("Skipped episode %d", ep.episode_id)

            progress.update(1)

        sim.close()

        # Checkpoint after each scene
        if args.output_mode == "frames":
            with open(anno_path, "w") as f:
                json.dump(annotations, f, indent=2)
            logging.info("Checkpoint (frames): %d annotations saved",
                         len(annotations))
        else:
            with open(llava_anno_path, "w") as f:
                json.dump(llava_annotations, f, indent=2)
            logging.info("Checkpoint (%s): %d LLaVA entries saved",
                         args.output_mode, len(llava_annotations))

    progress.close()

    if args.output_mode == "frames":
        with open(anno_path, "w") as f:
            json.dump(annotations, f, indent=2)
        logging.info("Done. Rendered: %d, Skipped: %d, annotations: %d",
                     rendered, skipped, len(annotations))
        logging.info("Annotations: %s", anno_path)
    else:
        with open(llava_anno_path, "w") as f:
            json.dump(llava_annotations, f, indent=2)
        logging.info("Done. Rendered: %d, Skipped: %d, LLaVA entries: %d",
                     rendered, skipped, len(llava_annotations))
        logging.info("LLaVA annotations: %s", llava_anno_path)


if __name__ == "__main__":
    main()
