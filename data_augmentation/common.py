"""Shared helpers for NavSpace data augmentation pipelines."""

from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:  # pragma: no cover - optional in some conda envs
    yaml = None

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.common import load_api_key  # noqa: E402

ACTION_NAMES = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
ACTION_ID_FROM_NAME = {"move_forward": 1, "turn_left": 2, "turn_right": 3}


def _expand_env_in_config(node: Any) -> Any:
    """Recursively expand ${VAR} / $VAR in string config values."""
    if isinstance(node, str):
        return os.path.expandvars(node)
    if isinstance(node, list):
        return [_expand_env_in_config(item) for item in node]
    if isinstance(node, dict):
        return {key: _expand_env_in_config(value) for key, value in node.items()}
    return node


def load_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    base_dir = Path(__file__).parent
    if config_path:
        path = Path(config_path)
    else:
        json_path = base_dir / "config.json"
        yaml_path = base_dir / "config.yaml"
        path = json_path if json_path.exists() else yaml_path

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            cfg = json.load(file)
    elif yaml is not None:
        with path.open("r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    else:
        raise ImportError("PyYAML is required to load .yaml config; use config.json instead.")
    cfg = _expand_env_in_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_r2r_episodes(json_path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(json_path)
    episodes = []
    for ep in payload["episodes"]:
        instr = ep.get("instruction", {}).get("instruction_text", "").strip()
        goals = ep.get("goals", [])
        if not instr or not goals:
            continue
        episodes.append(
            {
                "episode_id": int(ep["episode_id"]),
                "trajectory_id": ep.get("trajectory_id"),
                "scene_id": ep["scene_id"],
                "start_position": [float(x) for x in ep["start_position"]],
                "start_rotation": [float(x) for x in ep.get("start_rotation", [])],
                "goals": goals,
                "goal_position": [float(x) for x in goals[0]["position"]],
                "instruction_text": instr,
                "info": ep.get("info", {}),
            }
        )
    return episodes


def resolve_mp3d_scene(scene_id: str, mp3d_root: str | Path) -> Path:
    mp3d_root = Path(mp3d_root)
    scene_id = scene_id.replace("\\", "/")
    if scene_id.startswith("mp3d/"):
        scene_id = scene_id[len("mp3d/") :]
    candidate = mp3d_root / scene_id
    if candidate.exists():
        return candidate
    scan = scene_id.split("/")[0]
    fallback = mp3d_root / scan / f"{scan}.glb"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Scene not found for scene_id={scene_id!r} under {mp3d_root}")


def geodesic_distance(start: list[float], goal: list[float], sim) -> float:
    sp = np.array(start, dtype=np.float32)
    gp = np.array(goal, dtype=np.float32)
    path = habitat_sim_shortest_path(sp, gp, sim)
    if path is None:
        return float("inf")
    return float(path.geodesic_distance)


def habitat_sim_shortest_path(start, goal, sim):
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    found = sim.pathfinder.find_path(path)
    if not found:
        return None
    return path


def build_sim(scene_path: str | Path, cfg: dict[str, Any], *, with_sensors: bool = True):
    """Construct a Habitat sim for this scene.

    When ``with_sensors=False`` we skip the CameraSensorSpec so a renderer
    context is not required — useful for navmesh-only / semantic-only queries
    (e.g. ``vertical_perception/2_verify.py`` default mode).
    """
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec

    hcfg = cfg["habitat"]
    scene_path = str(scene_path)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = int(hcfg.get("gpu_device_id", 0))
    if not with_sensors:
        # Tell habitat-sim that we don't need a GL context (no rendering).
        sim_cfg.create_renderer = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    if with_sensors:
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [int(hcfg["height"]), int(hcfg["width"])]
        rgb_spec.hfov = float(hcfg["hfov"])
        rgb_spec.position = [0.0, float(hcfg["camera_height"]), 0.0]
        agent_cfg.sensor_specifications = [rgb_spec]
    else:
        agent_cfg.sensor_specifications = []
    agent_cfg.action_space = {
        "stop": ActionSpec("stop"),
        "move_forward": ActionSpec(
            "move_forward", ActuationSpec(amount=float(hcfg["forward_step"]))
        ),
        "turn_left": ActionSpec(
            "turn_left", ActuationSpec(amount=float(hcfg["turn_angle"]))
        ),
        "turn_right": ActionSpec(
            "turn_right", ActuationSpec(amount=float(hcfg["turn_angle"]))
        ),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh_path = os.path.splitext(scene_path)[0] + ".navmesh"
    if os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
    if not sim.pathfinder.is_loaded:
        sim.close()
        raise RuntimeError(f"NavMesh not loaded: {navmesh_path}")
    return sim


def _to_action_id(action) -> int:
    if action is None:
        return 0
    if isinstance(action, str):
        return ACTION_ID_FROM_NAME.get(action, 0)
    return int(action)


def rollout_episode(
    sim,
    episode: dict[str, Any],
    cfg: dict[str, Any],
    *,
    capture_rgb: bool = False,
    capture_last_rgb: bool = False,
) -> dict[str, Any]:
    """Follow shortest path; return actions and optional first/last RGB frames."""
    from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
    from habitat_sim.utils.common import quat_from_coeffs
    import habitat_sim

    hcfg = cfg["habitat"]
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode["start_position"], dtype=np.float32)
    if episode.get("start_rotation"):
        agent_state.rotation = quat_from_coeffs(episode["start_rotation"])
    sim.get_agent(0).set_state(agent_state)

    follower = GreedyGeodesicFollower(
        sim.pathfinder,
        sim.get_agent(0),
        goal_radius=float(hcfg.get("goal_radius", 0.5)),
    )
    goal = episode["goal_position"]

    actions: list[int] = [-1]
    first_rgb: Optional[np.ndarray] = None
    last_rgb: Optional[np.ndarray] = None
    step = 0
    max_steps = int(hcfg.get("max_steps", 500))
    aid = 0

    while step < max_steps:
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"]
        if capture_rgb and first_rgb is None:
            first_rgb = np.array(rgb)
        if capture_last_rgb:
            last_rgb = np.array(rgb)

        try:
            raw_action = follower.next_action_along(goal)
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "actions": actions,
                "first_rgb": first_rgb,
                "last_rgb": last_rgb,
            }

        aid = _to_action_id(raw_action)
        if aid == 0:
            break

        action_name = ACTION_NAMES.get(aid)
        if action_name is None:
            break
        sim.step(action_name)
        actions.append(aid)
        step += 1

    success = aid == 0 and len(actions) >= 4
    return {
        "success": success,
        "actions": actions,
        "num_steps": len(actions) - 1,
        "first_rgb": first_rgb,
        "last_rgb": last_rgb,
    }


def count_turns(actions: list[int]) -> int:
    return sum(1 for a in actions if a in (2, 3))


def height_diff(episode: dict[str, Any]) -> float:
    sy = episode["start_position"][1]
    gy = episode["goal_position"][1]
    return abs(float(gy) - float(sy))


def resolve_mp3d_house(scene_id: str, mp3d_root: str | Path) -> Path:
    glb = resolve_mp3d_scene(scene_id, mp3d_root)
    house = glb.with_suffix(".house")
    if not house.exists():
        raise FileNotFoundError(f"House file not found: {house}")
    return house


def parse_mp3d_floor_info(house_path: str | Path, *, y_gap: float = 1.5) -> dict[str, Any]:
    """Cluster MP3D room-center y-coordinates into floor levels."""
    ys: list[float] = []
    with Path(house_path).open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                if parts[0] == "R":
                    ys.append(float(parts[6]))
            except ValueError:
                continue

    if not ys:
        # Fallback to visible surface vertices.
        with Path(house_path).open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 6 and parts[0] == "V":
                    try:
                        ys.append(float(parts[5]))
                    except ValueError:
                        continue

    if not ys:
        raise ValueError(f"No y samples in {house_path}")

    room_heights = sorted(set(round(y, 1) for y in ys))
    clusters: list[list[float]] = []
    for y in room_heights:
        if not clusters or y - clusters[-1][-1] > y_gap:
            clusters.append([y])
        else:
            clusters[-1].append(y)

    floor_height = {
        idx + 1: sum(cluster) / len(cluster) for idx, cluster in enumerate(clusters)
    }
    return {
        "total_floors": len(floor_height),
        "floor_height": floor_height,
    }


def get_scene_total_levels(sim) -> int:
    """Return the number of semantic levels in the loaded scene (>=1)."""
    try:
        levels = list(sim.semantic_scene.levels)
    except Exception:  # noqa: BLE001
        return 0
    return len(levels)


def _bbox_min_max(aabb) -> tuple[list[float], list[float]]:
    """habitat-sim BBox in 0.2.5 only exposes center/sizes."""
    c = list(aabb.center)
    s = list(aabb.sizes)
    return (
        [float(c[i]) - 0.5 * float(s[i]) for i in range(3)],
        [float(c[i]) + 0.5 * float(s[i]) for i in range(3)],
    )


def _aabb_volume(aabb) -> float:
    s = list(aabb.sizes)
    return float(s[0]) * float(s[1]) * float(s[2])


def _level_floor_y(level) -> float:
    """Floor plane (lowest y across the level's regions)."""
    ys: list[float] = []
    for r in list(level.regions) or []:
        try:
            mn_, _ = _bbox_min_max(r.aabb)
            ys.append(mn_[1])
        except Exception:  # noqa: BLE001
            continue
    if ys:
        return min(ys)
    # Fall back to the level's own AABB if it has one.
    try:
        mn_, _ = _bbox_min_max(level.aabb)
        return mn_[1]
    except Exception:  # noqa: BLE001
        return 0.0


def _sorted_levels(scene) -> list[tuple[int, object, float]]:
    """Return ``(raw_index, level, floor_y)`` sorted by floor_y ascending."""
    items: list[tuple[int, object, float]] = []
    for i, lvl in enumerate(list(scene.levels)):
        items.append((i, lvl, _level_floor_y(lvl)))
    items.sort(key=lambda kv: kv[2])
    return items


def _point_in_aabb(p: list[float], aabb, *, xz_tol: float = 0.2, y_tol: float = 0.5) -> bool:
    mn_, mx_ = _bbox_min_max(aabb)
    return (
        mn_[0] - xz_tol <= p[0] <= mx_[0] + xz_tol
        and mn_[2] - xz_tol <= p[2] <= mx_[2] + xz_tol
        and mn_[1] - y_tol <= p[1] <= mx_[1] + y_tol
    )


def get_floor_level_for_point(sim, xyz) -> Optional[dict[str, Any]]:
    """Resolve which semantic level a 3D point belongs to.

    On habitat-sim >= 0.2.5-main this uses ``semantic_scene.get_regions_for_point``
    when available; otherwise (e.g. habitat-sim 0.2.5 release) it falls back
    to region AABB containment and finally to nearest level-floor-y.
    Returns ``None`` only when there are no semantic levels at all.
    """
    try:
        levels = list(sim.semantic_scene.levels)
    except Exception:  # noqa: BLE001
        return None
    if not levels:
        return None

    scene = sim.semantic_scene
    sorted_levels = _sorted_levels(scene)  # [(raw_idx, level, floor_y), ...]
    sorted_index_of_raw = {raw: i for i, (raw, _, _) in enumerate(sorted_levels)}

    def _level_sorted_idx(level) -> int:
        for raw, lvl, _ in sorted_levels:
            if lvl is level or getattr(lvl, "id", None) == getattr(level, "id", None):
                return sorted_index_of_raw[raw]
        return 0

    p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

    # ── Path 1: official semantic API on newer habitat-sim ──
    try:
        import magnum as mn  # type: ignore[import-not-found]

        get_regions = getattr(scene, "get_regions_for_point", None)
        if callable(get_regions):
            try:
                indices = list(get_regions(mn.Vector3(*p)))
            except Exception:  # noqa: BLE001
                indices = []
            for ridx in indices:
                try:
                    region = scene.regions[int(ridx)]
                except (IndexError, ValueError):
                    continue
                level = getattr(region, "level", None)
                if level is None:
                    continue
                return {
                    "level_index": _level_sorted_idx(level),
                    "level_id": str(getattr(level, "id", "")),
                    "region_index": int(ridx),
                    "region_id": str(getattr(region, "id", "")),
                    "region_category": (
                        region.category.name()
                        if getattr(region, "category", None) is not None
                        else ""
                    ),
                    "floor_height": float(_level_floor_y(level)),
                    "via": "region_api",
                }
    except ImportError:
        pass

    # ── Path 2: manual region containment (works on habitat-sim 0.2.5) ──
    best_hit: Optional[tuple[int, object, float]] = None  # (raw_region_idx, region, volume)
    for ridx, region in enumerate(scene.regions):
        try:
            if not _point_in_aabb(p, region.aabb):
                continue
        except Exception:  # noqa: BLE001
            continue
        vol = _aabb_volume(region.aabb)
        if best_hit is None or vol < best_hit[2]:
            best_hit = (ridx, region, vol)

    if best_hit is not None:
        ridx, region, _ = best_hit
        level = getattr(region, "level", None)
        if level is not None:
            return {
                "level_index": _level_sorted_idx(level),
                "level_id": str(getattr(level, "id", "")),
                "region_index": int(ridx),
                "region_id": str(getattr(region, "id", "")),
                "region_category": (
                    region.category.name()
                    if getattr(region, "category", None) is not None
                    else ""
                ),
                "floor_height": float(_level_floor_y(level)),
                "via": "region_aabb",
            }

    # ── Path 3: nearest floor-y among levels ──
    if not sorted_levels:
        return None
    # Prefer the highest level whose floor_y is at-or-below the point's y.
    chosen_sorted_idx = 0
    chosen_raw_idx, chosen_level, chosen_floor = sorted_levels[0]
    for sidx, (raw, lvl, fy) in enumerate(sorted_levels):
        if fy <= p[1] + 0.5:
            chosen_sorted_idx = sidx
            chosen_raw_idx, chosen_level, chosen_floor = raw, lvl, fy
        else:
            break
    return {
        "level_index": chosen_sorted_idx,
        "level_id": str(getattr(chosen_level, "id", chosen_sorted_idx)),
        "region_index": -1,
        "region_id": "",
        "region_category": "",
        "floor_height": float(chosen_floor),
        "via": "nearest_y",
    }


def annotate_vertical_floors_from_sim(
    sim,
    start_pos: list[float],
    goal_pos: list[float],
) -> Optional[dict[str, Any]]:
    """Resolve (start_level_index, end_level_index, total_levels) via Habitat semantic API."""
    total = get_scene_total_levels(sim)
    if total <= 0:
        return None
    start_meta = get_floor_level_for_point(sim, start_pos)
    end_meta = get_floor_level_for_point(sim, goal_pos)
    if start_meta is None or end_meta is None:
        return None
    si = int(start_meta["level_index"])
    ei = int(end_meta["level_index"])
    direction = "up" if ei > si else ("down" if ei < si else "same")
    return {
        "start_level_index": si,
        "end_level_index": ei,
        "total_levels": int(total),
        "direction": direction,
        "height_diff": abs(float(goal_pos[1]) - float(start_pos[1])),
        "start_region_category": start_meta.get("region_category", ""),
        "end_region_category": end_meta.get("region_category", ""),
        "start_via": start_meta.get("via", ""),
        "end_via": end_meta.get("via", ""),
    }


def annotate_vertical_floors(
    scene_id: str,
    start_pos: list[float],
    goal_pos: list[float],
    mp3d_root: str | Path,
) -> dict[str, Any]:
    house_path = resolve_mp3d_house(scene_id, mp3d_root)
    info = parse_mp3d_floor_info(house_path)
    floor_height = info["floor_height"]

    def pos_to_floor(pos: list[float]) -> int:
        y = float(pos[1])
        return min(floor_height.keys(), key=lambda f: abs(floor_height[f] - y))

    start_floor = pos_to_floor(start_pos)
    end_floor = pos_to_floor(goal_pos)

    total_floors = int(info["total_floors"])
    if total_floors <= 1 and abs(float(goal_pos[1]) - float(start_pos[1])) >= 1.0:
        # Fallback when house metadata lacks multi-floor structure.
        total_floors = 2
        if float(goal_pos[1]) >= float(start_pos[1]):
            start_floor, end_floor = 1, 2
        else:
            start_floor, end_floor = 2, 1

    direction = "up" if end_floor > start_floor else "down"

    return {
        "start_floor": int(start_floor),
        "end_floor": int(end_floor),
        "total_floors": int(total_floors),
        "direction": direction,
        "height_diff": abs(float(goal_pos[1]) - float(start_pos[1])),
    }


def rollout_episode_with_frames(
    sim,
    episode: dict[str, Any],
    cfg: dict[str, Any],
    *,
    frame_stride: int = 5,
    max_frames: int = 12,
) -> dict[str, Any]:
    """Rollout shortest path and sample RGB frames along the way."""
    from habitat_sim.nav.greedy_geodesic_follower import GreedyGeodesicFollower
    from habitat_sim.utils.common import quat_from_coeffs
    import habitat_sim

    hcfg = cfg["habitat"]
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(episode["start_position"], dtype=np.float32)
    if episode.get("start_rotation"):
        agent_state.rotation = quat_from_coeffs(episode["start_rotation"])
    sim.get_agent(0).set_state(agent_state)

    follower = GreedyGeodesicFollower(
        sim.pathfinder,
        sim.get_agent(0),
        goal_radius=float(hcfg.get("goal_radius", 0.5)),
    )
    goal = episode["goal_position"]

    actions: list[int] = [-1]
    frames: list[np.ndarray] = []
    step = 0
    max_steps = int(hcfg.get("max_steps", 500))
    aid = 0

    while step < max_steps:
        obs = sim.get_sensor_observations()
        rgb = np.array(obs["rgb"])
        if step % frame_stride == 0 and len(frames) < max_frames:
            frames.append(rgb)

        try:
            raw_action = follower.next_action_along(goal)
        except Exception as exc:
            return {"success": False, "error": str(exc), "actions": actions, "frames": frames}

        aid = _to_action_id(raw_action)
        if aid == 0:
            break
        action_name = ACTION_NAMES.get(aid)
        if action_name is None:
            break
        sim.step(action_name)
        actions.append(aid)
        step += 1

    success = aid == 0 and len(actions) >= 4
    return {
        "success": success,
        "actions": actions,
        "num_steps": len(actions) - 1,
        "frames": frames,
    }


def sample_random_navigable_pair(
    sim,
    *,
    min_geodesic: float,
    max_geodesic: float,
    max_height_diff: float,
    rng: random.Random,
    max_attempts: int = 200,
) -> Optional[dict[str, Any]]:
    """Random (start, goal) pair on a connected nav island.

    Returns dict with start_position / goal_position / geodesic_distance on
    success, ``None`` if no valid pair was found within ``max_attempts``.
    """
    import habitat_sim

    pf = sim.pathfinder
    if not pf.is_loaded:
        return None

    for _ in range(max_attempts):
        start = pf.get_random_navigable_point()
        goal = pf.get_random_navigable_point()
        if start is None or goal is None:
            continue
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            continue
        if abs(float(goal[1]) - float(start[1])) > float(max_height_diff):
            continue

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if not pf.find_path(path):
            continue
        geo = float(path.geodesic_distance)
        if not math.isfinite(geo):
            continue
        if geo < float(min_geodesic) or geo > float(max_geodesic):
            continue

        yaw = rng.uniform(-math.pi, math.pi)
        half = 0.5 * yaw
        start_rotation = [0.0, math.sin(half), 0.0, math.cos(half)]

        return {
            "start_position": [float(x) for x in start.tolist()],
            "goal_position": [float(x) for x in goal.tolist()],
            "start_rotation": start_rotation,
            "geodesic_distance": geo,
        }
    return None


def checkpoint_append(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single record as JSONL — used for resumable per-episode logs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint_load_ids(path: str | Path, key: str = "episode_id") -> set:
    """Load already-processed ids from a JSONL checkpoint file."""
    path = Path(path)
    if not path.exists():
        return set()
    done: set = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in obj:
                done.add(obj[key])
    return done


def checkpoint_iter(path: str | Path):
    """Yield all records from a JSONL checkpoint file."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def detect_stairs_in_frames(cfg: dict[str, Any], frames: list[np.ndarray]) -> dict[str, Any]:
    if not frames:
        return {"stairs_frame_count": 0, "stairs_detected": False, "details": []}

    prompt = (
        "You verify indoor navigation images for stairs.\n"
        "For each image index, say whether stairs are clearly visible.\n"
        'Return JSON only: {"results": [{"index": 0, "has_stairs": true/false}, ...]}'
    )
    result = qwen_json(
        cfg,
        prompt,
        system="Return valid JSON only.",
        images=frames,
        model=cfg["qwen"]["vision_model"],
    )
    details = result.get("results", [])
    count = sum(1 for item in details if item.get("has_stairs"))
    return {
        "stairs_frame_count": count,
        "stairs_detected": count > 0,
        "details": details,
    }


# Spatial-relationship filter aligned with NavSpace-Datasets/Spatial Relationship/
# spatialrel_vln.json (ordinal + relation prepositions; no bare "through/past/around").
SPATIAL_REGEX = re.compile(
    r"(?:"
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b|"
    r"\b(between|along|beside)\b|"
    r"\b(on your left|on your right|to the left of|to the right of)\b|"
    r"\b(leftmost|rightmost|next to|opposite|very end|end of)\b|"
    r"\bthrough the (?:first|second|third|fourth|fifth|last)\b|"
    r"\b(?:first|second|third|fourth|fifth) (?:open )?door\b|"
    r"\bacross from\b"
    r")",
    re.IGNORECASE,
)


def spatial_match_keyword(text: str) -> Optional[str]:
    """Return the first matched spatial keyword/phrase, if any."""
    match = SPATIAL_REGEX.search(text)
    if not match:
        return None
    return match.group(0)


def match_spatial_instruction(text: str) -> bool:
    return spatial_match_keyword(text) is not None


def ordinal_word(n: int) -> str:
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    return words.get(n, f"{n}th")


def build_vertical_instruction_rule(
    meta: dict[str, Any],
    *,
    style: str,
    rng: random.Random,
) -> str:
    """Fallback rule-based vertical instruction if Qwen fails."""
    sf, ef, total = meta["start_floor"], meta["end_floor"], meta["total_floors"]
    if style == "absolute":
        if meta["direction"] == "up":
            return f"Please head up to the {ordinal_word(ef)} floor and stop."
        return f"Please go to the {ordinal_word(ef)} floor and stop."
    if style == "relative":
        delta = abs(ef - sf)
        word = "floor" if delta == 1 else "floors"
        if meta["direction"] == "up":
            return f"Go up {delta} {word} and stop."
        return f"Go down {delta} {word} and stop."
    if ef == total:
        return "Please go to the top floor and stop."
    if ef == 1:
        return "Please go to the bottom floor of the house and stop."
    return f"Please go to the {ordinal_word(ef)} floor and stop."


def merge_action_segments(
    actions: list[int],
    *,
    forward_step: float = 0.25,
    turn_angle: float = 30.0,
) -> list[dict[str, Any]]:
    """Merge consecutive identical actions into segments."""
    segments: list[dict[str, Any]] = []
    for aid in actions:
        if aid not in ACTION_NAMES:
            continue
        name = ACTION_NAMES[aid]
        if segments and segments[-1]["type"] == name:
            segments[-1]["count"] += 1
        else:
            segments.append({"type": name, "count": 1})

    merged: list[dict[str, Any]] = []
    for seg in segments:
        if seg["type"] == "move_forward":
            meters = round(seg["count"] * forward_step, 2)
            if meters <= 0:
                continue
            merged.append({"type": "forward", "meters": meters})
        elif seg["type"] == "turn_left":
            degrees = seg["count"] * turn_angle
            merged.append({"type": "turn_left", "degrees": int(degrees)})
        elif seg["type"] == "turn_right":
            degrees = seg["count"] * turn_angle
            merged.append({"type": "turn_right", "degrees": int(degrees)})
    return merged


def _format_meters(value: float) -> str:
    text = f"{value:g}"
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def _precise_opening(rng: random.Random) -> tuple[str, bool]:
    """Return (opening prefix, use_leading_first).

    Matches the distribution in NavSpace-Datasets/Precise Movement/precisemove_vln.json.
    """
    choice = rng.choices(
        [
            "from_position",
            "first_only",
            "starting_from",
            "from_location",
            "from_this",
            "from_spot",
            "from_here",
        ],
        weights=[71, 32, 24, 10, 8, 5, 5],
        k=1,
    )[0]
    if choice == "from_position":
        return "From your current position,", True
    if choice == "first_only":
        return "First,", False
    if choice == "starting_from":
        return "Starting from your current position,", True
    if choice == "from_location":
        return "From your current location,", True
    if choice == "from_this":
        return "From this position,", True
    if choice == "from_spot":
        return "From your current spot,", True
    return "From here,", True


def _precise_forward_phrase(m: str, rng: random.Random) -> str:
    templates = [
        "go straight for {m} meters",
        "go straight {m} meters",
        "go forward {m} meters",
        "go forward for {m} meters",
        "walk {m} meters forward",
        "walk forward {m} meters",
        "move forward {m} meters",
        "continue straight for {m} meters",
        "continue for {m} meters",
        "go {m} meters",
        "proceed straight for {m} meters",
        "head forward {m} meters",
    ]
    return rng.choice(templates).format(m=m)


def _precise_turn_phrase(seg: dict[str, Any], rng: random.Random) -> str:
    """Varied turn wording; always includes the numeric degree for validation."""
    degrees = int(seg["degrees"])
    is_left = seg["type"] == "turn_left"
    ccw = "counter-clockwise"
    cw = "clockwise"

    if degrees == 180:
        return rng.choice(
            [
                f"turn 180 degrees {ccw if is_left else cw}",
                f"turn half a circle {ccw if is_left else cw}",
                f"make a 180-degree {'left' if is_left else 'right'} turn",
                f"complete a full {ccw if is_left else cw} rotation",
                f"rotate {ccw if is_left else cw} by 180 degrees",
            ]
        )

    directional = [
        f"turn {'left' if is_left else 'right'} {degrees} degrees",
        f"turn {degrees} degrees {ccw if is_left else cw}",
        f"make a {degrees}-degree {'left' if is_left else 'right'} turn",
        f"rotate {ccw if is_left else cw} by {degrees} degrees",
    ]
    return rng.choice(directional)


def _precise_turn_and_forward_phrase(
    turn_seg: dict[str, Any],
    forward_seg: dict[str, Any],
    rng: random.Random,
) -> str:
    """Single clause with both turn and forward, as in the benchmark set."""
    turn = _precise_turn_phrase(turn_seg, rng)
    meters = _format_meters(forward_seg["meters"])
    tail = rng.choice(
        [
            f"and walk {meters} meters forward",
            f"and go forward {meters} meters",
            f"and go straight for {meters} meters",
            f"and continue for {meters} meters",
            f"and move forward {meters} meters",
            f"and proceed {meters} meters",
        ]
    )
    return f"{turn} {tail}"


def _precise_closing(rng: random.Random) -> str:
    return rng.choice(
        [
            ", and stop.",
            ", and stop",
            ", then stop.",
            " before stopping.",
            ", and come to a stop.",
        ]
    )


def build_precise_instruction(
    segments: list[dict[str, Any]],
    *,
    rng: Optional[random.Random] = None,
) -> str:
    """Rule-based instruction aligned with NavSpace Precise Movement benchmark phrasing."""
    rng = rng or random.Random()

    if not segments:
        return "From your current position, stop."

    opening, use_first = _precise_opening(rng)
    layout = rng.choice(["semicolon", "period", "comma_then"])

    phrase_groups: list[str] = []
    idx = 0
    while idx < len(segments):
        seg = segments[idx]
        if (
            seg["type"] in ("turn_left", "turn_right")
            and idx + 1 < len(segments)
            and segments[idx + 1]["type"] == "forward"
            and rng.random() < 0.35
        ):
            phrase_groups.append(
                _precise_turn_and_forward_phrase(seg, segments[idx + 1], rng)
            )
            idx += 2
            continue
        if seg["type"] == "forward":
            phrase_groups.append(
                _precise_forward_phrase(_format_meters(seg["meters"]), rng)
            )
        else:
            phrase_groups.append(_precise_turn_phrase(seg, rng))
        idx += 1

    if len(phrase_groups) == 1:
        return f"{opening} {phrase_groups[0]}{_precise_closing(rng)}"

    mid_connectors = ["then", "next", "after that", "and then"]
    closing = _precise_closing(rng)

    if layout == "period":
        first_prefix = "first " if use_first else ""
        sentences = [f"{first_prefix}{phrase_groups[0]}"]
        for phrase in phrase_groups[1:-1]:
            lead = rng.choice(["Then", "Next", "After that"])
            sentences.append(f"{lead}, {phrase}")
        sentences.append(f"Finally, {phrase_groups[-1]}{closing}")
        tail = sentences[1:]
        if tail and not tail[-1].endswith("."):
            tail[-1] = tail[-1].rstrip() + ("." if not closing.endswith(".") else "")
        return f"{opening} {sentences[0]}. " + ". ".join(tail)

    if layout == "comma_then":
        parts: list[str] = []
        for i, phrase in enumerate(phrase_groups):
            if i == 0:
                prefix = "first " if use_first else ""
                parts.append(f"{prefix}{phrase}")
            elif i == len(phrase_groups) - 1:
                parts.append(f"finally {phrase}")
            else:
                parts.append(f"{rng.choice(mid_connectors)} {phrase}")
        body = ", ".join(parts)
        if closing.startswith(","):
            return f"{opening} {body}{closing}"
        return f"{opening} {body}{closing}"

    parts = []
    for i, phrase in enumerate(phrase_groups):
        if i == 0:
            prefix = "first " if use_first else ""
            parts.append(f"{prefix}{phrase}")
        elif i == len(phrase_groups) - 1:
            parts.append(f"finally {phrase}")
        else:
            parts.append(f"{rng.choice(mid_connectors)} {phrase}")
    body = "; ".join(parts)
    if closing.startswith(","):
        return f"{opening} {body}{closing}"
    return f"{opening} {body}{closing}"


def extract_numbers_from_instruction(text: str) -> list[float]:
    nums = [float(x) for x in re.findall(r"(\d+\.?\d*)", text)]
    lower = text.lower()
    if "half a circle" in lower or "half circle" in lower:
        if 180.0 not in nums:
            nums.append(180.0)
    if "full clockwise rotation" in lower or "full counter-clockwise rotation" in lower:
        if 180.0 not in nums:
            nums.append(180.0)
    return nums


def validate_precise_instruction(text: str, segments: list[dict[str, Any]]) -> bool:
    expected: list[float] = []
    for seg in segments:
        if seg["type"] == "forward":
            expected.append(float(seg["meters"]))
        else:
            expected.append(float(seg["degrees"]))
    found = extract_numbers_from_instruction(text)
    if len(found) != len(expected):
        return False
    remaining = list(expected)
    for val in found:
        matched = False
        for idx, exp in enumerate(remaining):
            if abs(val - exp) <= 0.01:
                remaining.pop(idx)
                matched = True
                break
        if not matched:
            return False
    return not remaining


def rgb_to_b64(rgb: np.ndarray, resize: int = 384) -> str:
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if resize:
        bgr = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output: {text[:200]!r}")


def qwen_call(
    cfg: dict[str, Any],
    *,
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    images: Optional[list[np.ndarray]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Call Qwen via DashScope OpenAI-compatible API."""
    from openai import OpenAI

    qcfg = cfg["qwen"]
    if not api_key:
        env_name = qcfg.get("key_env", "DASHSCOPE_API_KEY")
        env_value = os.environ.get(env_name) if env_name else None
        fallback = qcfg.get("api_key")
        if env_value:
            api_key = env_value
        elif fallback:
            api_key = fallback
        else:
            api_key = load_api_key(None, env_name, None)
    client = OpenAI(api_key=api_key, base_url=qcfg["base_url"])
    model = model or (qcfg["vision_model"] if images else qcfg["text_model"])

    content: list[dict[str, Any]] = []
    if images:
        for rgb in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{rgb_to_b64(rgb)}"},
                }
            )
    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    retry_delay = float(qcfg.get("initial_retry_delay", 2.0))
    max_retries = int(qcfg.get("max_retries", 5))
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            message = response.choices[0].message.content
            if isinstance(message, str):
                return message.strip()
            if isinstance(message, list):
                parts = [p.get("text", "") for p in message if isinstance(p, dict)]
                return "\n".join(parts).strip()
            return str(message or "").strip()
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Qwen API attempt %d failed: %s", attempt + 1, exc)
            time.sleep(retry_delay)
            retry_delay *= 1.5
    raise RuntimeError(f"Qwen API failed after {max_retries} retries: {last_error}")


def qwen_json(
    cfg: dict[str, Any],
    prompt: str,
    *,
    system: Optional[str] = None,
    images: Optional[list[np.ndarray]] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    text = qwen_call(cfg, prompt=prompt, system=system, images=images, model=model)
    return _extract_json_object(text)


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "to", "of", "on",
    "in", "at", "for", "by", "with", "into", "onto", "from", "as", "is",
    "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "your", "you", "it", "its", "there", "do", "does", "doing",
    "you'll", "ll", "we", "i", "go", "head", "move", "walk", "turn",
    "down", "up", "left", "right", "forward",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def content_tokens(text: str) -> list[str]:
    """Lower-cased content tokens used for instruction-overlap checks."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "") if tok.lower() not in _STOPWORDS]


def token_overlap_ratio(reference: str, candidate: str) -> float:
    ref = set(content_tokens(reference))
    if not ref:
        return 1.0
    cand = set(content_tokens(candidate))
    if not cand:
        return 0.0
    return len(ref & cand) / len(ref)


def to_custom_instructions(
    records: list[dict[str, Any]],
    *,
    id_field: str = "episode_id",
    instruction_field: str = "instruction",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rec in records:
        eid = int(rec[id_field])
        if eid in seen:
            continue
        seen.add(eid)
        instr = rec[instruction_field]
        if isinstance(instr, list):
            instructions = instr
        else:
            instructions = [instr]
        out.append({"id": eid, "instructions": instructions})
    return out


def pipeline_output_dir(cfg: dict[str, Any], name: str) -> Path:
    root = REPO_ROOT / cfg["paths"]["output_root"] / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def add_repo_to_path() -> Path:
    return REPO_ROOT
