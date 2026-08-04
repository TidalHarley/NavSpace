from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from filelock import FileLock

from evaluation.config import TASK_DATASET_MAP


LOGGER = logging.getLogger(__name__)


ACTION_NAME_MAP = {
    "move forward": "move_forward",
    "forward": "move_forward",
    "turn left": "turn_left",
    "left": "turn_left",
    "turn right": "turn_right",
    "right": "turn_right",
    "move backward": "move_backward",
    "backward": "move_backward",
    "look up": "look_up",
    "look down": "look_down",
    "stop": "stop",
}


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def locked_load_json(json_path: Path) -> list[dict[str, Any]]:
    lock_path = Path(str(json_path) + ".lock")
    with FileLock(str(lock_path)):
        if not json_path.exists() or json_path.stat().st_size == 0:
            return []
        try:
            with json_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            corrupt_path = json_path.with_suffix(json_path.suffix + f".corrupt.{os.getpid()}")
            try:
                json_path.replace(corrupt_path)
                LOGGER.warning("Moved corrupt JSON result file %s to %s", json_path, corrupt_path)
            except OSError:
                LOGGER.warning("Failed to move corrupt JSON result file %s", json_path, exc_info=True)
            return []


def locked_dump_json(data: list[dict[str, Any]], json_path: Path) -> None:
    lock_path = Path(str(json_path) + ".lock")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        tmp_path = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        tmp_path.replace(json_path)


def load_dataset_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            payload = json.load(file)
    else:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    if isinstance(payload, dict) and "episodes" in payload:
        return payload["episodes"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format in {path}")


def resolve_task_dataset_path(repo_root: Path, task: str) -> Path:
    try:
        relative_path = TASK_DATASET_MAP[task]
    except KeyError as exc:
        raise KeyError(f"Unknown NavSpace task '{task}'") from exc
    return repo_root / relative_path


def load_api_key(
    direct_key: Optional[str],
    env_name: Optional[str],
    key_file: Optional[Path],
    key_field: str = "api_key",
) -> str:
    if direct_key:
        return direct_key
    if env_name:
        import os

        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
    if key_file and key_file.exists():
        with key_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        key_value = payload.get(key_field)
        if key_value:
            return key_value
    raise FileNotFoundError(
        "API key not found. Provide --api-key, set the environment variable, "
        "or supply a valid --api-key-file."
    )


def encode_image_b64(image_bgr: np.ndarray, resize_to: Optional[int] = None) -> str:
    image = image_bgr
    if resize_to:
        image = cv2.resize(image_bgr, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_images_as_video(
    images: list[np.ndarray],
    original_fps: float,
    max_frames_num: int,
    target_fps: float = 1.0,
    force_sample: bool = False,
) -> tuple[np.ndarray, str, float]:
    if max_frames_num == 0:
        return np.zeros((1, 224, 224, 3)), "0.00s", 0.0

    total_frames = len(images)
    video_duration = total_frames / original_fps if original_fps else 0.0
    sampling_interval = max(1, round(original_fps / target_fps)) if target_fps else 1
    frame_indices = list(range(0, total_frames, sampling_interval))

    if len(frame_indices) > max_frames_num or force_sample:
        frame_indices = np.linspace(0, total_frames - 1, max_frames_num, dtype=int).tolist()

    time_stamps = [idx / original_fps for idx in frame_indices]
    time_str = ",".join(f"{stamp:.2f}s" for stamp in time_stamps)
    sampled_frames = np.stack([images[index] for index in frame_indices])
    return sampled_frames, time_str, video_duration


ARROW_ACTION_MAP = {"↑": "move_forward", "←": "turn_left", "→": "turn_right"}


def extract_actions(text: str, max_actions: int = 4) -> list[str]:
    # SNav targets emit arrow glyphs (e.g. "Final Answer: ↑→↑"); prefer the
    # segment after the last "Final Answer:" so stray glyphs are ignored.
    segment = text.split("Final Answer:")[-1] if "Final Answer:" in text else text
    # Arrow mode is for SNav-style answers (glyphs + literal uppercase STOP).
    # Do not treat lowercase English "stop" as arrow-mode — that breaks LLM verb parsing.
    if any(a in segment for a in ARROW_ACTION_MAP) or re.search(r"\bSTOP\b", segment):
        actions = []
        for match in re.finditer(r"↑|←|→|STOP", segment):
            tok = match.group(0)
            if tok == "STOP":
                actions.append("stop")
                break
            actions.append(ARROW_ACTION_MAP[tok])
            if len(actions) >= max_actions:
                break
        return actions

    pattern = re.compile(
        r"\b(move forward|turn left|turn right|move backward|look up|look down|forward|left|right|backward|stop)\b",
        flags=re.IGNORECASE,
    )
    actions = []
    for match in pattern.finditer(text):
        normalized = ACTION_NAME_MAP[match.group(0).lower()]
        actions.append(normalized)
        if len(actions) >= max_actions:
            break
    return actions


def get_rgb(observation: dict[str, Any]) -> np.ndarray:
    rgb = observation["color_sensor"]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8)


def get_depth(observation: dict[str, Any]) -> np.ndarray:
    depth = observation["depth_sensor"]
    return np.squeeze(depth)


def ensure_size_rgb(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a HabitatSim RGB frame to the target (height, width) without
    touching the channel order.

    Earlier versions of this helper used cv2.COLOR_RGB2BGR before returning,
    which silently fed channel-swapped frames to the LLaVA-Video SigLIP
    image_processor (SigLIP is pretrained on RGB and expects RGB). That bug
    caused the visual encoder to see red↔blue swapped images both at eval
    and downstream embedding time. We now keep the frame as RGB so it lines
    up with the SigLIP normalization and with the on-disk JPGs that the
    Stage-1 training dataset (``snav_training/dataset_snav.py``) loads.
    """
    if frame_rgb.shape[:2] != (height, width):
        frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return frame_rgb


# Backwards-compatible alias for any external callers that still import the
# old name. Behaviour is the NEW RGB-preserving behaviour; the old red↔blue
# swap is intentionally gone.
ensure_size_bgr = ensure_size_rgb


def _scene_name_candidates(scene_id: str) -> set[str]:
    scene_path = Path(scene_id)
    candidates = set()
    if scene_path.suffix:
        candidates.add(scene_path.stem.replace(".basis", ""))
    if scene_path.parent.name:
        candidates.add(scene_path.parent.name)
    if scene_path.name:
        candidates.add(scene_path.name.replace(".basis.glb", "").replace(".glb", ""))
    return {candidate for candidate in candidates if candidate}


def resolve_scene_path(scene_id: str, hm3d_base_path: Path) -> Optional[Path]:
    raw_path = Path(scene_id)
    if raw_path.exists():
        return raw_path

    for split in ("train", "val", "test"):
        split_dir = hm3d_base_path / split
        if not split_dir.exists():
            continue
        for scene_name in _scene_name_candidates(scene_id):
            for candidate in split_dir.glob(f"{scene_name}*/*.basis.glb"):
                if candidate.exists():
                    return candidate
    LOGGER.error("Unable to resolve HM3D scene path for scene_id=%s", scene_id)
    return None


def habitat_quaternion_from_wxyz(rotation: list[float]):
    from habitat_sim.utils.common import quat_from_coeffs

    return quat_from_coeffs([rotation[1], rotation[2], rotation[3], rotation[0]])


def create_simulator(
    scene_path: Path,
    frame_width: int,
    frame_height: int,
    include_depth: bool = False,
    gpu_device_id: int | None = None,
):
    import os

    import habitat_sim

    if gpu_device_id is None:
        env_gpu = os.environ.get("NAVSPACE_SIM_GPU_ID")
        gpu_device_id = int(env_gpu) if env_gpu not in (None, "") else -1

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True
    sim_cfg.requires_textures = True
    sim_cfg.gpu_device_id = gpu_device_id

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [frame_height, frame_width]
    rgb_sensor.position = [0.0, 1.5, 0.0]
    rgb_sensor.orientation = [0.0, 0.0, 0.0]
    # Wider HFOV than the habitat-sim 90° default — gives the model more
    # peripheral context (a 90° pinhole crops a lot of side-room/doorway info
    # at 1.5 m height). Training renders MUST use the same HFOV; see
    # snav_training/data_generation/render_streamvln.py.
    rgb_sensor.hfov = 120

    sensors = [rgb_sensor]
    if include_depth:
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth_sensor"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [frame_height, frame_width]
        depth_sensor.position = [0.0, 1.5, 0.0]
        depth_sensor.orientation = [0.0, 0.0, 0.0]
        sensors.append(depth_sensor)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }
    configuration = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    simulator = habitat_sim.Simulator(configuration)
    agent = simulator.initialize_agent(0)
    return simulator, agent


def append_result(
    result_path: Path,
    instruction: str,
    result_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    lock_path = Path(str(result_path) + ".lock")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        if result_path.exists() and result_path.stat().st_size > 0:
            try:
                with result_path.open("r", encoding="utf-8") as file:
                    existing = json.load(file)
            except json.JSONDecodeError:
                corrupt_path = result_path.with_suffix(result_path.suffix + f".corrupt.{os.getpid()}")
                try:
                    result_path.replace(corrupt_path)
                    LOGGER.warning("Moved corrupt JSON result file %s to %s", result_path, corrupt_path)
                except OSError:
                    LOGGER.warning("Failed to move corrupt JSON result file %s", result_path, exc_info=True)
                existing = []
        else:
            existing = []
        shortest_path = float(result_payload.get("shortest_path_length", 0.0))
        for item in existing:
            if not isinstance(item, dict) or not item:
                continue
            existing_instruction = list(item.keys())[0]
            if existing_instruction != instruction:
                continue
            payload = item[existing_instruction]
            try:
                existing_shortest = float(payload.get("shortest_path_length", 0.0))
            except Exception:
                existing_shortest = 0.0
            if abs(existing_shortest - shortest_path) < 1e-6:
                return existing
        existing.append({instruction: result_payload})
        tmp_path = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(existing, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        tmp_path.replace(result_path)
        return existing


def build_resume_index(items: Iterable[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    resume_index: dict[tuple[str, float], dict[str, Any]] = {}
    for item in items:
        instruction = list(item.keys())[0]
        payload = item[instruction]
        shortest_path = float(payload.get("shortest_path_length", 0.0))
        resume_index[(instruction, shortest_path)] = payload
    return resume_index


def summarize_results(items: Iterable[dict[str, Any]]) -> dict[str, float]:
    success_values: list[float] = []
    nav_error_values: list[float] = []
    os_values: list[float] = []
    spl_values: list[float] = []

    for item in items:
        payload = list(item.values())[0]
        if isinstance(payload, dict):
            success = float(payload.get("success", 0.0))
            nav_error = float(payload.get("nav_error", 0.0))
            oracle_success = float(payload.get("os", 0.0))
            shortest = float(payload.get("shortest_path_length", 0.0))
            actual = float(payload.get("actual_path_length", 0.0))
            if not all(math.isfinite(value) for value in (success, nav_error, oracle_success, shortest, actual)):
                continue
            spl = success * (shortest / max(actual, shortest)) if success else 0.0
        elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
            success = float(payload[0])
            nav_error = float(payload[1])
            oracle_success = float(payload[2])
            if not all(math.isfinite(value) for value in (success, nav_error, oracle_success)):
                continue
            spl = 0.0
        else:
            continue
        success_values.append(success)
        nav_error_values.append(nav_error)
        os_values.append(oracle_success)
        spl_values.append(spl)

    if not success_values:
        return {"count": 0, "sr": 0.0, "ne": 0.0, "os": 0.0, "spl": 0.0}

    return {
        "count": float(len(success_values)),
        "sr": float(np.mean(success_values)),
        "ne": float(np.mean(nav_error_values)),
        "os": float(np.mean(os_values)),
        "spl": float(np.mean(spl_values)),
    }


def format_summary(summary: dict[str, float]) -> str:
    return (
        f"[{int(summary['count'])}] Eval Results: "
        f"Success:{summary['sr']:.3f}, Nav Error:{summary['ne']:.2f}, "
        f"OS:{summary['os']:.3f}, SPL:{summary['spl']:.3f}"
    )
