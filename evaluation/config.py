from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


TASK_DATASET_MAP = {
    "environment_state": Path("NavSpace-Datasets/Environment State/envstate_vln.json"),
    "space_structure": Path("NavSpace-Datasets/Space Structure/spacestructure_vln.json"),
    "precise_movement": Path("NavSpace-Datasets/Precise Movement/precisemove_vln.json"),
    "viewpoint_shifting": Path("NavSpace-Datasets/Viewpoint Shifting/viewpointsft_vln.json"),
    "vertical_perception": Path("NavSpace-Datasets/Vertical Perception/verticalpercep_vln.json"),
    "spatial_relationship": Path("NavSpace-Datasets/Spatial Relationship/spatialrel_vln.json"),
}


PROFILE_DEFAULTS = {
    "gemini-pro": {
        "provider": "openai_compatible",
        "model": "gemini-2.5-pro",
        "base_url": "https://api.chatanywhere.tech/v1",
        "key_env": "OPENAI_API_KEY",
        "key_file": None,
        "frame_width": 224,
        "frame_height": 224,
        "encode_resize": 224,
        "max_retries": 4,
        "initial_retry_delay": 2.0,
        "use_system_message": True,
    },
    "gemini-flash": {
        "provider": "openai_compatible",
        "model": "gemini-2.5-flash",
        "base_url": "https://api.chatanywhere.tech/v1",
        "key_env": "OPENAI_API_KEY",
        "key_file": None,
        "frame_width": 224,
        "frame_height": 224,
        "encode_resize": 224,
        "max_retries": 4,
        "initial_retry_delay": 2.0,
        "use_system_message": True,
    },
    "qwen72b": {
        "provider": "openai_compatible",
        "model": "qwen2.5-vl-72b-instruct",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "key_file": None,
        "frame_width": 224,
        "frame_height": 224,
        "encode_resize": 224,
        "max_retries": 5,
        "initial_retry_delay": 4.0,
        "use_system_message": True,
    },
    "glm4.5v": {
        "provider": "zhipu",
        "model": "glm-4.5v",
        "base_url": None,
        "key_env": "ZHIPU_API_KEY",
        "key_file": None,
        "frame_width": 224,
        "frame_height": 224,
        "encode_resize": 224,
        "max_retries": 5,
        "initial_retry_delay": 4.0,
        "use_system_message": False,
    },
    "glm-4.1v-thinking-flash": {
        "provider": "zhipu",
        "model": "glm-4.1v-thinking-flash",
        "base_url": None,
        "key_env": "ZHIPU_API_KEY",
        "key_file": None,
        "frame_width": 224,
        "frame_height": 224,
        "encode_resize": 224,
        "max_retries": 5,
        "initial_retry_delay": 4.0,
        "use_system_message": False,
    },
}


@dataclass
class LlmEvalConfig:
    profile: str
    provider: str
    model: str
    base_url: Optional[str]
    api_key: str
    output_dir: Path
    result_path: Path
    log_path: Path
    task: str
    trajectory_path: Path
    hm3d_base_path: Path
    shard_id: int = 0
    num_shards: int = 1
    frame_width: int = 224
    frame_height: int = 224
    encode_resize: Optional[int] = 224
    max_frames_num: int = 8
    target_fps: float = 1.0
    max_steps: int = 150
    success_distance: float = 3.0
    actions_per_inference: int = 4
    future_steps_prompt: int = 6
    max_retries: int = 4
    initial_retry_delay: float = 2.0
    use_system_message: bool = True
    system_prompt: str = "you are a helpful assistant"
    resume_from: Optional[Path] = None
    deviation_guard: bool = True
    deviation_patience: int = 12
    request_timeout: float = 120.0
