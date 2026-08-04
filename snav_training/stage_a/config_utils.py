"""Minimal Habitat config helpers for Stage-A R2R/RxR collect.

Hardcoded absolute paths from the original CorrectNav tree are removed.
``collect_r2r_train*.py`` / ``collect_rxr_train.py`` override
``data_path`` / ``scenes_dir`` / ``scene_dataset`` after calling
``r2r_train_config``.
"""

from __future__ import annotations

import os
from pathlib import Path

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    LookDownActionConfig,
    LookUpActionConfig,
    TopDownMapMeasurementConfig,
)
from habitat.config.read_write import read_write

from paths import MP3D_ROOT, R2R_ROOT


def _default_vln_config_path() -> str:
    habitat_root = Path(habitat.__file__).resolve().parents[1]
    for name in ("vln_r2r.yaml", "vln_rxr.yaml"):
        cand = habitat_root / "habitat" / "config" / "benchmark" / "nav" / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "Cannot locate Habitat VLN yaml under "
        f"{habitat_root}/habitat/config/benchmark/nav/. "
        "Install habitat-lab with VLN configs, or pass path= explicitly."
    )


R2R_CONFIG_PATH = os.environ.get("R2R_HABITAT_CONFIG", _default_vln_config_path())
RXR_CONFIG_PATH = os.environ.get("RXR_HABITAT_CONFIG", R2R_CONFIG_PATH)


def r2r_train_config(
    path: str = R2R_CONFIG_PATH,
    stage: str = "train",
    part_idx=None,
    img_size: int = 384,
):
    """Base VLN train config; callers override dataset/scene paths."""
    del part_idx  # kept for API compatibility with collect scripts
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = str(MP3D_ROOT)
        habitat_config.habitat.dataset.data_path = str(
            R2R_ROOT / "{split}" / "{split}.json.gz"
        )
        habitat_config.habitat.simulator.scene_dataset = str(
            MP3D_ROOT / "mp3d.scene_dataset_config.json"
        )
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update(
            {
                "look_up": LookUpActionConfig(tilt_angle=30),
                "look_down": LookDownActionConfig(tilt_angle=30),
            }
        )

        habitat_config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

        rgb = habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        depth = habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        rgb.width = img_size
        rgb.height = img_size
        rgb.hfov = 90
        depth.width = img_size
        depth.height = img_size
        depth.hfov = 90
        depth.max_depth = 5.0
        depth.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0

    return habitat_config
