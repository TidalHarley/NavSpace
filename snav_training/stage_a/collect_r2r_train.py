import argparse
import gzip
import json
import os
from pathlib import Path

import cv2
import habitat
from habitat.config.read_write import read_write
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from tqdm import tqdm

from config_utils import r2r_train_config
from paths import MP3D_ROOT, NAVIGATION_ROOT, R2R_ROOT, TRAIN_DATA_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TRAIN_DATA_ROOT / "training_data_r2r_tt"
HABITAT_LAB_ROOT = Path(habitat.__file__).resolve().parents[1]
R2R_CONFIG_PATH = str(HABITAT_LAB_ROOT / "habitat" / "config" / "benchmark" / "nav" / "vln_r2r.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="train")
    parser.add_argument("--part-idx", default=None)
    parser.add_argument("--traj-num", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--save-size", type=int, default=384)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_mp3d_scenes_dir(mp3d_root: Path) -> Path:
    if (mp3d_root / "mp3d").is_dir():
        return mp3d_root

    compat_root = Path("/tmp/habitat_scene_datasets")
    compat_link = compat_root / "mp3d"
    compat_root.mkdir(parents=True, exist_ok=True)

    if compat_link.exists() or compat_link.is_symlink():
        if compat_link.is_symlink() and os.readlink(compat_link) == str(mp3d_root):
            return compat_root
        if compat_link.is_symlink():
            compat_link.unlink()
        else:
            raise RuntimeError(f"Unexpected compatibility path: {compat_link}")

    compat_link.symlink_to(mp3d_root)
    return compat_root


def resolve_dataset_path(stage: str, part_idx: str | None) -> Path:
    if part_idx is None:
        candidate = R2R_ROOT / stage / f"{stage}.json.gz"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing dataset file: {candidate}")

    candidates = [
        R2R_ROOT / stage / f"{stage}_part{part_idx}.json.gz",
        R2R_ROOT / stage / f"{stage}_{part_idx}.json.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = R2R_ROOT / stage / f"{stage}.json.gz"
    if fallback.exists():
        print(f"part dataset not found for part_idx={part_idx}, fallback to {fallback}")
        return fallback

    raise FileNotFoundError(f"Missing dataset file for stage={stage}, part_idx={part_idx}")


def load_episode_count(dataset_path: Path) -> int:
    open_fn = gzip.open if dataset_path.suffix == ".gz" else open
    with open_fn(dataset_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        episodes = payload.get("episodes")
        if episodes is None:
            raise ValueError(f"Dataset file does not contain 'episodes': {dataset_path}")
        return len(episodes)

    if isinstance(payload, list):
        return len(payload)

    raise TypeError(f"Unsupported dataset format in {dataset_path}")


def build_habitat_config(stage: str, part_idx: str | None, img_size: int):
    habitat_config = r2r_train_config(
        path=R2R_CONFIG_PATH,
        stage=stage,
        part_idx=None,
        img_size=img_size,
    )
    dataset_path = resolve_dataset_path(stage, part_idx)
    scenes_dir = get_mp3d_scenes_dir(MP3D_ROOT)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = str(scenes_dir)
        habitat_config.habitat.dataset.data_path = str(dataset_path)
        habitat_config.habitat.simulator.scene_dataset = str(MP3D_ROOT / "mp3d.scene_dataset_config.json")
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = 120

    return habitat_config


def serialize_position(position) -> list[float]:
    if hasattr(position, "tolist"):
        return position.tolist()
    return list(position)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset_path(args.stage, args.part_idx)
    traj_num = args.traj_num if args.traj_num is not None else load_episode_count(dataset_path)
    print(f"collecting {traj_num} trajectories from {dataset_path}")

    habitat_config = build_habitat_config(
        stage=args.stage,
        part_idx=args.part_idx,
        img_size=args.img_size,
    )

    env = habitat.Env(habitat_config)
    path_planner = ShortestPathFollower(env.sim, 1, False)

    try:
        for traj_idx in tqdm(range(traj_num), desc="collect_r2r_train"):
            obs = env.reset()
            ori_traj_id = obs["instruction"]["trajectory_id"]
            traj_data_path = output_root / f"traj_{traj_idx}_{ori_traj_id}"
            gt_act_json_path = traj_data_path / "gt_acts.json"

            if gt_act_json_path.exists() and not args.overwrite:
                continue

            traj_data_path.mkdir(parents=True, exist_ok=True)
            instruction = env.current_episode.instruction.instruction_text
            reference_path = env.current_episode.reference_path
            state = []
            step_idx = 0
            gt_act_sequences = []

            for reference_pos in reference_path[1:]:
                while True:
                    act = path_planner.get_next_action(reference_pos)
                    if act is None:
                        raise RuntimeError(f"ShortestPathFollower returned None at traj_idx={traj_idx}")

                    rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (args.save_size, args.save_size))
                    rgb_save_path = traj_data_path / f"step_{step_idx}_rgb.jpg"
                    cv2.imwrite(str(rgb_save_path), rgb)

                    state.append(
                        serialize_position(env.sim.get_agent_state().sensor_states["rgb"].position)
                    )

                    if act != 0:
                        gt_act_sequences.append(act)
                        obs = env.step(act)
                        step_idx += 1
                    else:
                        break

            gt_act_sequences.append(0)
            env.step(0)

            gt_act_dict = {
                "instruction": instruction,
                "trajectory_id": ori_traj_id,
                "traj_idx": traj_idx,
                "reference_path_len": len(reference_path),
                "state_positions": state,
                "gt_act_sequences": gt_act_sequences,
            }
            with open(gt_act_json_path, "w", encoding="utf-8") as f:
                json.dump(gt_act_dict, f, indent=2, ensure_ascii=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
