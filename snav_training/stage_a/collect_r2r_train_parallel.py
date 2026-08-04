#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
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
DEFAULT_RUNS_DIR = TRAIN_DATA_ROOT / "r2r_collect_runs"
HABITAT_LAB_ROOT = Path(habitat.__file__).resolve().parents[1]
R2R_CONFIG_PATH = str(HABITAT_LAB_ROOT / "habitat" / "config" / "benchmark" / "nav" / "vln_r2r.yaml")


def write_json(path: Path, payload) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


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


def load_dataset_payload(dataset_path: Path) -> dict:
    open_fn = gzip.open if dataset_path.suffix == ".gz" else open
    with open_fn(dataset_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "episodes" not in payload:
        raise ValueError(f"Dataset file does not contain 'episodes': {dataset_path}")
    return payload


def load_episode_count(dataset_path: Path) -> int:
    return len(load_dataset_payload(dataset_path).get("episodes") or [])


def build_dataset_shards(dataset_path: Path, run_dir: Path, num_workers: int, max_traj_num: int | None):
    payload = load_dataset_payload(dataset_path)
    episodes = payload.get("episodes") or []
    if max_traj_num is not None and max_traj_num > 0:
        episodes = episodes[:max_traj_num]

    for dataset_index, episode in enumerate(episodes):
        info = dict(episode.get("info") or {})
        info["dataset_index"] = dataset_index
        episode["info"] = info

    shard_dir = run_dir / "dataset_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_specs = []
    for worker_index in range(num_workers):
        shard_episodes = [
            episode
            for episode in episodes
            if int(episode.get("info", {}).get("dataset_index", -1)) % num_workers == worker_index
        ]
        shard_payload = dict(payload)
        shard_payload["episodes"] = shard_episodes
        shard_path = shard_dir / f"worker_{worker_index:02d}.json.gz"
        with gzip.open(shard_path, "wt", encoding="utf-8") as f:
            json.dump(shard_payload, f, ensure_ascii=False)
        shard_specs.append(
            {
                "worker_index": worker_index,
                "dataset_path": str(shard_path),
                "episode_count": len(shard_episodes),
            }
        )

    write_json(
        run_dir / "dataset_manifest.json",
        {
            "source_dataset_path": str(dataset_path),
            "episode_count": len(episodes),
            "num_workers": num_workers,
            "max_traj_num": max_traj_num,
            "created_at_epoch": time.time(),
            "shards": shard_specs,
        },
    )
    return shard_specs


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


def build_habitat_config(dataset_path: Path, stage: str, img_size: int):
    habitat_config = r2r_train_config(
        path=R2R_CONFIG_PATH,
        stage=stage,
        part_idx=None,
        img_size=img_size,
    )
    scenes_dir = get_mp3d_scenes_dir(MP3D_ROOT)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = str(scenes_dir)
        habitat_config.habitat.dataset.data_path = str(dataset_path)
        habitat_config.habitat.simulator.scene_dataset = str(MP3D_ROOT / "mp3d.scene_dataset_config.json")
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = 120

    return habitat_config


def serialize_position(position):
    if hasattr(position, "tolist"):
        return position.tolist()
    return list(position)


def collect_worker(args):
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    total_episodes = load_episode_count(dataset_path)
    print(
        f"worker {args.worker_index + 1}/{args.num_workers} collecting {total_episodes} trajectories from {dataset_path}",
        flush=True,
    )

    habitat_config = build_habitat_config(
        dataset_path=dataset_path,
        stage=args.stage,
        img_size=args.img_size,
    )

    env = habitat.Env(habitat_config)
    path_planner = ShortestPathFollower(env.sim, 1, False)
    processed = 0

    try:
        for local_traj_idx in tqdm(range(total_episodes), desc=f"collect_r2r[{args.worker_index + 1}/{args.num_workers}]"):
            obs = env.reset()
            current_episode = env.current_episode
            info = dict(getattr(current_episode, "info", {}) or {})
            global_traj_idx = int(info.get("dataset_index", local_traj_idx))
            ori_traj_id = obs["instruction"]["trajectory_id"]
            traj_data_path = output_root / f"traj_{global_traj_idx}_{ori_traj_id}"
            gt_act_json_path = traj_data_path / "gt_acts.json"

            if gt_act_json_path.exists() and not args.overwrite:
                processed += 1
                continue

            traj_data_path.mkdir(parents=True, exist_ok=True)
            instruction = current_episode.instruction.instruction_text
            reference_path = current_episode.reference_path
            state = []
            step_idx = 0
            gt_act_sequences = []

            for reference_pos in reference_path[1:]:
                while True:
                    act = path_planner.get_next_action(reference_pos)
                    if act is None:
                        raise RuntimeError(f"ShortestPathFollower returned None at global_traj_idx={global_traj_idx}")

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
                "traj_idx": global_traj_idx,
                "local_worker_traj_idx": local_traj_idx,
                "worker_index": args.worker_index,
                "num_workers": args.num_workers,
                "reference_path_len": len(reference_path),
                "state_positions": state,
                "gt_act_sequences": gt_act_sequences,
            }
            with open(gt_act_json_path, "w", encoding="utf-8") as f:
                json.dump(gt_act_dict, f, indent=2, ensure_ascii=False)

            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"worker {args.worker_index + 1}/{args.num_workers} processed={processed}/{total_episodes} "
                    f"last_traj=traj_{global_traj_idx}_{ori_traj_id}",
                    flush=True,
                )
    finally:
        env.close()


def launch_workers(args):
    source_dataset_path = Path(args.dataset_path) if args.dataset_path else resolve_dataset_path(args.stage, args.part_idx)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}_utc"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    shard_specs = build_dataset_shards(
        dataset_path=source_dataset_path,
        run_dir=run_dir,
        num_workers=args.num_workers,
        max_traj_num=args.traj_num,
    )

    worker_specs = []
    script_path = Path(__file__).resolve()
    for spec in shard_specs:
        worker_index = spec["worker_index"]
        log_path = run_dir / f"worker_{worker_index:02d}.log"
        cmd = [
            sys.executable,
            str(script_path),
            "--mode",
            "worker",
            "--stage",
            str(args.stage),
            "--dataset-path",
            str(spec["dataset_path"]),
            "--img-size",
            str(args.img_size),
            "--save-size",
            str(args.save_size),
            "--output-dir",
            str(args.output_dir),
            "--num-workers",
            str(args.num_workers),
            "--worker-index",
            str(worker_index),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(SCRIPT_DIR),
                start_new_session=True,
            )
        worker_specs.append(
            {
                "worker_index": worker_index,
                "episode_count": spec["episode_count"],
                "pid": process.pid,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )

    launch_payload = {
        "run_dir": str(run_dir),
        "source_dataset_path": str(source_dataset_path),
        "output_dir": str(args.output_dir),
        "num_workers": args.num_workers,
        "traj_num": args.traj_num,
        "img_size": args.img_size,
        "save_size": args.save_size,
        "overwrite": args.overwrite,
        "launch_time_epoch": time.time(),
        "workers": worker_specs,
    }
    write_json(run_dir / "workers.json", launch_payload)
    print(json.dumps(launch_payload, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="launch", choices=["launch", "worker"])
    parser.add_argument("--stage", default="train")
    parser.add_argument("--part-idx", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--traj-num", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--save-size", type=int, default=384)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must be within [0, num_workers).")

    if args.mode == "launch":
        launch_workers(args)
        return

    if args.dataset_path is None:
        args.dataset_path = str(resolve_dataset_path(args.stage, args.part_idx))
    collect_worker(args)


if __name__ == "__main__":
    main()
