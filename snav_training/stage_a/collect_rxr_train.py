import argparse
import gzip
import json
import os
import re
from pathlib import Path

import attr
import cv2
import habitat
from habitat.config.read_write import read_write
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from tqdm import tqdm

from config_utils import r2r_train_config
from paths import MP3D_ROOT, NAVIGATION_ROOT, RXR_ROOT, TRAIN_DATA_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent

def _iter_navigation_root_candidates():
    yield NAVIGATION_ROOT
    yield SCRIPT_DIR.parents[1]
    if len(SCRIPT_DIR.parents) > 2:
        yield SCRIPT_DIR.parents[2]

OUTPUT_ROOT = TRAIN_DATA_ROOT / "training_data_rxr_ce_tt"
NUM_WORKERS_ENV = "RXR_COLLECT_NUM_WORKERS"
WORKER_INDEX_ENV = "RXR_COLLECT_WORKER_INDEX"
MAX_TRAJ_NUM_ENV = "RXR_COLLECT_MAX_TRAJ_NUM"


def parse_language_prefixes(raw_value: str | None):
    if raw_value is None:
        return ("en",)
    prefixes = tuple(
        item.strip() for item in str(raw_value).split(",") if item.strip()
    )
    return prefixes or None


def language_matches(language: str | None, allowed_prefixes) -> bool:
    if allowed_prefixes is None:
        return True
    if language is None:
        return False
    return any(str(language).startswith(prefix) for prefix in allowed_prefixes)


@attr.s(auto_attribs=True)
class RxRInstructionData(InstructionData):
    instruction_id: str | None = None
    language: str | None = None
    annotator_id: str | None = None
    edit_distance: float | None = None
    timed_instruction: list[dict] | None = None


def get_optional_attr(obj, attr_name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def validate_worker_args(num_workers: int, worker_index: int) -> None:
    if num_workers < 1:
        raise ValueError(f"--num-workers must be >= 1, got {num_workers}")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(
            f"--worker-index must be in [0, {num_workers - 1}], got {worker_index}"
        )


def configure_worker_env(
    num_workers: int,
    worker_index: int,
    max_traj_num: int | None,
) -> None:
    os.environ[NUM_WORKERS_ENV] = str(num_workers)
    os.environ[WORKER_INDEX_ENV] = str(worker_index)

    if max_traj_num is None:
        os.environ.pop(MAX_TRAJ_NUM_ENV, None)
    else:
        os.environ[MAX_TRAJ_NUM_ENV] = str(max_traj_num)


def get_worker_settings() -> tuple[int, int, int | None]:
    num_workers = int(os.environ.get(NUM_WORKERS_ENV, "1"))
    worker_index = int(os.environ.get(WORKER_INDEX_ENV, "0"))
    max_traj_num_raw = os.environ.get(MAX_TRAJ_NUM_ENV)
    max_traj_num = int(max_traj_num_raw) if max_traj_num_raw is not None else None
    validate_worker_args(num_workers, worker_index)
    return num_workers, worker_index, max_traj_num


def apply_episode_shard(episodes: list[VLNEpisode]) -> list[VLNEpisode]:
    num_workers, worker_index, max_traj_num = get_worker_settings()
    if num_workers == 1 and max_traj_num is None:
        return episodes

    sharded_episodes = []
    for episode in episodes:
        dataset_index = get_optional_attr(
            get_optional_attr(episode, "info"), "dataset_index"
        )
        if dataset_index is None:
            raise ValueError("RxR episode is missing info.dataset_index")
        if max_traj_num is not None and dataset_index >= max_traj_num:
            continue
        if dataset_index % num_workers != worker_index:
            continue
        sharded_episodes.append(episode)

    return sharded_episodes


@registry.register_dataset(name="RxRVLN-v1")
class RxRDatasetV1(Dataset):
    episodes: list[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config=None) -> None:
        self.episodes = []
        self.instruction_vocab = VocabDict(word_list=[])

        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt", encoding="utf-8") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )
        # Filter by instruction.language when RXR_ALLOWED_LANGUAGE_PREFIXES is set
        # (e.g. "en" → en-US + en-IN). Reads language from RxRInstructionData, not a
        # pre-filtered json.gz.
        allowed_raw = os.environ.get("RXR_ALLOWED_LANGUAGE_PREFIXES")
        if allowed_raw is not None:
            prefixes = parse_language_prefixes(allowed_raw)
            if prefixes is not None:
                before = len(self.episodes)
                self.episodes = [
                    ep
                    for ep in self.episodes
                    if language_matches(
                        getattr(ep.instruction, "language", None), prefixes
                    )
                ]
                print(
                    f"[RxRVLN-v1] language filter prefixes={prefixes}: "
                    f"{before} -> {len(self.episodes)} episodes"
                )
        self.episodes = apply_episode_shard(self.episodes)

    def from_json(self, json_str: str, scenes_dir: str | None = None) -> None:
        deserialized = json.loads(json_str)
        episodes = deserialized.get("episodes")
        if episodes is None:
            raise ValueError("RxR dataset file does not contain 'episodes'.")

        vocab_words = set()
        for dataset_index, raw_episode in enumerate(episodes):
            episode = dict(raw_episode)
            episode_info = dict(episode.get("info") or {})
            episode_info["dataset_index"] = dataset_index
            episode["info"] = episode_info
            raw_instruction = dict(episode.get("instruction") or {})
            instruction_text = raw_instruction.get("instruction_text", "")
            instruction_tokens = raw_instruction.get("instruction_tokens")
            if instruction_tokens is None:
                instruction_tokens = instruction_text.split()
            vocab_words.update(instruction_tokens)

            instruction_payload = {
                "instruction_text": instruction_text,
                "instruction_tokens": instruction_tokens,
            }
            episode["instruction"] = instruction_payload

            vln_episode = VLNEpisode(**episode)

            if scenes_dir is not None:
                scene_id = vln_episode.scene_id
                default_prefix = "data/scene_datasets/"
                if scene_id.startswith(default_prefix):
                    scene_id = scene_id[len(default_prefix) :]
                vln_episode.scene_id = os.path.join(scenes_dir, scene_id)

            vln_episode.instruction = RxRInstructionData(
                instruction_text=instruction_text,
                instruction_tokens=instruction_tokens,
                instruction_id=raw_instruction.get("instruction_id"),
                language=raw_instruction.get("language"),
                annotator_id=raw_instruction.get("annotator_id"),
                edit_distance=raw_instruction.get("edit_distance"),
                timed_instruction=raw_instruction.get("timed_instruction"),
            )
            for goal_idx, goal in enumerate(vln_episode.goals):
                vln_episode.goals[goal_idx] = NavigationGoal(**goal)
            self.episodes.append(vln_episode)

        self.instruction_vocab = VocabDict(word_list=sorted(vocab_words))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="train")
    parser.add_argument("--role", default="guide", choices=["guide", "follower"])
    parser.add_argument("--use-gt", action="store_true")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--traj-num", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--save-size", type=int, default=384)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allowed-language-prefixes",
        default="en",
        help="Comma-separated allowed language prefixes. Default keeps English only. Use empty string to keep all.",
    )
    return parser.parse_args()


def resolve_vln_config_path() -> Path:
    habitat_root = Path(habitat.__file__).resolve().parents[1]
    candidates = [
        habitat_root / "habitat" / "config" / "benchmark" / "nav" / "vln_rxr.yaml",
        habitat_root / "habitat" / "config" / "benchmark" / "nav" / "vln_r2r.yaml",
    ]

    for navigation_root in _iter_navigation_root_candidates():
        habitat_lab_root = navigation_root / "ovon" / "habitat-lab" / "habitat-lab"
        candidates.extend(
            [
                habitat_lab_root / "habitat" / "config" / "benchmark" / "nav" / "vln_rxr.yaml",
                habitat_lab_root / "habitat" / "config" / "benchmark" / "nav" / "vln_r2r.yaml",
            ]
        )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate a VLN Habitat config. Searched:\n{searched}")


VLN_CONFIG_PATH = str(resolve_vln_config_path())


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


def resolve_dataset_path(
    stage: str,
    role: str,
    use_gt: bool,
    dataset_path: str | None,
) -> Path:
    if dataset_path is not None:
        candidate = Path(dataset_path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing dataset file: {candidate}")

    suffix = "_gt" if use_gt else ""
    candidate = RXR_ROOT / stage / f"{stage}_{role}{suffix}.json.gz"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Missing dataset file: {candidate}")


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


def build_habitat_config(stage: str, img_size: int, dataset_path: Path):
    habitat_config = r2r_train_config(
        path=VLN_CONFIG_PATH,
        stage=stage,
        part_idx=None,
        img_size=img_size,
    )
    scenes_dir = get_mp3d_scenes_dir(MP3D_ROOT)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.type = "RxRVLN-v1"
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


def sanitize_component(value) -> str:
    if value is None:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(value)).strip("-") or "unknown"


def get_role(info) -> str | None:
    if info is None:
        return None
    if isinstance(info, dict):
        return info.get("role")
    return getattr(info, "role", None)


def main() -> None:
    args = parse_args()
    allowed_language_prefixes = parse_language_prefixes(args.allowed_language_prefixes)
    validate_worker_args(args.num_workers, args.worker_index)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset_path(
        stage=args.stage,
        role=args.role,
        use_gt=args.use_gt,
        dataset_path=args.dataset_path,
    )
    episode_count = load_episode_count(dataset_path)
    max_traj_num = episode_count if args.traj_num is None else min(args.traj_num, episode_count)
    configure_worker_env(
        num_workers=args.num_workers,
        worker_index=args.worker_index,
        max_traj_num=max_traj_num,
    )
    print(f"using habitat config: {VLN_CONFIG_PATH}")
    print(
        "collecting up to "
        f"{max_traj_num} trajectories from {dataset_path} "
        f"with worker {args.worker_index + 1}/{args.num_workers}"
    )

    habitat_config = build_habitat_config(
        stage=args.stage,
        img_size=args.img_size,
        dataset_path=dataset_path,
    )

    env = habitat.Env(habitat_config)
    traj_num = len(env.episodes)
    print(f"worker {args.worker_index + 1}/{args.num_workers} will collect {traj_num} trajectories")
    path_planner = ShortestPathFollower(env.sim, 1, False)

    try:
        for worker_traj_idx in tqdm(
            range(traj_num),
            desc=f"collect_rxr_ce_train[{args.worker_index + 1}/{args.num_workers}]",
        ):
            obs = env.reset()
            episode = env.current_episode
            instruction_obj = get_optional_attr(episode, "instruction")
            episode_info = get_optional_attr(episode, "info")
            traj_idx = get_optional_attr(episode_info, "dataset_index", worker_traj_idx)
            episode_id = get_optional_attr(episode, "episode_id")
            trajectory_id = get_optional_attr(episode, "trajectory_id")
            instruction_id = get_optional_attr(instruction_obj, "instruction_id")
            instruction = get_optional_attr(instruction_obj, "instruction_text", "")
            language = get_optional_attr(instruction_obj, "language")
            annotator_id = get_optional_attr(instruction_obj, "annotator_id")
            reference_path = episode.reference_path

            if not language_matches(language, allowed_language_prefixes):
                continue

            traj_dir_name = "_".join(
                [
                    f"traj_{traj_idx}",
                    f"ep_{sanitize_component(episode_id)}",
                    f"trajid_{sanitize_component(trajectory_id)}",
                    f"inst_{sanitize_component(instruction_id)}",
                ]
            )
            traj_data_path = output_root / traj_dir_name
            gt_act_json_path = traj_data_path / "gt_acts.json"

            if gt_act_json_path.exists() and not args.overwrite:
                continue

            traj_data_path.mkdir(parents=True, exist_ok=True)
            state = []
            step_idx = 0
            gt_act_sequences = []

            for reference_pos in reference_path[1:]:
                while True:
                    act = path_planner.get_next_action(reference_pos)
                    if act is None:
                        raise RuntimeError(
                            "ShortestPathFollower returned None at "
                            f"traj_idx={traj_idx}, worker_traj_idx={worker_traj_idx}"
                        )

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
                "instruction_id": instruction_id,
                "language": language,
                "annotator_id": annotator_id,
                "episode_id": episode_id,
                "trajectory_id": trajectory_id,
                "role": get_role(episode_info),
                "scene_id": get_optional_attr(episode, "scene_id"),
                "traj_idx": traj_idx,
                "worker_traj_idx": worker_traj_idx,
                "worker_index": args.worker_index,
                "num_workers": args.num_workers,
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
