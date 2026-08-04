import argparse
import json
import os
import re
import textwrap
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from paths import TRAIN_DATA_ROOT  # noqa: E402

DEFAULT_INPUT_DIR = TRAIN_DATA_ROOT / "training_data_r2r_tt"
DEFAULT_OUTPUT_JSON = TRAIN_DATA_ROOT / "r2r_stepwise_train_jupyter_full.json"
STEP_IMAGE_RE = re.compile(r"^step_(\d+)_rgb\.jpg$")
TRAJ_DIR_RE = re.compile(r"^traj_(\d+)_(.+)$")
SAMPLE_ID_RE = re.compile(r"^sample_(\d+)_(.+)_(\d+)$")


def multi_step_gt(gt_act_sequences, step_idx, gt_step):
    gt_act_lang = []
    for i in range(gt_step):
        idx = int(step_idx) + i
        if idx > len(gt_act_sequences) - 1:
            break
        act = int(gt_act_sequences[idx])
        if act == 1:
            gt_act_lang.append("Move forward")
        elif act == 2:
            gt_act_lang.append("Turn left")
        elif act == 3:
            gt_act_lang.append("Turn right")
        else:
            gt_act_lang.append("Stop")
    while len(gt_act_lang) < gt_step:
        gt_act_lang.append("Stop")
    return ",".join(gt_act_lang)


def build_prompt(instruction, gt_step):
    prompt = f"""
    You are navigating in an indoor environment given the instruction: {instruction};
    You are given the observation history of previous steps you have taken;
    You should:
    1) evaluate the history to decide which step of instruction you are at.
    2) Predict actions for the next {gt_step} steps to follow up the given instruction until you reach the goal;
    Notice that:
    1) You can only choose from the following four actions: Move forward, Turn left, Turn right, Stop;
    2) Move forward means to move 0.25 meters straight ahead, and turning left or right is a 30-degree turn.
    3) If you believe you have reached the target or caught in obstacles, you should choose the stop action.
    ----
    Starting below, you should strictly follow this format:
    Final Answer: Your predicted actions for the next {gt_step} steps
    """
    return textwrap.dedent(prompt).strip()


def sorted_traj_dirs(raw_training_data_path):
    traj_dirs = []
    for entry in raw_training_data_path.iterdir():
        if not entry.is_dir():
            continue
        match = TRAJ_DIR_RE.match(entry.name)
        if match:
            traj_dirs.append((int(match.group(1)), entry.name, entry))
        else:
            traj_dirs.append((10**9, entry.name, entry))
    traj_dirs.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in traj_dirs]


def sorted_step_images(traj_dir):
    step_images = []
    for entry in traj_dir.iterdir():
        if not entry.is_file():
            continue
        match = STEP_IMAGE_RE.match(entry.name)
        if not match:
            continue
        step_images.append((int(match.group(1)), entry))
    step_images.sort(key=lambda item: item[0])
    return [entry for _, entry in step_images]


def select_history_images(image_paths, step_idx, max_images):
    history = image_paths[: step_idx + 1]
    if len(history) <= max_images:
        return history

    last_idx = len(history) - 1
    sampled = []
    seen = set()
    for i in range(max_images):
        idx = round(i * last_idx / (max_images - 1))
        if idx not in seen:
            sampled.append(idx)
            seen.add(idx)

    if len(sampled) < max_images:
        for idx in range(last_idx + 1):
            if idx in seen:
                continue
            sampled.append(idx)
            seen.add(idx)
            if len(sampled) == max_images:
                break

    sampled.sort()
    return [history[idx] for idx in sampled]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing traj_xxx folders with step_*_rgb.jpg and gt_acts.json.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--gt-step",
        type=int,
        default=6,
        help="How many future actions to predict.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        help="Maximum number of history images per sample.",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=-1,
        help="Optional trajectory cap for debugging. -1 means all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional sample cap for debugging. -1 means all.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N processed trajectories. Set <=0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Rewrite output JSON every N processed trajectories. Set <=0 to only write at the end.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output JSON and regenerate from scratch.",
    )
    return parser.parse_args()


def write_output_json(output_json_path, payload):
    tmp_path = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_json_path)


def parse_sample_id(sample_id):
    match = SAMPLE_ID_RE.match(sample_id)
    if match is None:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def load_resume_state(output_json_path, overwrite):
    if overwrite or not output_json_path.exists():
        return [], set(), 0

    with open(output_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise TypeError(f"Existing output JSON must be a list: {output_json_path}")

    processed_traj_names = set()
    max_sample_id = -1
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("Existing output JSON contains a non-object sample.")
        parsed = parse_sample_id(item.get("id", ""))
        if parsed is None:
            raise ValueError(f"Unexpected sample id in existing output: {item.get('id')}")
        sample_id, traj_name, _ = parsed
        max_sample_id = max(max_sample_id, sample_id)
        processed_traj_names.add(traj_name)

    return payload, processed_traj_names, max_sample_id + 1


def main():
    args = parse_args()
    raw_training_data_path = Path(args.input_dir)
    output_json_path = Path(args.output_json)

    if args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if not raw_training_data_path.exists():
        raise FileNotFoundError(f"Input directory not found: {raw_training_data_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    vlnce_image_dict, processed_traj_names, sample_id = load_resume_state(
        output_json_path=output_json_path,
        overwrite=args.overwrite,
    )
    traj_dir_list = sorted_traj_dirs(raw_training_data_path)
    if args.max_trajs > 0:
        traj_dir_list = traj_dir_list[: args.max_trajs]

    print(f"input_dir={raw_training_data_path}")
    print(f"traj_count={len(traj_dir_list)}")
    print(f"resume_existing_trajs={len(processed_traj_names)}")
    print(f"resume_existing_samples={len(vlnce_image_dict)}")
    print(f"next_sample_id={sample_id}")

    processed_trajs = len(processed_traj_names)
    for traj_idx, traj_dir in enumerate(traj_dir_list, start=1):
        if traj_dir.name in processed_traj_names:
            continue

        gt_act_path = traj_dir / "gt_acts.json"
        if not gt_act_path.exists():
            continue

        with open(gt_act_path, "r", encoding="utf-8") as file:
            gt_acts_dict = json.load(file)

        instruction = gt_acts_dict["instruction"]
        gt_act_sequences = gt_acts_dict["gt_act_sequences"]
        image_paths = sorted_step_images(traj_dir)
        valid_steps = min(len(image_paths), len(gt_act_sequences))

        for step_idx in range(valid_steps):
            if args.max_samples > 0 and sample_id >= args.max_samples:
                break

            history_images = select_history_images(
                image_paths=image_paths,
                step_idx=step_idx,
                max_images=args.max_images,
            )
            if not history_images:
                continue

            history_rel = [os.path.relpath(path, TRAIN_DATA_ROOT) for path in history_images]
            image_tokens = "\n".join(["<image>"] * len(history_rel))
            multi_step_prompt = build_prompt(instruction, args.gt_step)
            gt_act = multi_step_gt(gt_act_sequences, step_idx, args.gt_step)

            new_conversation = [
                {"from": "human", "value": f"{image_tokens}\n{multi_step_prompt}"},
                {"from": "gpt", "value": f"Final Answer: {gt_act}"},
            ]

            new_data = {
                "id": f"sample_{sample_id:07d}_{traj_dir.name}_{step_idx:04d}",
                "conversations": new_conversation,
                "image": history_rel,
                "data_source": "r2r",
            }
            vlnce_image_dict.append(new_data)
            sample_id += 1

        processed_trajs += 1
        processed_traj_names.add(traj_dir.name)

        if args.progress_every > 0 and processed_trajs % args.progress_every == 0:
            print(
                f"processed_trajs={processed_trajs}/{len(traj_dir_list)} "
                f"samples_written={sample_id} last_traj={traj_dir.name}",
                flush=True,
            )

        if args.checkpoint_every > 0 and processed_trajs % args.checkpoint_every == 0:
            write_output_json(output_json_path, vlnce_image_dict)
            print(
                f"checkpoint_output={output_json_path} "
                f"checkpoint_samples={len(vlnce_image_dict)}",
                flush=True,
            )

        if args.max_samples > 0 and sample_id >= args.max_samples:
            break

    write_output_json(output_json_path, vlnce_image_dict)

    print(f"output_json={output_json_path}")
    print(f"samples_written={len(vlnce_image_dict)}")


if __name__ == "__main__":
    main()
