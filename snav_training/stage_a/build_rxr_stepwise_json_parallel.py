#!/usr/bin/env python3
import argparse
import gzip
import heapq
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from paths import RXR_ROOT, TRAIN_DATA_ROOT  # noqa: E402

DEFAULT_INPUT_DIR = TRAIN_DATA_ROOT / "training_data_rxr_ce_tt"
# Default matches Stage-A YAML (English-only; see --allowed-language-prefixes).
DEFAULT_OUTPUT_JSON = TRAIN_DATA_ROOT / "rxr_stepwise_train_jupyter_full_en.json"
DEFAULT_RUNS_DIR = TRAIN_DATA_ROOT / "rxr_stepwise_runs"
DEFAULT_DATASET_PATH = RXR_ROOT / "train" / "train_guide.json.gz"
STEP_IMAGE_RE = re.compile(r"^step_(\d+)_rgb\.jpg$")
TRAJ_DIR_RE = re.compile(r"^traj_(\d+)_")


def parse_language_prefixes(raw_value):
    if raw_value is None:
        return ("en",)
    prefixes = tuple(
        item.strip() for item in str(raw_value).split(",") if item.strip()
    )
    return prefixes or None


def language_matches(language, allowed_prefixes):
    if allowed_prefixes is None:
        return True
    if language is None:
        return False
    return any(str(language).startswith(prefix) for prefix in allowed_prefixes)


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


def parse_traj_idx(name):
    match = TRAJ_DIR_RE.match(name)
    if match is None:
        return 10**9
    return int(match.group(1))


def sanitize_component(value):
    if value is None:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(value)).strip("-") or "unknown"


def sorted_traj_dirs(raw_training_data_path):
    traj_dirs = []
    for entry in raw_training_data_path.iterdir():
        if not entry.is_dir():
            continue
        traj_dirs.append((parse_traj_idx(entry.name), entry.name, entry))
    traj_dirs.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in traj_dirs]


def sorted_step_images(traj_dir):
    step_images = []
    for entry in traj_dir.iterdir():
        if not entry.is_file():
            continue
        match = STEP_IMAGE_RE.match(entry.name)
        if match is None:
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


def write_output_json(output_json_path, payload_iter):
    tmp_path = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        count = 0
        for item in payload_iter:
            if not first:
                f.write(",\n")
            json.dump(item, f, ensure_ascii=False)
            first = False
            count += 1
        f.write("\n]\n")
    tmp_path.replace(output_json_path)
    return count


def iter_valid_jsonl(path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"skip_invalid_jsonl path={path} line={line_no}",
                    flush=True,
                )


def load_completed_trajs(shard_path):
    completed_trajs = set()
    sample_count = 0
    traj_count = 0
    for bundle in iter_valid_jsonl(shard_path):
        traj_name = bundle.get("traj_name")
        samples = bundle.get("samples") or []
        if traj_name is None or not isinstance(samples, list):
            continue
        completed_trajs.add(traj_name)
        sample_count += len(samples)
        traj_count += 1
    return completed_trajs, traj_count, sample_count


def build_sample_bundle(traj_dir, image_root, gt_step, max_images, allowed_language_prefixes):
    gt_act_path = traj_dir / "gt_acts.json"
    if not gt_act_path.exists():
        return None

    with open(gt_act_path, "r", encoding="utf-8") as file:
        gt_acts_dict = json.load(file)

    instruction = gt_acts_dict["instruction"]
    language = gt_acts_dict.get("language")
    if not language_matches(language, allowed_language_prefixes):
        return None
    gt_act_sequences = gt_acts_dict["gt_act_sequences"]
    image_paths = sorted_step_images(traj_dir)
    valid_steps = min(len(image_paths), len(gt_act_sequences))
    if valid_steps <= 0:
        return None

    samples = []
    for step_idx in range(valid_steps):
        history_images = select_history_images(
            image_paths=image_paths,
            step_idx=step_idx,
            max_images=max_images,
        )
        if not history_images:
            continue

        history_rel = [os.path.relpath(path, image_root) for path in history_images]
        image_tokens = "\n".join(["<image>"] * len(history_rel))
        multi_step_prompt = build_prompt(instruction, gt_step)
        gt_act = multi_step_gt(gt_act_sequences, step_idx, gt_step)

        samples.append(
            {
                "step_idx": step_idx,
                "conversations": [
                    {"from": "human", "value": f"{image_tokens}\n{multi_step_prompt}"},
                    {"from": "gpt", "value": f"Final Answer: {gt_act}"},
                ],
                "image": history_rel,
                "data_source": "rxr",
            }
        )

    if not samples:
        return None

    return {
        "traj_name": traj_dir.name,
        "traj_idx": parse_traj_idx(traj_dir.name),
        "samples": samples,
    }


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def format_worker_name(worker_index, num_workers):
    return f"{worker_index + 1}/{num_workers}"


def build_worker_meta(args, assigned_traj_count, processed_traj_count, sample_count, shard_path, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    return {
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "assigned_traj_count": assigned_traj_count,
        "processed_traj_count": processed_traj_count,
        "sample_count": sample_count,
        "elapsed_sec": round(elapsed, 2),
        "samples_per_sec": round(sample_count / elapsed, 2),
        "trajs_per_sec": round(processed_traj_count / elapsed, 4),
        "shard_path": str(shard_path),
        "finished": processed_traj_count >= assigned_traj_count,
        "updated_at_epoch": time.time(),
    }


def write_json(path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_traj_manifest(input_dir, run_dir, max_trajs):
    manifest_path = run_dir / "traj_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        traj_items = payload.get("traj_items") or []
        return [
            (int(item["traj_idx"]), str(item["traj_name"]))
            for item in traj_items
        ]

    traj_dir_list = sorted_traj_dirs(input_dir)
    if max_trajs > 0:
        traj_dir_list = traj_dir_list[: max_trajs]
    return [(parse_traj_idx(traj_dir.name), traj_dir.name) for traj_dir in traj_dir_list]


def build_traj_manifest_from_dataset(dataset_path, max_trajs, allowed_language_prefixes):
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    episodes = payload.get("episodes") or []
    traj_items = []
    for traj_idx, episode in enumerate(episodes):
        instruction = episode.get("instruction") or {}
        language = instruction.get("language")
        if not language_matches(language, allowed_language_prefixes):
            continue
        traj_name = "_".join(
            [
                f"traj_{traj_idx}",
                f"ep_{sanitize_component(episode.get('episode_id'))}",
                f"trajid_{sanitize_component(episode.get('trajectory_id'))}",
                f"inst_{sanitize_component(instruction.get('instruction_id'))}",
            ]
        )
        traj_items.append(
            {
                "traj_idx": traj_idx,
                "traj_name": traj_name,
                "language": language,
            }
        )
        if max_trajs > 0 and len(traj_items) >= max_trajs:
            break

    return traj_items


def run_worker(args):
    start_time = time.time()
    input_dir = Path(args.input_dir)
    image_root = Path(args.image_root)
    run_dir = Path(args.run_dir)
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_path = shard_dir / f"worker_{args.worker_index:02d}.jsonl"
    meta_path = shard_dir / f"worker_{args.worker_index:02d}.meta.json"

    completed_traj_names, processed_traj_count, sample_count = load_completed_trajs(shard_path)

    print(f"worker_start={format_worker_name(args.worker_index, args.num_workers)}", flush=True)
    print(f"run_dir={run_dir}", flush=True)

    allowed_language_prefixes = parse_language_prefixes(args.allowed_language_prefixes)
    traj_items = load_traj_manifest(
        input_dir=input_dir,
        run_dir=run_dir,
        max_trajs=args.max_trajs,
    )
    assigned_traj_items = [
        (traj_idx, traj_name)
        for traj_idx, traj_name in traj_items
        if traj_idx % args.num_workers == args.worker_index
    ]
    assigned_traj_count = len(assigned_traj_items)

    print(f"input_dir={input_dir}", flush=True)
    print(f"worker={format_worker_name(args.worker_index, args.num_workers)}", flush=True)
    print(f"assigned_trajs={assigned_traj_count}", flush=True)
    print(f"resume_processed_trajs={processed_traj_count}", flush=True)
    print(f"resume_samples={sample_count}", flush=True)

    for traj_offset, (traj_idx, traj_name) in enumerate(assigned_traj_items, start=1):
        if traj_name in completed_traj_names:
            continue

        traj_dir = input_dir / traj_name
        bundle = build_sample_bundle(
            traj_dir=traj_dir,
            image_root=image_root,
            gt_step=args.gt_step,
            max_images=args.max_images,
            allowed_language_prefixes=allowed_language_prefixes,
        )
        if bundle is None:
            continue

        append_jsonl(shard_path, bundle)
        completed_traj_names.add(traj_dir.name)
        processed_traj_count += 1
        sample_count += len(bundle["samples"])

        if args.progress_every > 0 and processed_traj_count % args.progress_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"processed_trajs={processed_traj_count}/{assigned_traj_count} "
                f"samples_written={sample_count} "
                f"samples_per_sec={sample_count / elapsed:.2f} "
                f"last_traj={traj_dir.name}",
                flush=True,
            )

        if args.meta_every > 0 and processed_traj_count % args.meta_every == 0:
            write_json(
                meta_path,
                build_worker_meta(
                    args=args,
                    assigned_traj_count=assigned_traj_count,
                    processed_traj_count=processed_traj_count,
                    sample_count=sample_count,
                    shard_path=shard_path,
                    start_time=start_time,
                ),
            )

    final_meta = build_worker_meta(
        args=args,
        assigned_traj_count=assigned_traj_count,
        processed_traj_count=processed_traj_count,
        sample_count=sample_count,
        shard_path=shard_path,
        start_time=start_time,
    )
    write_json(meta_path, final_meta)
    print(f"worker_done={format_worker_name(args.worker_index, args.num_workers)}", flush=True)
    print(json.dumps(final_meta, ensure_ascii=False), flush=True)


def iter_bundle_stream(path):
    for bundle in iter_valid_jsonl(path):
        traj_name = bundle.get("traj_name")
        traj_idx = bundle.get("traj_idx", parse_traj_idx(traj_name or ""))
        samples = bundle.get("samples") or []
        if traj_name is None or not isinstance(samples, list):
            continue
        yield {
            "traj_name": traj_name,
            "traj_idx": traj_idx,
            "samples": samples,
        }


def bundle_sort_key(bundle, shard_idx):
    return (bundle["traj_idx"], bundle["traj_name"], shard_idx)


def make_final_sample(sample_id, bundle, sample_payload):
    step_idx = int(sample_payload["step_idx"])
    return {
        "id": f"sample_{sample_id:07d}_{bundle['traj_name']}_{step_idx:04d}",
        "conversations": sample_payload["conversations"],
        "image": sample_payload["image"],
        "data_source": sample_payload.get("data_source", "rxr"),
    }


def merge_run(run_dir, output_json_path):
    shard_dir = run_dir / "shards"
    shard_paths = sorted(shard_dir.glob("worker_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    iterators = [iter_bundle_stream(path) for path in shard_paths]
    heap = []
    for shard_idx, iterator in enumerate(iterators):
        first = next(iterator, None)
        if first is None:
            continue
        heapq.heappush(heap, (bundle_sort_key(first, shard_idx), shard_idx, first))

    def merged_samples():
        sample_id = 0
        while heap:
            _, shard_idx, bundle = heapq.heappop(heap)
            samples = bundle["samples"]
            samples.sort(key=lambda item: int(item["step_idx"]))
            for sample_payload in samples:
                yield make_final_sample(sample_id, bundle, sample_payload)
                sample_id += 1

            nxt = next(iterators[shard_idx], None)
            if nxt is not None:
                heapq.heappush(heap, (bundle_sort_key(nxt, shard_idx), shard_idx, nxt))

    sample_count = write_output_json(output_json_path, merged_samples())
    summary = {
        "run_dir": str(run_dir),
        "output_json": str(output_json_path),
        "sample_count": sample_count,
        "shard_count": len(shard_paths),
        "merged_at_epoch": time.time(),
    }
    write_json(run_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def launch_workers(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}_utc"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    dataset_path = Path(args.dataset_path)
    if dataset_path.exists():
        traj_items = build_traj_manifest_from_dataset(
            dataset_path=dataset_path,
            max_trajs=args.max_trajs,
            allowed_language_prefixes=parse_language_prefixes(args.allowed_language_prefixes),
        )
    else:
        traj_dir_list = sorted_traj_dirs(input_dir)
        if args.max_trajs > 0:
            traj_dir_list = traj_dir_list[: args.max_trajs]
        traj_items = [
            {"traj_idx": parse_traj_idx(traj_dir.name), "traj_name": traj_dir.name}
            for traj_dir in traj_dir_list
        ]
    write_json(
        run_dir / "traj_manifest.json",
        {
            "input_dir": str(input_dir),
            "dataset_path": str(dataset_path),
            "allowed_language_prefixes": parse_language_prefixes(args.allowed_language_prefixes),
            "traj_count": len(traj_items),
            "created_at_epoch": time.time(),
            "traj_items": traj_items,
        },
    )

    worker_specs = []
    script_path = Path(__file__).resolve()
    for worker_index in range(args.num_workers):
        log_path = run_dir / f"worker_{worker_index:02d}.log"
        cmd = [
            sys.executable,
            str(script_path),
            "--mode",
            "worker",
            "--run-dir",
            str(run_dir),
            "--input-dir",
            str(args.input_dir),
            "--image-root",
            str(args.image_root),
            "--output-json",
            str(args.output_json),
            "--gt-step",
            str(args.gt_step),
            "--max-images",
            str(args.max_images),
            "--num-workers",
            str(args.num_workers),
            "--worker-index",
            str(worker_index),
            "--progress-every",
            str(args.progress_every),
            "--meta-every",
            str(args.meta_every),
            "--max-trajs",
            str(args.max_trajs),
            "--allowed-language-prefixes",
            str(args.allowed_language_prefixes),
        ]
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
                "pid": process.pid,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )

    launch_payload = {
        "run_dir": str(run_dir),
        "input_dir": str(args.input_dir),
        "image_root": str(args.image_root),
        "dataset_path": str(args.dataset_path),
        "output_json": str(args.output_json),
        "num_workers": args.num_workers,
        "gt_step": args.gt_step,
        "max_images": args.max_images,
        "progress_every": args.progress_every,
        "meta_every": args.meta_every,
        "launch_time_epoch": time.time(),
        "workers": worker_specs,
        "merge_command": [
            sys.executable,
            str(script_path),
            "--mode",
            "merge",
            "--run-dir",
            str(run_dir),
            "--output-json",
            str(args.output_json),
        ],
    }
    write_json(run_dir / "workers.json", launch_payload)
    print(json.dumps(launch_payload, ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="launch",
        choices=["launch", "worker", "merge"],
        help="launch: spawn detached workers; worker: generate one shard; merge: merge shard files.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing traj_xxx folders with step_*_rgb.jpg and gt_acts.json.",
    )
    parser.add_argument(
        "--image-root",
        default=str(TRAIN_DATA_ROOT),
        help="Root used to compute relative image paths in final JSON.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Final merged JSON path.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory used to store per-run logs and shards.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="RxR dataset file used to build a deterministic trajectory manifest.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name for launch mode.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run directory for worker/merge mode.",
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
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker shards to launch.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Worker index for worker mode.",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=-1,
        help="Optional trajectory cap for debugging. -1 means all.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N processed trajectories.",
    )
    parser.add_argument(
        "--meta-every",
        type=int,
        default=200,
        help="Rewrite worker meta every N processed trajectories.",
    )
    parser.add_argument(
        "--allowed-language-prefixes",
        default="en",
        help="Comma-separated allowed language prefixes. Default keeps English only. Use empty string to keep all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must be within [0, num_workers).")

    if args.mode == "launch":
        launch_workers(args)
        return

    if args.run_dir is None:
        raise ValueError("--run-dir is required for worker/merge mode.")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "worker":
        run_worker(args)
        return

    if args.mode == "merge":
        merge_run(run_dir=run_dir, output_json_path=Path(args.output_json))
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
