#!/usr/bin/env python3
import argparse
import heapq
import json
import subprocess
import sys
import time
from pathlib import Path

from rgbs2video import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_JSON,
    TRAIN_DATA_ROOT,
    build_prompt,
    multi_step_gt,
    select_history_images,
    sorted_step_images,
    sorted_traj_dirs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = TRAIN_DATA_ROOT / "r2r_stepwise_runs"


def parse_traj_idx(name: str) -> int:
    parts = name.split("_", 2)
    if len(parts) < 2:
        return 10**9
    try:
        return int(parts[1])
    except ValueError:
        return 10**9


def write_json(path: Path, payload) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def write_output_json(output_json_path: Path, payload_iter) -> int:
    tmp_path = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for item in payload_iter:
            if not first:
                f.write(",\n")
            json.dump(item, f, ensure_ascii=False)
            first = False
            count += 1
        f.write("\n]\n")
    tmp_path.replace(output_json_path)
    return count


def iter_valid_jsonl(path: Path):
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
                print(f"skip_invalid_jsonl path={path} line={line_no}", flush=True)


def append_jsonl(path: Path, payload) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def build_sample_bundle(traj_dir: Path, image_root: Path, gt_step: int, max_images: int):
    gt_act_path = traj_dir / "gt_acts.json"
    if not gt_act_path.exists():
        return None

    with open(gt_act_path, "r", encoding="utf-8") as file:
        gt_acts_dict = json.load(file)

    instruction = gt_acts_dict["instruction"]
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

        history_rel = [str(path.relative_to(image_root)) for path in history_images]
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
                "data_source": "r2r",
            }
        )

    if not samples:
        return None

    return {
        "traj_name": traj_dir.name,
        "traj_idx": parse_traj_idx(traj_dir.name),
        "samples": samples,
    }


def iter_bundle_stream(path: Path):
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


def bundle_sort_key(bundle, shard_idx: int):
    return (bundle["traj_idx"], bundle["traj_name"], shard_idx)


def make_final_sample(sample_id: int, bundle, sample_payload):
    step_idx = int(sample_payload["step_idx"])
    return {
        "id": f"sample_{sample_id:07d}_{bundle['traj_name']}_{step_idx:04d}",
        "conversations": sample_payload["conversations"],
        "image": sample_payload["image"],
        "data_source": sample_payload.get("data_source", "r2r"),
    }


def build_worker_meta(args, assigned_traj_count: int, processed_traj_count: int, sample_count: int, shard_path: Path, start_time: float):
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


def run_worker(args):
    start_time = time.time()
    input_dir = Path(args.input_dir)
    image_root = Path(args.image_root)
    run_dir = Path(args.run_dir)
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_path = shard_dir / f"worker_{args.worker_index:02d}.jsonl"
    meta_path = shard_dir / f"worker_{args.worker_index:02d}.meta.json"

    traj_dir_list = sorted_traj_dirs(input_dir)
    if args.max_trajs > 0:
        traj_dir_list = traj_dir_list[: args.max_trajs]

    assigned_traj_dirs = [
        traj_dir
        for traj_dir in traj_dir_list
        if parse_traj_idx(traj_dir.name) % args.num_workers == args.worker_index
    ]

    completed_trajs = set()
    sample_count = 0
    if shard_path.exists():
        for bundle in iter_valid_jsonl(shard_path):
            traj_name = bundle.get("traj_name")
            samples = bundle.get("samples") or []
            if traj_name is None or not isinstance(samples, list):
                continue
            completed_trajs.add(traj_name)
            sample_count += len(samples)

    processed_traj_count = len(completed_trajs)
    for traj_dir in assigned_traj_dirs:
        if traj_dir.name in completed_trajs:
            continue

        bundle = build_sample_bundle(
            traj_dir=traj_dir,
            image_root=image_root,
            gt_step=args.gt_step,
            max_images=args.max_images,
        )
        if bundle is None:
            continue

        append_jsonl(shard_path, bundle)
        processed_traj_count += 1
        sample_count += len(bundle["samples"])

        if args.progress_every > 0 and processed_traj_count % args.progress_every == 0:
            print(
                f"worker={args.worker_index + 1}/{args.num_workers} "
                f"processed_trajs={processed_traj_count}/{len(assigned_traj_dirs)} "
                f"samples={sample_count} last_traj={traj_dir.name}",
                flush=True,
            )

        if args.meta_every > 0 and processed_traj_count % args.meta_every == 0:
            write_json(
                meta_path,
                build_worker_meta(
                    args=args,
                    assigned_traj_count=len(assigned_traj_dirs),
                    processed_traj_count=processed_traj_count,
                    sample_count=sample_count,
                    shard_path=shard_path,
                    start_time=start_time,
                ),
            )

    write_json(
        meta_path,
        build_worker_meta(
            args=args,
            assigned_traj_count=len(assigned_traj_dirs),
            processed_traj_count=processed_traj_count,
            sample_count=sample_count,
            shard_path=shard_path,
            start_time=start_time,
        ),
    )


def merge_run(run_dir: Path, output_json_path: Path, max_samples: int):
    shard_dir = run_dir / "shards"
    shard_paths = sorted(shard_dir.glob("worker_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    iterators = [iter_bundle_stream(path) for path in shard_paths]
    heap = []
    for shard_idx, iterator in enumerate(iterators):
        first = next(iterator, None)
        if first is not None:
            heapq.heappush(heap, (bundle_sort_key(first, shard_idx), shard_idx, first))

    def merged_samples():
        sample_id = 0
        while heap:
            _, shard_idx, bundle = heapq.heappop(heap)
            samples = bundle["samples"]
            samples.sort(key=lambda item: int(item["step_idx"]))
            for sample_payload in samples:
                if max_samples > 0 and sample_id >= max_samples:
                    return
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
        "max_samples": max_samples,
    }
    write_json(run_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def launch_workers(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}_utc"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    traj_dir_list = sorted_traj_dirs(Path(args.input_dir))
    if args.max_trajs > 0:
        traj_dir_list = traj_dir_list[: args.max_trajs]

    write_json(
        run_dir / "traj_manifest.json",
        {
            "input_dir": str(args.input_dir),
            "traj_count": len(traj_dir_list),
            "created_at_epoch": time.time(),
            "traj_names": [traj_dir.name for traj_dir in traj_dir_list],
        },
    )

    worker_specs = []
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
        "output_json": str(args.output_json),
        "num_workers": args.num_workers,
        "gt_step": args.gt_step,
        "max_images": args.max_images,
        "progress_every": args.progress_every,
        "meta_every": args.meta_every,
        "max_trajs": args.max_trajs,
        "max_samples": args.max_samples,
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
            "--max-samples",
            str(args.max_samples),
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
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--gt-step", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--max-trajs", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--meta-every", type=int, default=200)
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
        merge_run(run_dir=run_dir, output_json_path=Path(args.output_json), max_samples=args.max_samples)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
