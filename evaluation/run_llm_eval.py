from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import load_api_key, resolve_task_dataset_path, sanitize_name
from evaluation.config import LlmEvalConfig, PROFILE_DEFAULTS, TASK_DATASET_MAP
from evaluation.simulation import run_llm_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NavSpace LLM evaluation.")
    parser.add_argument("--profile", default="gemini-pro", choices=sorted(PROFILE_DEFAULTS.keys()))
    parser.add_argument("--provider", help="Override provider type.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--base-url", help="Override OpenAI-compatible base URL.")
    parser.add_argument("--api-key", help="API key passed directly.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key.")
    parser.add_argument("--api-key-file", help="JSON file containing the API key.")
    parser.add_argument("--api-key-field", default="api_key", help="JSON field name in --api-key-file.")
    parser.add_argument("--task", default="environment_state", choices=sorted(TASK_DATASET_MAP.keys()))
    parser.add_argument("--trajectory-path", help="Explicit dataset path. Overrides --task.")
    parser.add_argument("--hm3d-base-path", required=True, help="Path to hm3d_v0.2 scene directory.")
    parser.add_argument("--output-dir", default="outputs/llm", help="Directory for logs and results.")
    parser.add_argument("--result-name", help="Result JSON file name.")
    parser.add_argument("--log-name", help="Log file name.")
    parser.add_argument("--resume-from", help="Optional previous result JSON for resume/reuse.")
    parser.add_argument("--model-id", type=int, default=0, help="Shard id, kept for compatibility with old scripts.")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of trajectory shards.")
    parser.add_argument("--frame-width", type=int, help="RGB sensor width override.")
    parser.add_argument("--frame-height", type=int, help="RGB sensor height override.")
    parser.add_argument("--encode-resize", type=int, help="Resize each frame before base64 encoding.")
    parser.add_argument("--max-frames-num", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--actions-per-inference", type=int, default=4)
    parser.add_argument("--future-steps-prompt", type=int, default=6)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--initial-retry-delay", type=float)
    parser.add_argument("--system-prompt", default="you are a helpful assistant")
    parser.add_argument("--disable-system-message", action="store_true")
    parser.add_argument("--disable-deviation-guard", action="store_true")
    parser.add_argument("--deviation-patience", type=int, default=12)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/dataset/key resolution without launching Habitat-Sim.",
    )
    parser.add_argument(
        "--dry-run-probe-api",
        action="store_true",
        help="During --dry-run, also send a minimal text-only request to verify the API key.",
    )
    return parser


def _resolve_value(args: argparse.Namespace, field_name: str) -> Any:
    cli_value = getattr(args, field_name)
    if cli_value is not None:
        return cli_value
    return PROFILE_DEFAULTS[args.profile].get(field_name)


def build_config(args: argparse.Namespace, repo_root: Path) -> LlmEvalConfig:
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = (
        Path(args.trajectory_path).resolve()
        if args.trajectory_path
        else resolve_task_dataset_path(repo_root, args.task).resolve()
    )
    result_name = args.result_name or f"llm_eval_{sanitize_name(args.profile)}_{sanitize_name(args.task)}.json"
    log_name = args.log_name or f"llm_eval_{sanitize_name(args.profile)}_{sanitize_name(args.task)}.log"
    api_key_file = Path(args.api_key_file).resolve() if args.api_key_file else None
    api_key = load_api_key(
        direct_key=args.api_key,
        env_name=args.api_key_env or _resolve_value(args, "key_env"),
        key_file=api_key_file,
        key_field=args.api_key_field,
    )

    return LlmEvalConfig(
        profile=args.profile,
        provider=_resolve_value(args, "provider"),
        model=_resolve_value(args, "model"),
        base_url=_resolve_value(args, "base_url"),
        api_key=api_key,
        output_dir=output_dir,
        result_path=output_dir / result_name,
        log_path=output_dir / log_name,
        task=args.task,
        trajectory_path=trajectory_path,
        hm3d_base_path=Path(args.hm3d_base_path).resolve(),
        shard_id=args.model_id,
        num_shards=args.num_shards,
        frame_width=_resolve_value(args, "frame_width") or 224,
        frame_height=_resolve_value(args, "frame_height") or 224,
        encode_resize=_resolve_value(args, "encode_resize"),
        max_frames_num=args.max_frames_num,
        target_fps=args.target_fps,
        max_steps=args.max_steps,
        success_distance=args.success_distance,
        actions_per_inference=args.actions_per_inference,
        future_steps_prompt=args.future_steps_prompt,
        max_retries=_resolve_value(args, "max_retries") or 4,
        initial_retry_delay=_resolve_value(args, "initial_retry_delay") or 2.0,
        use_system_message=not args.disable_system_message and bool(_resolve_value(args, "use_system_message")),
        system_prompt=args.system_prompt,
        resume_from=Path(args.resume_from).resolve() if args.resume_from else None,
        deviation_guard=not args.disable_deviation_guard,
        deviation_patience=args.deviation_patience,
    )


def configure_logging(log_path: Path) -> None:
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _dry_run_report(config: LlmEvalConfig, probe_api: bool) -> None:
    from evaluation.common import load_dataset_file, resolve_scene_path

    logging.info("Dry-run: profile=%s provider=%s model=%s", config.profile, config.provider, config.model)
    logging.info("Dry-run: base_url=%s", config.base_url)
    logging.info("Dry-run: trajectory_path=%s", config.trajectory_path)
    logging.info("Dry-run: hm3d_base_path=%s (exists=%s)", config.hm3d_base_path, config.hm3d_base_path.exists())
    logging.info("Dry-run: output_dir=%s", config.output_dir)
    assert config.api_key, "API key resolution returned an empty string."
    logging.info("Dry-run: API key resolved (length=%d, tail=***%s)", len(config.api_key), config.api_key[-4:])

    episodes = load_dataset_file(config.trajectory_path)
    logging.info("Dry-run: loaded %d episodes from dataset.", len(episodes))
    sample = episodes[: min(3, len(episodes))]
    for episode in sample:
        scene_path = resolve_scene_path(episode["scene_id"], config.hm3d_base_path)
        status = str(scene_path) if scene_path else "NOT FOUND"
        logging.info("Dry-run: episode %s -> %s", episode.get("episode_id"), status)

    if probe_api:
        logging.info("Dry-run: probing API with a tiny text-only request ...")
        if config.provider == "openai_compatible":
            from openai import OpenAI

            client = OpenAI(api_key=config.api_key, base_url=config.base_url)
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            logging.info("Dry-run: API probe succeeded (choices=%d)", len(response.choices))
        elif config.provider == "zhipu":
            from zai import ZhipuAiClient

            client = ZhipuAiClient(api_key=config.api_key)
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": "ping"}],
            )
            logging.info("Dry-run: Zhipu probe succeeded (choices=%d)", len(response.choices))
        else:
            logging.warning("Dry-run: unknown provider %s, skipping probe.", config.provider)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = build_config(args, repo_root)
    configure_logging(config.log_path)
    logging.info("Running NavSpace LLM evaluation with profile=%s model=%s", config.profile, config.model)
    if args.dry_run:
        _dry_run_report(config, probe_api=args.dry_run_probe_api)
        logging.info("Dry-run finished without launching Habitat.")
        return 0
    summary = run_llm_evaluation(config)
    logging.info(
        "Finished evaluation. SR=%.3f NE=%.3f OS=%.3f SPL=%.3f",
        summary["sr"],
        summary["ne"],
        summary["os"],
        summary["spl"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
