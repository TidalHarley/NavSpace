from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import (
    append_result,
    build_resume_index,
    create_simulator,
    ensure_size_bgr,
    extract_actions,
    format_summary,
    get_rgb,
    habitat_quaternion_from_wxyz,
    load_dataset_file,
    locked_load_json,
    process_images_as_video,
    resolve_scene_path,
    resolve_task_dataset_path,
    sanitize_name,
    summarize_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NavSpace evaluation with a local SNav/LLaVA-style model.")
    parser.add_argument("--model-id", type=int, default=0, help="Shard id.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--task", default="environment_state")
    parser.add_argument("--trajectory-path", help="Optional explicit dataset path.")
    parser.add_argument("--hm3d-base-path", required=True)
    parser.add_argument("--model-path", required=True, help="Path to the local SNav/LLaVA checkpoint.")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--output-dir", default="outputs/snav")
    parser.add_argument("--resume-from")
    parser.add_argument("--frame-width", type=int, default=224)
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--max-frames-num", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--future-steps-prompt", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend. Use 'flash_attention_2' only if flash-attn is installed.",
    )
    return parser


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def llava_inference(
    model,
    tokenizer,
    conv_template: str,
    video,
    frame_time: str,
    video_time: float,
    instruction: str,
    future_steps: int,
    device: str,
) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    import copy
    import torch

    prompt = f"""You are navigating in an indoor environment given the instruction: {instruction};
You are given the observation history of previous steps you have taken.
You should:
1) evaluate the history to decide which step of instruction you are at.
2) Predict actions for the next {future_steps} steps to follow up the given instruction until you reach the goal.
Notice that:
1) You can only choose from the following four actions: Move forward, Turn left, Turn right, Stop;
2) Move forward means to move 0.25 meters straight ahead, and turning left or right is a 30-degree turn.
3) If you believe you have reached the target or caught in obstacles, you should choose the stop action.
Output only:
Final Answer: Move forward, Turn left, Turn right, Stop
"""
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled "
        f"from it. These frames are located at {frame_time}."
    )
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type="cuda"):
        outputs = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    trajectory_path = (
        Path(args.trajectory_path).resolve()
        if args.trajectory_path
        else resolve_task_dataset_path(repo_root, args.task).resolve()
    )
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"snav_{sanitize_name(args.task)}.json"
    log_path = output_dir / f"snav_{sanitize_name(args.task)}.log"
    configure_logging(log_path)

    import torch
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        torch_dtype="bfloat16",
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        overwrite_config={"text_config": None, "vision_config": None},
    )
    model.eval()

    episodes = load_dataset_file(trajectory_path)
    resume_source = Path(args.resume_from).resolve() if args.resume_from else result_path
    resume_index = build_resume_index(locked_load_json(resume_source)) if resume_source.exists() else {}

    for traj_idx, episode in enumerate(episodes):
        if traj_idx % args.num_shards != args.model_id:
            continue

        instruction = episode["instruction"]["instruction_text"]
        shortest_path = float(episode.get("info", {}).get("geodesic_distance", 0.0))
        if (instruction, shortest_path) in resume_index:
            continue

        scene_path = resolve_scene_path(episode["scene_id"], Path(args.hm3d_base_path).resolve())
        if not scene_path:
            logging.warning("Skipping episode %s because scene is missing.", episode["episode_id"])
            continue

        simulator, agent = create_simulator(
            scene_path=scene_path,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            include_depth=False,
        )
        try:
            start_position = np.array(episode["start_position"], dtype=np.float32)
            if simulator.pathfinder.is_loaded and not simulator.pathfinder.is_navigable(start_position):
                start_position = simulator.pathfinder.snap_point(start_position)
            initial_state = simulator.get_agent(0).get_state()
            initial_state.position = start_position
            if len(episode["start_rotation"]) == 4:
                initial_state.rotation = habitat_quaternion_from_wxyz(episode["start_rotation"])
            agent.set_state(initial_state)

            goal = episode["goals"][0]
            goal_position = np.array(goal["position"], dtype=np.float32)
            goal_radius = float(goal.get("radius", args.success_distance))
            observation = simulator.get_sensor_observations()
            rgb_history = [
                ensure_size_bgr(get_rgb(observation), args.frame_width, args.frame_height)
            ]
            actual_path_length = 0.0
            oracle_success = 0
            success = 0
            step_idx = 0

            while step_idx < args.max_steps:
                video_frames, frame_time, video_time = process_images_as_video(
                    rgb_history, original_fps=1.0, max_frames_num=args.max_frames_num
                )
                video = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
                video = [video.to(args.device).half()]
                output_text = llava_inference(
                    model=model,
                    tokenizer=tokenizer,
                    conv_template=args.conv_template,
                    video=video,
                    frame_time=frame_time,
                    video_time=video_time,
                    instruction=instruction,
                    future_steps=args.future_steps_prompt,
                    device=args.device,
                )
                actions = extract_actions(output_text, max_actions=4) or ["stop"]
                stop_now = False
                for action in actions:
                    current_distance = np.linalg.norm(agent.get_state().position - goal_position)
                    if current_distance < goal_radius:
                        oracle_success = 1
                    if action == "stop" or step_idx >= args.max_steps:
                        stop_now = True
                        break
                    agent.act(action)
                    step_idx += 1
                    if action == "move_forward":
                        actual_path_length += 0.25
                    observation = simulator.get_sensor_observations()
                    rgb_history.append(
                        ensure_size_bgr(get_rgb(observation), args.frame_width, args.frame_height)
                    )
                if stop_now:
                    break

            final_distance = np.linalg.norm(agent.get_state().position - goal_position)
            if final_distance < goal_radius:
                success = 1

            results = append_result(
                result_path,
                instruction,
                {
                    "success": success,
                    "nav_error": float(final_distance),
                    "os": oracle_success,
                    "shortest_path_length": shortest_path,
                    "actual_path_length": actual_path_length,
                },
            )
            logging.info(format_summary(summarize_results(results)))
        finally:
            simulator.close()

    logging.info("Finished SNav evaluation for task=%s", args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
