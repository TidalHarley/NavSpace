#!/usr/bin/env python3
"""NavSpace SNav evaluation (H200 trainaligned gold standard).
"""
from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import (  # noqa: E402
    append_result,
    build_resume_index,
    create_simulator,
    ensure_size_rgb,
    format_summary,
    get_rgb,
    habitat_quaternion_from_wxyz,
    load_dataset_file,
    locked_load_json,
    resolve_scene_path,
    resolve_task_dataset_path,
    sanitize_name,
    summarize_results,
)

SYSTEM_PROMPT = (
    "You are an autonomous indoor navigation agent. "
    "You observe the environment through sequential RGB frames and follow "
    "natural language instructions to reach a goal location. "
    "At each decision step, predict the next {n} low-level actions from: "
    "FORWARD (↑) moves 25 cm, TURN LEFT (←) rotates 30°, "
    "TURN RIGHT (→) rotates 30°, STOP ends navigation."
)

EN_ACTIONS = {
    "move forward": "move_forward",
    "forward": "move_forward",
    "turn left": "turn_left",
    "left": "turn_left",
    "turn right": "turn_right",
    "right": "turn_right",
    "stop": "stop",
}
ARROW_ACTIONS = {"↑": "move_forward", "←": "turn_left", "→": "turn_right"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NavSpace benchmark with an SNav / LLaVA-Video checkpoint "
        "(H200 trainaligned protocol)."
    )
    parser.add_argument("--model-id", type=int, default=0, help="Shard id.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--task", default="environment_state")
    parser.add_argument("--trajectory-path", help="Optional explicit dataset path.")
    parser.add_argument("--hm3d-base-path", required=True)
    parser.add_argument("--model-path", required=True, help="Path to the local SNav checkpoint.")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--output-dir", default="outputs/snav")
    parser.add_argument("--resume-from")
    parser.add_argument("--frame-width", type=int, default=384)
    parser.add_argument("--frame-height", type=int, default=384)
    parser.add_argument("--max-frames-num", type=int, default=16, help="History frames (open-source default 16).")
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--future-steps-prompt", type=int, default=6)
    parser.add_argument("--actions-per-inference", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map-gpu", type=int, default=0)
    parser.add_argument("--sim-gpu-id", type=int, default=None, help="Physical GPU id for Habitat-Sim rendering.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=-1, help="Per-shard cap for smoke tests. -1 means no cap.")
    parser.add_argument("--log-outputs", action="store_true", help="Log raw model outputs for debugging.")
    parser.add_argument("--min-stop-step", type=int, default=0, help="If >0, ignore STOP before this many executed steps.")
    parser.add_argument(
        "--early-stop-replacement",
        default="turn_right",
        choices=["turn_right", "move_forward", "none"],
        help="Action to use if early STOP filtering leaves no actions.",
    )
    parser.add_argument(
        "--vision-tower-path",
        default="",
        help="Local SigLIP dir to override mm_vision_tower (offline / portable ckpts).",
    )
    return parser


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def sample_history_images(images: list[np.ndarray], max_images_num: int) -> tuple[list[np.ndarray], list[int], int]:
    total_frames = len(images)
    if total_frames == 0:
        return [], [], 0
    if max_images_num <= 0 or total_frames <= max_images_num:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_images_num, dtype=int).tolist()
    return [images[i] for i in indices], indices, total_frames


def build_history_time_instruction(frame_indices: list[int], total_frames: int, history_fps: float = 1.0) -> str | None:
    if total_frames <= 0 or not frame_indices:
        return None
    frame_time = ",".join([f"{idx / history_fps:.2f}s" for idx in frame_indices])
    video_time = total_frames / history_fps
    return (
        f"The video lasts for {video_time:.2f} seconds, and {len(frame_indices)} frames are uniformly sampled "
        f"from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    )


def build_prompt_body(instruction: str, future_steps: int) -> str:
    return (
        f"{SYSTEM_PROMPT.format(n=future_steps)}\n\n"
        f"These frames show your navigation history. Instruction: {instruction}\n"
        f"Predict the next {future_steps} actions."
    )


def process_eval_image(pil_image: Image.Image, image_processor, model_config, overwrite_image_aspect_ratio: str | None = None):
    from llava.mm_utils import expand2square, process_anyres_image, process_highres_image

    image_size = pil_image.size
    image_aspect_ratio = getattr(model_config, "image_aspect_ratio", None)
    if overwrite_image_aspect_ratio is not None:
        image_aspect_ratio = overwrite_image_aspect_ratio
    if image_aspect_ratio == "highres":
        image = process_highres_image(pil_image, image_processor, model_config.image_grid_pinpoints)
    elif image_aspect_ratio == "anyres" or (image_aspect_ratio and "anyres_max" in image_aspect_ratio):
        try:
            image = process_anyres_image(pil_image, image_processor, model_config.image_grid_pinpoints)
        except ValueError as exc:
            logging.warning("anyres preprocess failed, fallback to pad: %s", exc)
            image = expand2square(pil_image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    elif image_aspect_ratio == "pad":
        image = expand2square(pil_image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        image = image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][0]
    return image, image_size


def build_eval_inputs(history_frames, image_processor, model_config, max_history_images, future_steps, instruction, device, dtype):
    from llava.constants import DEFAULT_IMAGE_TOKEN

    sampled_frames, sampled_indices, total_frames = sample_history_images(history_frames, max_history_images)
    if not sampled_frames:
        raise ValueError("No history images are available for evaluation.")
    pil_images = [Image.fromarray(frame) for frame in sampled_frames]
    image_tensors = []
    image_sizes = []
    overwrite = "pad" if len(pil_images) > 1 else None
    for pil_image in pil_images:
        image_tensor, image_size = process_eval_image(
            pil_image,
            image_processor=image_processor,
            model_config=model_config,
            overwrite_image_aspect_ratio=overwrite,
        )
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensors.append(image_tensor.to(device=device, dtype=dtype))
        image_sizes.append(image_size)
    image_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * len(pil_images)
    time_instruction = build_history_time_instruction(sampled_indices, total_frames)
    prompt_body = build_prompt_body(instruction, future_steps)
    question = f"{image_tokens}{time_instruction}\n{prompt_body}" if time_instruction else f"{image_tokens}{prompt_body}"
    return image_tensors, image_sizes, question


def parse_actions_trainaligned(text: str, max_actions: int) -> list[str]:
    tail = text.split("Final Answer:")[-1] if "Final Answer:" in text else text
    actions: list[str] = []
    phrase_re = re.compile(
        r"\b(move forward|turn left|turn right|forward|left|right|stop)\b",
        flags=re.IGNORECASE,
    )
    i = 0
    while i < len(tail) and len(actions) < max_actions:
        if tail[i : i + 4].upper() == "STOP":
            actions.append("stop")
            i += 4
            continue
        ch = tail[i]
        if ch in ARROW_ACTIONS:
            actions.append(ARROW_ACTIONS[ch])
            i += 1
            continue
        match = phrase_re.match(tail, i)
        if match:
            actions.append(EN_ACTIONS[match.group(0).lower()])
            i = match.end()
            continue
        i += 1
    return actions[:max_actions]


def snav_inference(
    model,
    tokenizer,
    image_processor,
    conv_template: str,
    history_frames,
    instruction: str,
    future_steps: int,
    max_history_images: int,
    max_new_tokens: int,
    device: str,
    dtype,
) -> str:
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    image_tensors, image_sizes, question = build_eval_inputs(
        history_frames=history_frames,
        image_processor=image_processor,
        model_config=model.config,
        max_history_images=max_history_images,
        future_steps=future_steps,
        instruction=instruction,
        device=device,
        dtype=dtype,
    )
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
    # LLaVA generate may return either full sequences (prompt + new tokens)
    # or only newly generated tokens depending on the model wrapper.
    generated_ids = (
        output_ids[:, input_ids.shape[1] :]
        if output_ids.shape[1] > input_ids.shape[1]
        else output_ids
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


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
    task_tag = sanitize_name(args.task)
    result_path = output_dir / f"snav_{task_tag}.json"
    log_suffix = f".shard{args.model_id}of{args.num_shards}" if args.num_shards > 1 else ""
    log_path = output_dir / f"snav_{task_tag}{log_suffix}.log"
    configure_logging(log_path)

    import torch
    from llava.model.builder import load_pretrained_model

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device_map = {"": args.device_map_gpu}
    logging.info(
        "SNav NavSpace eval (trainaligned) | task=%s shard=%d/%d model=%s max_frames=%d future=%d exec=%d",
        args.task,
        args.model_id,
        args.num_shards,
        args.model_path,
        args.max_frames_num,
        args.future_steps_prompt,
        args.actions_per_inference,
    )

    overwrite_config: dict = {}
    if args.vision_tower_path:
        overwrite_config["mm_vision_tower"] = args.vision_tower_path
        from llava.model.multimodal_encoder import builder as _vt_builder
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model import llava_arch as _llava_arch

        _orig_build = _vt_builder.build_vision_tower

        def _build_vision_tower_siglip_first(vision_tower_cfg, **kwargs):
            name = getattr(
                vision_tower_cfg,
                "mm_vision_tower",
                getattr(vision_tower_cfg, "vision_tower", None),
            )
            if name and "siglip" in str(name).lower():
                return SigLipVisionTower(name, vision_tower_cfg=vision_tower_cfg, **kwargs)
            return _orig_build(vision_tower_cfg, **kwargs)

        _vt_builder.build_vision_tower = _build_vision_tower_siglip_first
        _llava_arch.build_vision_tower = _build_vision_tower_siglip_first

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        overwrite_config=overwrite_config or None,
    )
    model.eval()

    episodes = load_dataset_file(trajectory_path)
    resume_source = Path(args.resume_from).resolve() if args.resume_from else result_path
    resume_index = build_resume_index(locked_load_json(resume_source)) if resume_source.exists() else {}
    assigned_seen = 0

    for traj_idx, episode in enumerate(episodes):
        if traj_idx % args.num_shards != args.model_id:
            continue
        if args.max_episodes >= 0 and assigned_seen >= args.max_episodes:
            break
        assigned_seen += 1

        instruction = episode["instruction"]["instruction_text"]
        shortest_path = float(episode.get("info", {}).get("geodesic_distance", 0.0))
        if (instruction, shortest_path) in resume_index:
            continue

        scene_path = resolve_scene_path(episode["scene_id"], Path(args.hm3d_base_path).resolve())
        if not scene_path:
            logging.warning("Skipping episode %s because scene is missing.", episode.get("episode_id", traj_idx))
            continue

        simulator, agent = create_simulator(
            scene_path=scene_path,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            include_depth=False,
            gpu_device_id=args.sim_gpu_id,
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
            rgb_history = [ensure_size_rgb(get_rgb(observation), args.frame_width, args.frame_height)]
            actual_path_length = 0.0
            oracle_success = 0
            success = 0
            step_idx = 0

            while step_idx < args.max_steps:
                output_text = snav_inference(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    conv_template=args.conv_template,
                    history_frames=rgb_history,
                    instruction=instruction,
                    future_steps=args.future_steps_prompt,
                    max_history_images=args.max_frames_num,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    dtype=dtype,
                )
                actions = parse_actions_trainaligned(output_text, max_actions=args.actions_per_inference) or ["stop"]
                if args.min_stop_step > 0 and step_idx < args.min_stop_step and "stop" in actions:
                    filtered_actions = [action for action in actions if action != "stop"]
                    if not filtered_actions and args.early_stop_replacement != "none":
                        filtered_actions = [args.early_stop_replacement]
                    actions = filtered_actions or actions
                if args.log_outputs:
                    logging.info(
                        "episode=%s step=%d output=%r actions=%s",
                        episode.get("episode_id", traj_idx),
                        step_idx,
                        output_text[-500:],
                        actions,
                    )
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
                    current_distance = np.linalg.norm(agent.get_state().position - goal_position)
                    if current_distance < goal_radius:
                        oracle_success = 1
                    observation = simulator.get_sensor_observations()
                    rgb_history.append(ensure_size_rgb(get_rgb(observation), args.frame_width, args.frame_height))
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
            logging.info(
                "episode=%s final_distance=%.3f success=%d os=%d path=%.2f | %s",
                episode.get("episode_id", traj_idx),
                final_distance,
                success,
                oracle_success,
                actual_path_length,
                format_summary(summarize_results(results)),
            )
        finally:
            simulator.close()

    logging.info(
        "Finished SNav NavSpace evaluation for task=%s shard=%d/%d",
        args.task,
        args.model_id,
        args.num_shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
