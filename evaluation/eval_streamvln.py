from __future__ import annotations

import argparse
import copy
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import (
    append_result,
    build_resume_index,
    create_simulator,
    ensure_size_bgr,
    get_depth,
    get_rgb,
    habitat_quaternion_from_wxyz,
    load_dataset_file,
    locked_load_json,
    resolve_scene_path,
    resolve_task_dataset_path,
    sanitize_name,
    summarize_results,
    format_summary,
)


ACTIONS_MAPPING = {
    "STOP": "stop",
    "↑": "move_forward",
    "←": "turn_left",
    "→": "turn_right",
}

CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NavSpace evaluation with StreamVLN.")
    parser.add_argument("--model-id", type=int, default=0, help="Shard id.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--task", default="environment_state")
    parser.add_argument("--trajectory-path", help="Optional explicit dataset path.")
    parser.add_argument("--hm3d-base-path", required=True)
    parser.add_argument("--model-path", required=True, help="Path to the StreamVLN model directory.")
    parser.add_argument("--output-dir", default="outputs/streamvln")
    parser.add_argument("--resume-from")
    parser.add_argument("--frame-width", type=int, default=224)
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend for StreamVLN (use 'flash_attention_2' only if flash-attn is installed).",
    )
    return parser


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


def preprocess_intrinsic(intrinsic: np.ndarray, orig_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    intrinsic = copy.deepcopy(intrinsic)
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    intrinsic[:, 0] /= orig_size[0] / target_size[0]
    intrinsic[:, 1] /= orig_size[1] / target_size[1]
    intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
    if intrinsic.shape[0] == 1:
        intrinsic = intrinsic.squeeze(0)
    return intrinsic


def parse_model_output(output: str) -> list[str]:
    actions = []
    for char in output:
        if char in ACTIONS_MAPPING:
            actions.append(ACTIONS_MAPPING[char])
    return actions or ["stop"]


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
    result_path = output_dir / f"streamvln_{sanitize_name(args.task)}.json"
    log_path = output_dir / f"streamvln_{sanitize_name(args.task)}.log"
    configure_logging(log_path)

    import torch
    import transformers
    from transformers.image_utils import to_numpy_array

    from model.stream_video_vln import StreamVLNForCausalLM
    from utils.utils import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_MEMORY_TOKEN,
        DEFAULT_VIDEO_TOKEN,
        IMAGE_TOKEN_INDEX,
        MEMORY_TOKEN_INDEX,
        dict_to_cuda,
    )
    from depth_camera_filtering import filter_depth

    def preprocess_depth_image(depth_image, do_depth_scale: bool = True, depth_scale: int = 1000):
        resized_depth_image = depth_image.resize((args.frame_width, args.frame_height), Image.NEAREST)
        depth_array = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            depth_array = depth_array / depth_scale
        return depth_array

    def preprocess_qwen(
        sources,
        tokenizer,
        has_image: bool = False,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False,
    ):
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
            "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        tokenizer.chat_template = chat_template
        input_ids = []
        for source in sources:
            prompt = random.choice(CONJUNCTIONS) + DEFAULT_IMAGE_TOKEN
            source = copy.deepcopy(source)
            source[0]["value"] = (source[0]["value"] + f" {prompt}.").strip()
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
            for conv in source:
                role = roles.get(conv.get("from", conv.get("role")), conv.get("from", conv.get("role")))
                content = conv.get("value", conv.get("content"))
                input_id += tokenizer.apply_chat_template([{"role": role, "content": content}])
            for idx, token_id in enumerate(input_id):
                if token_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if token_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            input_ids.append(input_id)
        return torch.tensor(input_ids, dtype=torch.long)

    def get_axis_align_matrix():
        return torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=4096, padding_side="right"
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        config=config,
    )
    model.reset(200)
    model.eval()
    model.to(args.device)
    image_processor = model.get_vision_tower().image_processor

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
            include_depth=True,
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
            oracle_success = 0
            success = 0
            step_idx = 0
            actual_path_length = 0.0
            time_ids = []
            rgb_list = []
            depth_list = []
            pose_list = []
            intrinsic_list = []
            action_seq = []
            output_ids = None
            past_key_values = None

            while step_idx < args.max_steps:
                observation = simulator.get_sensor_observations()
                rgb_frame = ensure_size_bgr(get_rgb(observation), args.frame_width, args.frame_height)
                rgb_image = Image.fromarray(rgb_frame).convert("RGB")
                depth = get_depth(observation)
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = (depth * 10.0) * 1000
                current_state = simulator.get_agent(0).get_state()
                rotation = current_state.rotation
                yaw = quat_to_yaw(rotation.x, rotation.y, rotation.z, rotation.w)
                position = np.array(current_state.position)
                position[1] += 1.5

                fov = float(agent.agent_config.sensor_specifications[0].hfov)
                fx = (args.frame_width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
                fy = fx
                cx = (args.frame_width - 1.0) / 2.0
                cy = (args.frame_height - 1.0) / 2.0
                intrinsic_matrix = np.array(
                    [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                )
                intrinsic = preprocess_intrinsic(
                    intrinsic_matrix, [args.frame_width, args.frame_height], [args.frame_width, args.frame_height]
                )
                intrinsic_tensor = torch.from_numpy(intrinsic).float()

                rgb_tensor = image_processor.preprocess(images=rgb_image, return_tensors="pt")["pixel_values"][0]
                depth_tensor = torch.from_numpy(
                    preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode="I;16"))
                ).float()
                transform = np.array(
                    [
                        [np.cos(yaw), -np.sin(yaw), 0, position[0]],
                        [np.sin(yaw), np.cos(yaw), 0, position[1]],
                        [0, 0, 1, position[2]],
                        [0, 0, 0, 1],
                    ]
                )
                pose_tensor = torch.from_numpy(transform).double() @ get_axis_align_matrix()

                rgb_list.append(rgb_tensor)
                depth_list.append(depth_tensor)
                pose_list.append(pose_tensor)
                intrinsic_list.append(intrinsic_tensor)
                time_ids.append(step_idx)

                if not action_seq:
                    if output_ids is None:
                        prompt = (
                            "<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. "
                            "Devise an action sequence to follow the instruction using the four actions: "
                            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
                        )
                        conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
                        sources = copy.deepcopy(conversation)
                        sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)
                        add_system = True
                    else:
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        add_system = False

                    input_ids = preprocess_qwen([sources], tokenizer, True, add_system=add_system)
                    if output_ids is not None:
                        input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)

                    input_dict = {
                        "images": torch.stack(rgb_list[-1:]).unsqueeze(0),
                        "depths": torch.stack(depth_list[-1:]).unsqueeze(0),
                        "poses": torch.stack(pose_list[-1:]).unsqueeze(0),
                        "intrinsics": torch.stack(intrinsic_list[-1:]).unsqueeze(0),
                        "inputs": input_ids,
                        "env_id": traj_idx,
                        "time_ids": [time_ids],
                        "task_type": [0],
                    }
                    input_dict = dict_to_cuda(input_dict, args.device)
                    for key in ("images", "depths", "poses", "intrinsics"):
                        input_dict[key] = input_dict[key].to(torch.bfloat16)

                    outputs = model.generate(
                        **input_dict,
                        do_sample=True,
                        num_beams=1,
                        max_new_tokens=10000,
                        use_cache=True,
                        return_dict_in_generate=True,
                        past_key_values=past_key_values,
                        temperature=1.5,
                    )
                    output_ids = outputs.sequences
                    past_key_values = outputs.past_key_values
                    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                    action_seq = parse_model_output(decoded)

                action = action_seq.pop(0)
                if action == "stop":
                    model.reset_for_env(traj_idx)
                    break
                agent.act(action)
                step_idx += 1
                if action == "move_forward":
                    actual_path_length += 0.25
                current_distance = np.linalg.norm(agent.get_state().position - goal_position)
                if current_distance < goal_radius:
                    oracle_success = 1
                if step_idx % 32 == 0:
                    model.reset_for_env(traj_idx)
                    output_ids = None
                    past_key_values = None
                    time_ids = []

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

    logging.info("Finished StreamVLN evaluation for task=%s", args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
