from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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
    summarize_results,
)
from evaluation.config import LlmEvalConfig
from evaluation.providers import ProviderRequest, infer_actions


LOGGER = logging.getLogger(__name__)


def _load_resume_index(config: LlmEvalConfig) -> dict[tuple[str, float], dict]:
    resume_source = config.resume_from or config.result_path
    if resume_source.exists():
        try:
            return build_resume_index(locked_load_json(resume_source))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load resume file %s: %s", resume_source, exc)
    return {}


def run_llm_evaluation(config: LlmEvalConfig) -> dict[str, float]:
    episodes = load_dataset_file(config.trajectory_path)
    LOGGER.info("Loaded %s episodes from %s", len(episodes), config.trajectory_path)

    provider_request = ProviderRequest(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=config.max_retries,
        initial_retry_delay=config.initial_retry_delay,
        use_system_message=config.use_system_message,
        system_prompt=config.system_prompt,
        encode_resize=config.encode_resize,
    )
    resume_index = _load_resume_index(config)

    for traj_idx, episode in enumerate(episodes):
        if traj_idx % config.num_shards != config.shard_id:
            continue

        instruction = episode["instruction"]["instruction_text"]
        shortest_path = float(episode.get("info", {}).get("geodesic_distance", 0.0))
        episode_key = (instruction, shortest_path)
        if episode_key in resume_index:
            LOGGER.info("Skipping already processed episode %s", episode.get("episode_id"))
            continue

        scene_path = resolve_scene_path(episode["scene_id"], config.hm3d_base_path)
        if not scene_path:
            LOGGER.warning("Skipping episode %s due to missing scene.", episode.get("episode_id"))
            continue

        try:
            simulator, agent = create_simulator(
                scene_path=scene_path,
                frame_width=config.frame_width,
                frame_height=config.frame_height,
                include_depth=False,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to initialize simulator for episode %s: %s", episode.get("episode_id"), exc)
            continue

        try:
            start_position = np.array(episode["start_position"], dtype=np.float32)
            start_rotation = episode["start_rotation"]
            if simulator.pathfinder.is_loaded and not simulator.pathfinder.is_navigable(start_position):
                start_position = simulator.pathfinder.snap_point(start_position)

            goal = episode["goals"][0]
            goal_position = np.array(goal["position"], dtype=np.float32)
            goal_radius = float(goal.get("radius", config.success_distance))

            initial_state = simulator.get_agent(0).get_state()
            initial_state.position = start_position
            if len(start_rotation) == 4:
                initial_state.rotation = habitat_quaternion_from_wxyz(start_rotation)
            agent.set_state(initial_state)

            observation = simulator.get_sensor_observations()
            initial_frame = ensure_size_bgr(get_rgb(observation), config.frame_width, config.frame_height)
            rgb_history = [initial_frame]
            success = 0
            oracle_success = 0
            actual_path_length = 0.0
            step_idx = 0
            distance_increase_streak = 0
            previous_distance = np.linalg.norm(agent.get_state().position - goal_position)

            while step_idx < config.max_steps:
                sampled_frames, frame_time, video_time = process_images_as_video(
                    rgb_history,
                    original_fps=1.0,
                    max_frames_num=config.max_frames_num,
                    target_fps=config.target_fps,
                )
                model_output = infer_actions(
                    provider_request,
                    sampled_frames,
                    instruction,
                    frame_time,
                    video_time,
                    future_steps=config.future_steps_prompt,
                )
                actions = extract_actions(model_output, max_actions=config.actions_per_inference)
                if not actions:
                    LOGGER.warning("No actions parsed for episode %s, forcing stop.", episode.get("episode_id"))
                    actions = ["stop"]

                should_stop = False
                for action in actions:
                    current_distance = np.linalg.norm(agent.get_state().position - goal_position)
                    if current_distance < goal_radius:
                        oracle_success = 1

                    if step_idx >= config.max_steps or action == "stop":
                        should_stop = True
                        break

                    agent.act(action)
                    step_idx += 1
                    if action == "move_forward":
                        actual_path_length += 0.25
                    observation = simulator.get_sensor_observations()
                    frame_bgr = ensure_size_bgr(get_rgb(observation), config.frame_width, config.frame_height)
                    rgb_history.append(frame_bgr)

                    current_distance = np.linalg.norm(agent.get_state().position - goal_position)
                    if current_distance < goal_radius:
                        oracle_success = 1

                    if config.deviation_guard:
                        if current_distance > previous_distance:
                            distance_increase_streak += 1
                        else:
                            distance_increase_streak = 0
                        previous_distance = current_distance
                        if distance_increase_streak >= config.deviation_patience:
                            LOGGER.info(
                                "Stopping episode %s due to repeated distance increase.",
                                episode.get("episode_id"),
                            )
                            should_stop = True
                            break

                if should_stop:
                    break

            final_distance = np.linalg.norm(agent.get_state().position - goal_position)
            if final_distance < goal_radius:
                success = 1

            result_payload = {
                "success": success,
                "nav_error": float(final_distance),
                "os": oracle_success,
                "shortest_path_length": shortest_path,
                "actual_path_length": actual_path_length,
            }
            current_results = append_result(config.result_path, instruction, result_payload)
            summary = summarize_results(current_results)
            LOGGER.info(format_summary(summary))
        finally:
            simulator.close()

    final_results = locked_load_json(config.result_path) if config.result_path.exists() else []
    return summarize_results(final_results)
