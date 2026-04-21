from __future__ import annotations


DEFAULT_SYSTEM_PROMPT = "you are a helpful assistant"


def build_navigation_prompt(
    instruction: str,
    frame_time: str,
    video_time: float,
    num_frames: int,
    future_steps: int = 6,
) -> str:
    multi_step_prompt = f"""You are navigating in an indoor environment given the instruction: {instruction};
You are given the observation history of previous steps you have taken.
You should:
1) Evaluate the history to decide which step of the instruction you are at.
2) Predict actions for the next {future_steps} steps to follow up the given instruction until you reach the goal.
Notice that:
1) You can only choose from the following four actions: Move forward, Turn left, Turn right, Stop.
2) Move forward means to move 0.25 meters straight ahead, and turning left or right is a 30-degree turn.
3) If you believe you have reached the target or got stuck in obstacles, you should choose the stop action.
----
Starting below, your output should strictly follow this format, without any other information:
Final Answer: Your predicted actions for the next {future_steps} steps.
Please remember that you do not have to provide your analysis.
You can only choose from Move forward, Turn left, Turn right, Stop.
If you are unable to predict the next step, choose Stop.
Example:
Final Answer: Move forward, Turn left, Move forward, Turn right, Turn right, Stop
"""
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled "
        f"from it. These frames are located at {frame_time}. Please answer the following question "
        "related to this video."
    )
    return f"{time_instruction}\n{multi_step_prompt}"
