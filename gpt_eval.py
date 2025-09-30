from openai import OpenAI
import json
import time
import random
import os
import base64
import logging
import sys
import cv2
import io
from PIL import Image
sys.path.append(os.path.dirname(__file__))
from text_prompt import get_task_prompt
from PIL import Image
import requests
import copy
import os
import sys
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from filelock import FileLock
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector, quat_from_coeffs
from habitat.utils.visualizations import maps
import habitat
import logging
from config_utils1 import *
import numpy as np
import base64
from io import BytesIO
import re
import time
import cv2
import json
import argparse
import textwrap
import gzip
import math
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def encode_image(image):
    target_size=(224,224)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)   
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    byte_data = buffer.getvalue() 
    return base64.b64encode(byte_data).decode("utf-8")

def _load_api_key():
    current_dir = os.path.dirname(__file__)
    key_file_path = os.path.join(current_dir, 'gptv_key.json')
    try:
        with open(key_file_path, 'r') as f:
            keys = json.load(f)
            api_key = keys.get('api_key')
            if not api_key:
                logger.error(f"Error: 'api_key' field not found in '{key_file_path}'.")
            return api_key
    except FileNotFoundError:
        logger.error(f"Error: Key file '{key_file_path}' not found. Please ensure it exists in '{current_dir}' directory.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: Unable to decode JSON from '{key_file_path}'. Please check if the file format is correct.")
        return None
    except Exception as e:
        logger.error(f"Unknown error occurred while loading API key: {e}")
        return None

def infer(images,instr,video_time,frame_time) -> str:
    multi_step_prompt = f""" You are navigating in an indoor environment given the instruction: {instr};
    You are given the observation history of previous steps you have taken;
    You should:
    1) evaluate the history to decide which step of instruction you are at.
    2) Predict actions for the next 6 steps to follow up the given instruction until you reach the goal;
    Notice that:
    1) You can only choose from the following four actions: Move forward, Turn left, Turn right, Stop;
    2) Move forward means to move 0.25 meters straight ahead, and turning left or right is a 30-degree turn.
    3) If you believe you have reached the target or caught in obstacles, you should choose the stop action.
    ----
    Starting below, your output should strictly follow this format, without any other information:
    Final Answer: Your predicted actions for the next 6 steps.
    Please remember that you DON'T have to provide your analysis,you just have to provide your final answer,which are six steps.
    Please remember that you can ONLY choose four kinds of actions:Move forward,Turn left,Turn right,Stop.You can replace other actions as stop."turn around" is NOT allowed to apper in your answer.
    Please remember that if you are unable to predict next step,just select stop as your answer.(In this case,you don't have to tell me you are not certain about your answer.Just provide six actions as your answer)All in all,you should strictly follow the format.
    Here is an example for your output , you should STRICTLY FOLLOW its FORMAT to give out the actions you predict :
    Final Answer: Move forward, Move forward, Move forward, Move forward, Move forward, Move forward
    """
        
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(images)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = f"\n{time_instruciton}\n{multi_step_prompt}"
    content_parts=[]
    for i in images:
        image_base=encode_image(i)
        
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base}"
            }
        })

    api_key = _load_api_key()
    if not api_key:
        return "Error: API key not found or unable to load."
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")

    prompt_text = question
    content_parts.append({"type": "text", "text": prompt_text})
    messages = [
        {"role": "system", "content": "you are a helpful assiatnce"},
        {"role": "user", "content": content_parts}
    ]
    
    max_retries=4
    retry_delay=2
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gemini-2.5-pro",
                messages=messages,
            )

            msg = resp.choices[0].message
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content or "(No text output)"
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"API call failed, retry {attempt + 1}: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"API call failed, maximum retry attempts reached: {e}")
                return f"Error: API call failed - {e}"

    if isinstance(content, list):
        parts = []
        for p in content:
            t = p.get("text") if isinstance(p, dict) else None
            if t:
                parts.append(t)
        return "\n".join(parts) if parts else "(No available text output)"

    return "(Unable to parse model output)"

def process_images_as_video(images, original_fps, max_frames_num, target_fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 384, 384, 3)), "0.00s", 0.0
    
    total_frames = len(images)
    video_duration = total_frames / original_fps
    
    sampling_interval = max(1, round(original_fps / target_fps))
    frame_indices = list(range(0, total_frames, sampling_interval))
    time_stamps = [idx/original_fps for idx in frame_indices]
    
    if len(frame_indices) > max_frames_num or force_sample:
        uniform_samples = np.linspace(0, total_frames-1, max_frames_num, dtype=int)
        frame_indices = uniform_samples.tolist()
        time_stamps = [idx/original_fps for idx in frame_indices]
    
    time_str = ",".join(f"{t:.2f}s" for t in time_stamps)
    sampled_frames = np.stack([images[i] for i in frame_indices])
    return sampled_frames, time_str, video_duration



def put_text_with_autowrap(img, text, position, font_scale=1, color=(0,0,0), thickness=2):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    lines = textwrap.wrap(text, width=int(img.shape[1] / (text_width / len(text))))
    y = position[1]
    for line in lines:
        textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        gap = textsize[1] + 10
        y += gap
        cv2.putText(img, line, (position[0], y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return img

def locked_load_json(json_path):
    lock_path = json_path + ".lock"
    with FileLock(lock_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data

def locked_dump_json(data, json_path):
    lock_path = json_path + ".lock"
    with FileLock(lock_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

def get_scene_path_from_scene_id(scene_id):
    if os.path.exists(scene_id):
        logging.info(f"Scene ID is already a complete path: {scene_id}")
        return scene_id
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(HM3D_BASE_PATH, split)
        if os.path.exists(split_dir):
            for scene_dir in os.listdir(split_dir):
                if scene_dir.startswith(scene_id):
                    scene_file = os.path.join(split_dir, scene_dir, f"{scene_dir}.basis.glb")
                    if os.path.exists(scene_file):
                        return scene_file
    
    logging.error(f"Unable to find scene file corresponding to scene ID {scene_id}")
    return None

def get_sim_config(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True
    sim_cfg.requires_textures = True
    sim_cfg.gpu_device_id = -1
    return sim_cfg

def get_agent_config():
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [640, 720]
    rgb_sensor.position = [0.0, 1.5, 0.0]
    rgb_sensor.orientation = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "move_backward": habitat_sim.agent.ActionSpec("move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "look_up": habitat_sim.agent.ActionSpec("look_up", habitat_sim.agent.ActuationSpec(amount=30.0)), 
        "look_down": habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=30.0)),
    }
    return agent_cfg

def load_traj_points_from_json(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        return json.load(f)

def load_episode_actions(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["episode_id"]: item["action"] for item in data}


def setup_qwen_output_dirs(base_dir, traj_idx):
    qwen_base = os.path.join(base_dir, "Qwen_simpleaction", "data", "image", str(traj_idx + 1))
    os.makedirs(qwen_base, exist_ok=True)
    return qwen_base


def get_rgb(obs):
    rgb = obs.get("rgb")
    if rgb is None:
        rgb = obs.get("color_sensor")
    if rgb is None:
        raise KeyError("RGB image not found in observations. Please ensure RGB sensor is enabled in configuration.")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

def ensure_size_bgr(frame_rgb, width, height):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    if (bgr.shape[1], bgr.shape[0]) != (width, height):
        logging.debug(f"Resizing frame from {bgr.shape[1]}x{bgr.shape[0]} to {width}x{height}")
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return bgr

def get_instr_from_scene_id(scene_id):
    if os.path.exists(scene_id):
        logger.info(f"Scene ID is already a complete path: {scene_id}")
        return scene_id
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(HM3D_BASE_PATH, split)
        if os.path.exists(split_dir):
            for scene_dir in os.listdir(split_dir):
                if scene_dir.startswith(scene_id):
                    scene_file = os.path.join(split_dir, scene_dir, f"{scene_dir}.basis.glb")
                    if os.path.exists(scene_file):
                        return scene_file
    
    logger.error(f"Unable to find scene file corresponding to scene ID {scene_id}")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, required=True, help='Model ID')
    
    args = parser.parse_args()   
    
    idt=args.model_id
    HM3D_BASE_PATH = "/LingXi/habitat-data-0.2.5/scenes/hm3d_v0.2"
    ACTION_MAP = {
    "forward": "move_forward",
    "backward": "move_backward", 
    "left": "turn_left",
    "right": "turn_right",
    "look_up": "look_up",
    "look_down": "look_down",
    "stop": "stop"
    }
    RESET_INTERVAL = 0
    FRAME_WIDTH = 224
    FRAME_HEIGHT = 224
    FPS = 3.0
    SAVE_EVERY = 8 
    OUTPUT_SUBDIR = "navigation_video"
    EPISODE_ACTIONS_FILE = "episode_actions.json"
    success_distance=3.0

    current_task="looping"
    ckpt_chosen = f"yzh_gemini-2.5-pro_{current_task}"      # this is the output file name: you can change to whatever you want

    

    log_file = os.path.join('/LingXi/yhl/LLM', f"ckpt-{ckpt_chosen}.log")       # change the route 
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')


    points_filename = f"/LingXi/LLaVA-NeXT/vln_eval/traj_{current_task}_vln.json.gz"        # this is the NavSpace Dataset
    
    try:
        traj_points_data = load_traj_points_from_json(points_filename)
        traj_points = traj_points_data['episodes']
        logging.info(f"Loaded {len(traj_points)} episodes from trajectory points file.")
    except FileNotFoundError:
        logging.error(f"Trajectory file not found: {points_filename}. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading trajectory points from {points_filename}: {e}")
        sys.exit(1)



    for traj_idx, ep in enumerate(traj_points):
        if traj_idx % 8!=int(idt):
            continue
        file_path = os.path.join('/LingXi/yhl/LLM',  f"ckpt-{ckpt_chosen}.json")
        video_writer = None
        current_episode_id = ep['episode_id']
        scene_id = ep['scene_id']
        instr=ep['instruction']['instruction_text']
        
        scene_path = get_scene_path_from_scene_id(scene_id)
        if not scene_path:
            print(f"Unable to find scene file corresponding to scene ID {scene_id}. Skipping this trajectory.")
            continue
        
        try:
            sim_cfg = get_sim_config(scene_path)
            agent_cfg = get_agent_config()
            config = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            sim = habitat_sim.Simulator(config)
            agent = sim.initialize_agent(0)
           
        except Exception as e:
            print(f"Failed to initialize Habitat-Sim environment: {e}")
            continue

        start_desired = np.array(ep['start_position'], dtype=np.float32)
        start_rotation = ep['start_rotation']
        
        if sim.pathfinder.is_loaded:
            nearest_pos = sim.pathfinder.snap_point(start_desired)
            dist = np.linalg.norm(np.array(nearest_pos) - np.array(start_desired))
            if not sim.pathfinder.is_navigable(start_desired):
                start_desired = nearest_pos
        else:
            print("Pathfinder not loaded, using original starting position.")
        
        goal_info = ep['goals'][0]
        goal_pos = np.array(goal_info['position'], dtype=np.float32)
        goal_radius = float(goal_info.get('radius', success_distance))
        
        if 'info' in ep and 'geodesic_distance' in ep['info']:
            shortest_path_length = float(ep['info']['geodesic_distance'])
        else:
            logging.warning(f"No geodesic_distance information found in Episode {current_episode_id}, SPL cannot be calculated correctly.")
            shortest_path_length = 0.0
        
        initial_state = habitat_sim.AgentState()
        initial_state.position = start_desired
        if len(start_rotation) == 4:
            initial_state.rotation = np.quaternion(start_rotation[0], start_rotation[1], start_rotation[2], start_rotation[3])
        else:
            initial_state.rotation = np.quaternion(1, 0, 0, 0)
        
        agent.set_state(initial_state)
        start = start_desired.copy()

        success=0
        recorded_actions = []
        recorded_positions = []
        step_idx = 0
        max_steps = 150
        rgb_input_traj = []
        actual_path_length=0.0
        num_of_break=0
        initial_obs = sim.get_sensor_observations()
        print(list(initial_obs.keys()))
        initial_rgb_frame = get_rgb(initial_obs)
        initial_frame_bgr = ensure_size_bgr(initial_rgb_frame, FRAME_WIDTH, FRAME_HEIGHT)
        rgb_input_traj.append(initial_frame_bgr)
        flag=False
        Os=0
        num_of_diviation=0
        initial_agent_state = sim.get_agent(0).get_state()
        initial_pos = initial_agent_state.position
        initial_dist_to_goal = np.linalg.norm(initial_pos - goal_pos)
        previous_navigation_error=initial_dist_to_goal
        while True:            
            video, frame_time, video_time = process_images_as_video(rgb_input_traj, original_fps = 1, max_frames_num = 8, target_fps=1, force_sample=False)
            output=infer(video,instr,video_time,frame_time)
            print(output)
            output=output.split(',')
            yzh_stop=False
            for i in range(4):               
                match = re.search(r'\b(forward|turn left|turn right|stop)\b', output[i], re.IGNORECASE)
                if match:
                    action = match.group(0).lower() 
                    if action == 'forward':
                        act = "move_forward"
                    elif action == 'turn left':
                        act = 'turn_left'
                    elif action == 'turn right':
                        act = 'turn_right'
                    elif action == 'stop':
                        act = 'stop'
                    else:
                        logging.warning('no action in output!')
                else:
                    yzh_stop=True
                final_agent_state = sim.get_agent(0).get_state()
                final_pos = final_agent_state.position
                dist_to_goal = np.linalg.norm(final_pos - goal_pos)
                if dist_to_goal>=previous_navigation_error:
                    num_of_diviation+=1
                else:
                    num_of_diviation=0
                if num_of_diviation>=8:
                    num_of_break+=1
                    act='stop'
                previous_navigation_error=dist_to_goal
                current_message="current act is:"+act+", the step_idx is:"+str(step_idx)+", the distance to goal is:"+str(dist_to_goal)
                logging.info(f"Episode {current_episode_id}: "+current_message)
                if dist_to_goal<goal_radius:
                    Os=1
                step_idx+=1
                if step_idx == max_steps :
                    act = 'stop'
                if act=='stop':
                    flag=True
                    break
                agent.act(act)
                
                if act=='move_forward':
                    actual_path_length+=0.25
                
                if dist_to_goal<goal_radius:
                    Os=1
                obs = sim.get_sensor_observations() 
                rgb_frame = get_rgb(obs)
                frame_bgr = ensure_size_bgr(rgb_frame, FRAME_WIDTH, FRAME_HEIGHT)
                rgb_input_traj.append(frame_bgr)
            if flag:
                break
            if yzh_stop:
                break
        final_agent_state = sim.get_agent(0).get_state()
        final_pos = final_agent_state.position
        dist_to_goal = np.linalg.norm(final_pos - goal_pos)
        if dist_to_goal<goal_radius:
            success=1
        if os.path.exists(file_path):
            file=locked_load_json(file_path)                
        else:
            file = []

        file.append({
            instr: {
                'success': success,
                'nav_error': float(dist_to_goal),
                'os': Os,
                'shortest_path_length': shortest_path_length,  # l_i
                'actual_path_length': actual_path_length  # p_i
            }
        })

        locked_dump_json(file,file_path)

        
        success_rates, dtg_results, os_results, spl_results = [], [], [], []
        for item in file:
            result_dict = list(item.values())[0] 
            if (math.isfinite(result_dict['success']) and
                math.isfinite(result_dict['nav_error']) and
                math.isfinite(result_dict['os']) and
                math.isfinite(result_dict['shortest_path_length']) and
                math.isfinite(result_dict['actual_path_length'])):

                s_i = result_dict['success']
                l_i = result_dict['shortest_path_length']
                p_i = result_dict['actual_path_length']

                success_rates.append(s_i)
                dtg_results.append(result_dict['nav_error'])
                os_results.append(result_dict['os'])

                if s_i == 1:
                    spl_contribution = s_i * (l_i / max(p_i, l_i))
                else:
                    spl_contribution = 0.0
                spl_results.append(spl_contribution)
        avg_sr = np.mean(success_rates) if success_rates else 0
        avg_dtg = np.mean(dtg_results) if dtg_results else 0
        avg_os = np.mean(os_results) if os_results else 0
        avg_spl = np.mean(spl_results) if spl_results else 0

        num_eval = len(success_rates)
        eval_result = 'Success:%.3f, Nav Error:%.2f, OS:%.3f, SPL:%.3f' % (avg_sr, avg_dtg, avg_os, avg_spl)
        logging.info(f"[{num_eval}] Eval Results: " + eval_result)
        print(f"Episode {current_episode_id}: Distance: {dist_to_goal:.3f}m, Actual path: {actual_path_length:.2f}m, Shortest path: {shortest_path_length:.2f}m")

        sim.close()

    print("All trajectories processed.")
    logging.info("Number of excessive deviations: "+str(num_of_break))
