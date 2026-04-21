from flask import Flask, render_template_string, request # Import request to get sid
from flask_socketio import SocketIO, emit, send
import habitat_sim
import numpy as np
import base64
import cv2
import os
import glob
import random
import logging
import traceback
from pathlib import Path
import sys
import time
import queue # Import queue for inter-thread communication
import threading # Import threading for the worker thread
import json # New: for JSON file handling

# Make the repository root importable so that sibling packages (annotation_pipeline.llm_client,
# tools.*) can be used regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The LLM analysis button is optional. If no LLM client / API key is provided, the UI still
# works; the button just returns a friendly message instead of crashing the worker.
try:
    from annotation_pipeline import llm_client  # type: ignore
except Exception as _llm_import_err:  # pragma: no cover - defensive import
    llm_client = None  # type: ignore[assignment]
    _LLM_IMPORT_ERROR = _llm_import_err
else:
    _LLM_IMPORT_ERROR = None

# newest version of server!

# 详细日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('habitat_debug.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("🚀 启动 HM3D Habitat 调试版本")
logger.info("=" * 80)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'habitat_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)
#==========================================================================================================================
TRAJECTORY_LOG_FILE = os.environ.get('NAVSPACE_TRAJECTORY_LOG', 'trajectories.json')  # 定义轨迹日志文件名
MIN_FAMILIARIZATION_STEPS = int(os.environ.get('NAVSPACE_MIN_FAMILIARIZATION_STEPS', '200'))  # 熟悉场景所需的最少步数
DEFAULT_HM3D_BASE_PATH = "/LingXi/habitat-data-0.2.5/scenes/hm3d_v0.2"  # 历史路径，保留作后备

# 无可用场景时在日志与前端中提示（保持简短，细节见 docs/annotation.md）
_SCENE_HELP = (
    "未找到任何 .glb 场景。请设置其一："
    "NAVSPACE_SCENE_PATHS=/绝对路径/scene.glb[,更多...]；"
    "或 NAVSPACE_MP3D_ROOT=/path/to/mp3d（需含子目录下的 .glb）；"
    "或 NAVSPACE_HM3D_BASE_PATH=/path/to/hm3d_v0.2（含 train/val/test/*/…/*.basis.glb）。"
)


class SceneLoader:
    # 负责加载Habitat-Sim的虚拟环境场景文件
    # 如果未能找到任何有效的场景，会尝试加载备用的默认场景
    def __init__(self):
        # 1) 显式场景列表（最高优先级）：逗号分隔的 .glb / .basis.glb 绝对路径
        explicit = os.environ.get("NAVSPACE_SCENE_PATHS", "").strip()
        if explicit:
            paths: list = []
            for raw in explicit.split(","):
                p = raw.strip()
                if not p:
                    continue
                ap = os.path.abspath(os.path.expanduser(p))
                if os.path.isfile(ap) and ap.lower().endswith(".glb"):
                    paths.append(ap)
                else:
                    logger.warning(f"NAVSPACE_SCENE_PATHS 跳过（非文件或后缀不对）: {p}")
            if paths:
                self.hm3d_base_path = None
                self.scene_paths = paths
                logger.info(f"📁 使用 NAVSPACE_SCENE_PATHS 指定的 {len(paths)} 个场景文件。")
                return

        # MP3D（与 VLN-CE 相同布局：mp3d/<scene_id>/<scene_id>.glb）优先于 HM3D
        mp3d_root = os.environ.get("NAVSPACE_MP3D_ROOT", "").strip()
        if mp3d_root:
            logger.info(f"📁 使用 MP3D 场景根目录 (NAVSPACE_MP3D_ROOT): {mp3d_root}")
            self.hm3d_base_path = None
            self.scene_paths = self._load_mp3d_scenes(mp3d_root)
            if not self.scene_paths:
                logger.warning(
                    "⚠️ MP3D 目录下未发现 .glb，将回退到 HM3D 扫描（请检查路径或是否已解压场景）。"
                )
                self.hm3d_base_path = os.environ.get('NAVSPACE_HM3D_BASE_PATH', DEFAULT_HM3D_BASE_PATH)
                logger.debug(f"📁 HM3D 基础路径配置为: {self.hm3d_base_path}")
                self.scene_paths = self._load_hm3d_scenes()
        else:
            # HM3D 根目录，优先读环境变量 NAVSPACE_HM3D_BASE_PATH
            self.hm3d_base_path = os.environ.get('NAVSPACE_HM3D_BASE_PATH', DEFAULT_HM3D_BASE_PATH)
            logger.debug(f"📁 HM3D 基础路径配置为: {self.hm3d_base_path}")
            self.scene_paths = self._load_hm3d_scenes()
        logger.info(f"🎬 成功加载 {len(self.scene_paths)} 个场景文件")
        if not self.scene_paths:
            logger.error("❌ 未找到任何有效的场景文件，模拟器无法启动。")
            logger.error(_SCENE_HELP)

    def _load_mp3d_scenes(self, mp3d_root: str) -> list:
        """Scan Matterport3D meshes in the same layout as VLN-CE / Habitat defaults."""
        logger.debug("🔍 开始搜索 MP3D 场景文件 (.glb) ...")
        if not os.path.isdir(mp3d_root):
            logger.error(f"❌ NAVSPACE_MP3D_ROOT 不是有效目录: {mp3d_root}")
            return []
        # 常见布局：mp3d/<scene_id>/<scene_id>.glb；部分解压方式会多一层目录，故增加递归
        patterns = [
            os.path.join(mp3d_root, "*", "*.glb"),
            os.path.join(mp3d_root, "*", "meshes", "*.glb"),
        ]
        scene_files: list = []
        for pattern in patterns:
            found = sorted(glob.glob(pattern))
            scene_files.extend(found)
        # 递归兜底（目录较深或非标准布局时）
        if not scene_files:
            try:
                deep = glob.glob(os.path.join(mp3d_root, "**", "*.glb"), recursive=True)
                scene_files.extend(sorted(deep))
            except Exception as e:
                logger.error(f"MP3D 递归搜索失败: {e}")
        # 去重并保持顺序
        seen = set()
        unique = []
        for p in scene_files:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        logger.info(f"📊 MP3D: 在 {mp3d_root} 找到 {len(unique)} 个 .glb 场景。")
        return unique

    def _load_hm3d_scenes(self): # 修改方法名为 _load_hm3d_scenes
        logger.debug("🔍 开始搜索 HM3D 场景文件...")
        scene_files = []
        if not os.path.exists(self.hm3d_base_path):
            logger.error(f"❌ 错误: 配置的 HM3D 基础路径不存在: {self.hm3d_base_path}")
            return []
        
        # 遍历 train, val, test 分割目录
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.hm3d_base_path, split)
            if os.path.exists(split_dir):
                try:
                    # HM3D 场景文件通常是 .basis.glb 格式，位于 {split_dir}/{scene_id}/{scene_id}.basis.glb
                    # 这里的 "*", "*.basis.glb" 模式匹配 /scene_id/scene_id.basis.glb
                    pattern = os.path.join(split_dir, "*", "*.basis.glb") 
                    scenes = glob.glob(pattern)
                    scene_files.extend(scenes)
                    logger.debug(f"Found {len(scenes)} scenes in {split_dir}")
                except Exception as e:
                    logger.error(f"❌ 搜索 '{split_dir}' 路径下的场景文件失败: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"Split directory not found: {split_dir}")
        
        if not scene_files:
            logger.warning("⚠️ 未在 HM3D 目录下找到任何 .basis.glb 场景文件。尝试备用默认场景。")
            # 使用 HM3D 的默认场景路径作为备用
            default_hm3d_scene = "/LingXi/habitat-data-0.2.5/scenes/hm3d_v0.2/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb"
            if os.path.exists(default_hm3d_scene):
                scene_files = [default_hm3d_scene]
                logger.info("✅ 成功加载备用默认 HM3D 场景文件。")
            else:
                logger.error(f"❌ 备用默认 HM3D 场景文件也不存在: {default_hm3d_scene}")
                logger.error("请确保 'hm3d_base_path' 或默认场景路径指向正确且可访问的 HM3D 数据集文件。")
                
        logger.info(f"📊 最终找到 {len(scene_files)} 个场景文件。")
        return scene_files

#==========================================================================================================================
class Command:
    INIT_SIM = 1
    EXECUTE_ACTION = 2
    SWITCH_SCENE = 3
    STOP = 4 
    SUBMIT_INSTRUCTION = 5
    START_RECORDING = 6 # 开始记录路径
    STOP_RECORDING = 7  # 停止记录路径
    TRIGGER_LLM_ANALYSIS = 8 # 新增：触发大模型分析

class SimulatorWorker:
    # 处理所有模拟器操作的核心类。它会在后台线程中运行，负责初始化模拟器、执行动作、切换场景等
    def __init__(self, socketio_instance, scene_paths):
        self.socketio = socketio_instance
        self.command_queue = queue.Queue()
        self.sim = None     # This will hold the single Habitat-Sim instance
        self.agent_id = 0
        self.current_scene_index = 0
        self.scene_paths = scene_paths
        
        # 轨迹数据管理
        self.all_trajectories_log = self._load_trajectory_data()
        self.next_episode_id = max([ep['episode_id'] for ep in self.all_trajectories_log] or [0]) + 1
        self.next_trajectory_id = max([ep['trajectory_id'] for ep in self.all_trajectories_log if 'trajectory_id' in ep] or [0]) + 1
        
        # 存储每个客户端当前进行中的episode数据，以及其录制状态
        # {sid: {episode_details, 'recording_active': True/False, 'pending_submission': True/False}}
        self.client_current_episode = {} 

        # 键盘映射
        self.action_mapping = {
            'arrowup': 'move_forward',    # 上方向键
            'arrowdown': 'move_backward', # 下方向键
            'arrowleft': 'turn_left',     # 左方向键
            'arrowright': 'turn_right',   # 右方向键
            'pageup': 'look_up',          # PageUp 向上看
            'pagedown': 'look_down'       # PageDown 向下看
        }
        # Mapping Habitat-Sim actions to desired output action names for action_sequence
        self.action_display_names = {
            'move_forward': 'forward',
            'move_backward': 'backward',
            'turn_left': 'left',
            'turn_right': 'right',
            'look_up': 'look_up',
            'look_down': 'look_down'
        }

        self._running = False
        self._thread = None
        logger.info("SimulatorWorker initialized, ready to start thread.")

    def _load_trajectory_data(self):
        if os.path.exists(TRAJECTORY_LOG_FILE):
            with open(TRAJECTORY_LOG_FILE, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict) and "episodes" in data:
                        return data["episodes"]
                    logger.warning(f"Failed to decode JSON from {TRAJECTORY_LOG_FILE} or invalid format, starting fresh.")
                    return []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON from {TRAJECTORY_LOG_FILE}, starting fresh.")
                    return []
        return []

    def _save_trajectory_data(self):
        """Saves the complete trajectory log to a JSON file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(TRAJECTORY_LOG_FILE) or '.', exist_ok=True)
        with open(TRAJECTORY_LOG_FILE, 'w', encoding='utf-8') as f: # Added encoding for Chinese chars
            json.dump({"episodes": self.all_trajectories_log}, f, indent=4, ensure_ascii=False) 
        logger.info(f"Trajectory data saved to {TRAJECTORY_LOG_FILE}")

    def start(self):
        """Starts the simulator worker in a background thread."""
        if not self._running:
            self._running = True
            # Use socketio.start_background_task for proper Flask-SocketIO thread management
            self._thread = self.socketio.start_background_task(self._run_loop)
            logger.info("Simulator worker background task started.")

    def stop(self):
        """Sends a stop command to the worker and waits for it to finish."""
        if self._running:
            logger.info("Sending STOP command to simulator worker.")
            # Before stopping, finalize any pending episodes for all clients
            for sid in list(self.client_current_episode.keys()):
                self._finalize_current_episode(sid) # Save any unsaved episodes
            self.command_queue.put({'type': Command.STOP})
            self._running = False
            if self.sim:
                self.sim.close()
                self.sim = None
                logger.info("Simulator instance closed by worker stop.")

    def _run_loop(self):
        """Main loop for the simulator worker thread."""
        logger.info("Simulator worker _run_loop started.")
        while self._running:
            try:
                command = self.command_queue.get(timeout=0.05) # Check for commands every 50ms
                cmd_type = command.get('type')
                sid = command.get('sid') # Client SID to emit results back to

                if cmd_type == Command.STOP:
                    logger.info("Simulator worker received STOP command.")
                    self._running = False
                    break # Exit loop

                self._emit_log(sid, f"Worker processing command: {cmd_type}", level='debug')

                if cmd_type == Command.INIT_SIM:
                    self._handle_init_sim(sid, command.get('scene_path'))
                elif cmd_type == Command.EXECUTE_ACTION:
                    self._handle_execute_action(sid, command.get('action_key'))
                elif cmd_type == Command.SWITCH_SCENE:
                    self._handle_switch_scene(sid)
                elif cmd_type == Command.SUBMIT_INSTRUCTION:
                    self._handle_submit_instruction(sid, command.get('instruction_text'))
                elif cmd_type == Command.START_RECORDING: 
                    self._handle_start_recording(sid)
                elif cmd_type == Command.STOP_RECORDING:  
                    self._handle_stop_recording(sid)
                elif cmd_type == Command.TRIGGER_LLM_ANALYSIS: 
                    self._handle_trigger_llm_analysis(sid)
                else:
                    self._emit_log(sid, f"Unknown command type: {cmd_type}", level='error')

            except queue.Empty:
                pass # No command, continue polling
            except Exception as e:
                logger.error(f"Simulator worker encountered an unhandled error: {e}")
                logger.error(traceback.format_exc())
                if sid: # Attempt to emit error back to the client if sid is available
                    self._emit_error(sid, f"Worker internal error: {e}")
        logger.info("Simulator worker _run_loop finished.")

    # Helper methods to emit back to client via SocketIO from the worker thread
    def _emit_log(self, sid, message, level='info'):
        if level == 'info':
            logger.info(message)
        elif level == 'debug':
            logger.debug(message)
        elif level == 'error':
            logger.error(message)
        if sid:
            self.socketio.emit('backend_log', {'message': message}, room=sid)

    def _emit_observation(self, sid, image_data, status, recording_state=None):
        """Emits observation with current status and recording state."""
        if sid:
            payload = {'image': image_data, 'status': status}
            if recording_state:
                payload['recording_state'] = recording_state
            payload.update(self._annotation_progress_payload(sid))
            self.socketio.emit('observation', payload, room=sid)

    def _emit_error(self, sid, message):
        if sid:
            self.socketio.emit('error', {'message': message}, room=sid)

    def _emit_status_update(self, sid, recording_state, status_message):
        """Emits a general status update to the client."""
        if sid:
            payload = {
                'recording_state': recording_state,
                'message': status_message
            }
            payload.update(self._annotation_progress_payload(sid))
            self.socketio.emit('status_update', payload, room=sid)
    
    def _emit_llm_response(self, sid, text_response):
        """Emits LLM analysis text to the client."""
        if sid:
            self.socketio.emit('llm_response', {'text': text_response}, room=sid)

    def _annotation_progress_payload(self, sid):
        familiarization_steps = 0
        if sid in self.client_current_episode:
            familiarization_steps = self.client_current_episode[sid].get('familiarization_steps', 0)
        return {
            'familiarization_steps': familiarization_steps,
            'min_familiarization_steps': MIN_FAMILIARIZATION_STEPS,
            'can_annotate': familiarization_steps >= MIN_FAMILIARIZATION_STEPS,
        }

    def _can_start_annotation(self, sid):
        return self._annotation_progress_payload(sid)['can_annotate']

    # --- LLM API Placeholder ---
    def _call_llm_api(self, image_base64: str) -> str:
        """
        调用本地封装的大模型 (LLM) API。
        接收 Base64 编码的图像字符串作为输入。
        返回一个文本字符串，表示大模型对图像的分析结果，或错误信息。
        如果未安装 LLM 客户端或未配置 API Key，则返回一条提示信息而不抛异常。
        """
        logger.info("开始调用大模型 API 进行图像分析。")
        if llm_client is None:
            msg = (
                "未启用 LLM 辅助功能：annotation_pipeline.llm_client 不可用，"
                f"原因: {_LLM_IMPORT_ERROR}. 请参考 docs/annotation.md 配置 OPENAI_API_KEY。"
            )
            logger.warning(msg)
            return msg
        try:
            llm_response = llm_client.call_llm_api(image_base64)
            if isinstance(llm_response, str) and llm_response.startswith("错误:"):
                logger.error(f"大模型 API 返回错误: {llm_response}")
                return llm_response # 直接返回错误信息给前端

            logger.info("大模型 API 调用成功。")
            return llm_response

        except Exception as e:
            logger.error(f"调用大模型 API 发生异常: {e}")
            logger.error(traceback.format_exc()) # 记录完整的堆栈信息
            return f"调用大模型 API 失败: {e}"

    # 新增：独立处理大模型分析的函数
    def _handle_trigger_llm_analysis(self, sid):
        self._emit_log(sid, "Worker: Received TRIGGER_LLM_ANALYSIS command.", level='info')
        self._emit_llm_response(sid, '大模型正在分析图像...') # Inform frontend that analysis is in progress

        current_observation_b64 = self._get_observation()
        if current_observation_b64:
            try:
                llm_analysis = self._call_llm_api(current_observation_b64)
                self._emit_llm_response(sid, llm_analysis) # Send analysis result to frontend
                self._emit_log(sid, f"Worker: LLM analysis sent for SID {sid}.", level='info')
            except Exception as e:
                error_msg = f"Worker: Error during LLM API call: {e}"
                self._emit_log(sid, error_msg, level='error')
                self._emit_llm_response(sid, f"大模型分析失败: {error_msg}")
        else:
            self._emit_log(sid, "Worker: Failed to get current observation for LLM analysis.", level='warning')
            self._emit_llm_response(sid, '未能获取图片，无法进行大模型分析。')
        
        # Re-send current observation and status to ensure UI is updated
        # No change in recording state, just refresh image and status
        current_rec_state = 'idle'
        if sid in self.client_current_episode:
            if self.client_current_episode[sid].get('recording_active', False):
                current_rec_state = 'recording'
            elif self.client_current_episode[sid].get('pending_submission', False):
                current_rec_state = 'pending_submission'
        
        self._emit_observation(sid, self._get_observation(), f'✅ 大模型分析已完成', current_rec_state)


    # --- Habitat-Sim specific operations, isolated to this worker thread ---

    def _get_sim_config(self, scene_path):
        """Creates Habitat-Sim SimulatorConfiguration."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False
        sim_cfg.create_renderer = True
        sim_cfg.requires_textures = True
        sim_cfg.gpu_device_id = -1  # CPU headless mode, requires OSMesa or similar
        return sim_cfg

    def _get_agent_config(self):
        """Creates Habitat-Sim AgentConfiguration."""
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [640, 720]  # height, width
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
            "look_down": habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=30.0)), # 用户要求向下看，amount改为正值
        }
        return agent_cfg

    def _handle_init_sim(self, sid, scene_path=None):
        """Handles simulator initialization command."""
        self._emit_log(sid, "Worker: Initializing simulator...", level='info')
        
        # Finalize any existing episode for this SID before starting a new one
        self._finalize_current_episode(sid)
        # LLM output should not be reset on init, as it's now explicitly controlled.

        try:
            if scene_path is None:
                if not self.scene_paths:
                    self._emit_error(
                        sid,
                        "Worker: No scene paths available to initialize simulator! "
                        + _SCENE_HELP,
                    )
                    return False
                # 默认使用第一个场景
                scene_path = self.scene_paths[0] 
                self._emit_log(sid, f"Worker: No specific scene requested, using default: {os.path.basename(scene_path)}", level='info')
            
            # Check if scene path exists and is readable
            if not os.path.exists(scene_path):
                self._emit_error(sid, f"Worker: Scene file not found: {scene_path}")
                return False

            sim_cfg = self._get_sim_config(scene_path)
            agent_cfg = self._get_agent_config()
            config = habitat_sim.Configuration(sim_cfg, [agent_cfg])

            # Close existing simulator before creating a new one
            if self.sim is not None:
                self._emit_log(sid, "Worker: Closing existing simulator instance.", level='debug')
                self.sim.close()
                self.sim = None

            self._emit_log(sid, f"Worker: Creating new Habitat simulator instance for scene: {os.path.basename(scene_path)}", level='info')
            start_time = time.time()
            self.sim = habitat_sim.Simulator(config)
            init_duration = time.time() - start_time
            self._emit_log(sid, f"Worker: Simulator instance created! Took {init_duration:.2f} seconds.", level='info')

            agent = self.sim.initialize_agent(self.agent_id)
            self._emit_log(sid, f"Worker: Agent {self.agent_id} initialized.", level='debug')
            
            # Set initial agent position
            if self.sim.pathfinder.is_loaded:
                navigable_point = self.sim.pathfinder.get_random_navigable_point()  # 初始化的点是随机导航点
                initial_state = habitat_sim.AgentState()
                initial_state.position = navigable_point
                initial_state.rotation = np.quaternion(1, 0, 0, 0) # Default rotation (facing forward)
                agent.set_state(initial_state)
                self._emit_log(sid, f"Worker: Agent set to navigable point: {navigable_point}", level='info')
            else:
                initial_state = habitat_sim.AgentState()
                initial_state.position = np.array([0.0, 1.5, 0.0])
                initial_state.rotation = np.quaternion(1, 0, 0, 0)
                agent.set_state(initial_state)
                self._emit_log(sid, f"Worker: Pathfinder not loaded, Agent set to default position: {initial_state.position}", level='warning')
            
            # Initialize current episode data for this SID (resetting state)
            # A new episode only truly starts when "Start Recording" is clicked.
            # This ensures a clean slate for the client.
            
            agent_state_at_init = agent.get_state()
            self.client_current_episode[sid] = {
                "episode_id": None, # Will be set on START_RECORDING
                "trajectory_id": None, # Will be set on START_RECORDING
                "scene_id": scene_path,
                "start_position": None, # Will be set on START_RECORDING
                "start_rotation": None, # Will be set on START_RECORDING
                "info": {"geodesic_distance": 0.0}, 
                "goals": [], 
                "instruction": {"instruction_text": "no instruction yet"}, 
                "reference_path": [], 
                "action_sequence": [], 
                "recording_active": False,
                "pending_submission": False,
                "familiarization_steps": 0,
                # Store the actual initial state for context if needed, but not as 'start_position' for trajectory JSON
                "initial_agent_state_in_scene": {
                    "position": agent_state_at_init.position.tolist(),
                    "rotation": [agent_state_at_init.rotation.w, agent_state_at_init.rotation.x, agent_state_at_init.rotation.y, agent_state_at_init.rotation.z]
                }
            }

            # Get and send initial observation
            img_base64 = self._get_observation()
            if img_base64:
                self._emit_observation(sid, img_base64, f'✅ 场景已加载 (可用场景: {len(self.scene_paths)})', 'idle')
                self._emit_log(sid, "Worker: Initial observation sent.", level='info')
                return True
            else:
                self._emit_error(sid, "Worker: Simulator initialized but failed to get initial image.")
                return False

        except Exception as e:
            logger.critical(f"Worker: Habitat simulator initialization failed: {e}")
            logger.critical(f"完整堆栈跟踪: \n{traceback.format_exc()}")
            self.sim = None
            self._emit_error(sid, f"Worker: Simulator initialization failed. Error: {e}")
            return False

    def _get_observation(self):
        """Gets observation image and encodes to Base64 JPEG."""
        if self.sim is None:
            logger.error("Worker: Simulator not initialized, cannot get observation.")
            return None
            
        try:
            observations = self.sim.get_sensor_observations()
            if "color_sensor" in observations:
                rgb_img = observations["color_sensor"]
                if rgb_img.shape[2] == 4: # Convert RGBA to RGB
                    rgb_img = rgb_img[:, :, :3]
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
                success, buffer = cv2.imencode('.jpg', bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if success:
                    return base64.b64encode(buffer).decode('utf-8')
                else:
                    logger.error("Worker: cv2.imencode failed.")
            else:
                logger.error("Worker: 'color_sensor' not found in observations.")
        except Exception as e:
            logger.error(f"Worker: Error getting observation: {e}")
            logger.error(f"完整堆栈跟踪: \n{traceback.format_exc()}")
        return None

    def _handle_execute_action(self, sid, action_key):
        """Handles action execution command."""
        self._emit_log(sid, f"Worker: Executing action: '{action_key}'", level='debug')
        if self.sim is None:
            self._emit_error(sid, "Worker: Simulator not initialized, cannot execute action.")
            return False

        action_name = self.action_mapping.get(action_key.lower())
        if not action_name:
            self._emit_log(sid, f"Worker: Unknown action key: '{action_key}'", level='warning')
            return False
            
        try:
            agent = self.sim.get_agent(self.agent_id)
            agent.act(action_name)
            self._emit_log(sid, f"Worker: Action '{action_name}' executed.", level='debug')

            if sid in self.client_current_episode:
                current_episode = self.client_current_episode[sid]
                if not current_episode.get('recording_active', False) and not current_episode.get('pending_submission', False):
                    current_episode['familiarization_steps'] = current_episode.get('familiarization_steps', 0) + 1
                    familiarization_steps = current_episode['familiarization_steps']
                    if familiarization_steps == MIN_FAMILIARIZATION_STEPS:
                        self._emit_status_update(
                            sid,
                            'idle',
                            f'✅ 已完成场景熟悉：累计移动 {familiarization_steps} 步，现在可以开始正式记录与提交指令。'
                        )
            
            # Record agent's current position to the reference path ONLY IF recording is active
            if sid in self.client_current_episode and self.client_current_episode[sid].get('recording_active', False):
                current_agent_state = self.sim.get_agent(self.agent_id).get_state()
                self.client_current_episode[sid]["reference_path"].append(current_agent_state.position.tolist())
                # Append the action's display name to the action_sequence
                self.client_current_episode[sid]["action_sequence"].append(self.action_display_names.get(action_name, action_name))
                self._emit_log(sid, f"Worker: Position added to reference_path. Path length: {len(self.client_current_episode[sid]['reference_path'])}. Action '{action_name}' added to sequence.", level='debug')


            img_base64 = self._get_observation()
            if img_base64:
                # Determine the current recording state to send back to frontend
                current_rec_state = 'idle'
                if sid in self.client_current_episode:
                    if self.client_current_episode[sid].get('recording_active', False):
                        current_rec_state = 'recording'
                    elif self.client_current_episode[sid].get('pending_submission', False):
                        current_rec_state = 'pending_submission'

                self._emit_observation(sid, img_base64, f'✅ 执行动作: {action_key.upper()}', current_rec_state)
                return True
            else:
                self._emit_error(sid, f"Worker: Action {action_key.upper()} executed, but failed to get new image.")
                return False
        except Exception as e:
            logger.error(f"Worker: Failed to execute action '{action_name}': {e}")
            logger.error(f"完整堆栈跟踪: \n{traceback.format_exc()}")
            self._emit_error(sid, f"Worker: Failed to execute action {action_key.upper()}. Error: {e}")
            return False

    def _handle_switch_scene(self, sid):
        """Handles scene switch command."""
        self._emit_log(sid, "Worker: Handling scene switch request.", level='info')
        
        # Finalize any ongoing episode before switching scenes
        self._finalize_current_episode(sid)
        # LLM output should not be reset on scene switch, as it's now explicitly controlled.

        if len(self.scene_paths) <= 1:
            self._emit_log(sid, "Worker: Only 1 or 0 scenes available, cannot switch.", level='info')
            self._emit_observation(sid, self._get_observation(), f'ℹ️ 仅有一个场景，无法切换。', 'idle')
            return False

        # Randomly select a new scene, ensuring it's not the current one
        possible_indices = [i for i in range(len(self.scene_paths)) if i != self.current_scene_index]
        if not possible_indices: # Should not happen if len(self.scene_paths) > 1
            self._emit_log(sid, "Worker: No different scene to switch to.", level='warning')
            self._emit_observation(sid, self._get_observation(), f'ℹ️ 无法找到不同场景切换。', 'idle')
            return False

        new_index = random.choice(possible_indices)
        new_scene_path = self.scene_paths[new_index]
        
        self._emit_log(sid, f"Worker: Attempting to switch to new scene (index {new_index}): {os.path.basename(new_scene_path)}", level='info')
        
        # Re-initialize simulator with the new scene
        success = self._handle_init_sim(sid, new_scene_path)
        if success:
            self.current_scene_index = new_index
            self._emit_log(sid, f"Worker: Scene successfully switched to index {self.current_scene_index}.", level='info')
            self._emit_observation(sid, self._get_observation(), f'✅ 场景已切换 (场景 {self.current_scene_index + 1}/{len(self.scene_paths)})', 'idle')
            return True
        else:
            self._emit_error(sid, f"Worker: Failed to switch to scene {os.path.basename(new_scene_path)}.")
            self._emit_status_update(sid, 'idle', f'❌ 切换场景失败！')
            return False

    def _handle_start_recording(self, sid):
        """Handles the command to start recording a new trajectory."""
        self._emit_log(sid, f"Worker: Received START_RECORDING command for SID: {sid}", level='info')

        if sid not in self.client_current_episode or self.client_current_episode[sid].get('pending_submission', False):
            self._emit_error(sid, "Worker: Cannot start recording. Either simulator not initialized, or a previous episode is pending submission.")
            self._emit_status_update(sid, 'idle', '⚠️ 无法开始记录，请先提交当前指令或初始化模拟器。')
            return

        familiarization_steps = self.client_current_episode[sid].get('familiarization_steps', 0)
        if not self._can_start_annotation(sid):
            remaining_steps = max(0, MIN_FAMILIARIZATION_STEPS - familiarization_steps)
            self._emit_error(sid, f"Worker: Annotation is locked until the annotator explores at least {MIN_FAMILIARIZATION_STEPS} steps in the scene.")
            self._emit_status_update(
                sid,
                'idle',
                f'⚠️ 需先熟悉场景：当前 {familiarization_steps}/{MIN_FAMILIARIZATION_STEPS} 步，还需移动 {remaining_steps} 步后才能开始正式记录。'
            )
            return
        
        # Finalize any existing *incomplete* episode that wasn't explicitly stopped/submitted
        # This handles cases where user starts recording, then clicks 'Start Recording' again without stopping
        if self.client_current_episode[sid].get('recording_active', False):
            self._emit_log(sid, f"Worker: Previous recording for SID {sid} was active, auto-finalizing as incomplete.", level='warning')
            self._finalize_current_episode(sid) # Save the old one as incomplete
            # Then re-initialize client_current_episode for a fresh start
            self._handle_init_sim(sid, self.client_current_episode[sid]['scene_id']) # Re-init sim for a clean episode structure
            
            # After re-init_sim, the client_current_episode[sid] is reset, so proceed with new recording
            
        # --- LLM Integration: REMOVED from here, now handled by _handle_trigger_llm_analysis ---

        current_episode = self.client_current_episode[sid]

        # Populate episode details
        current_episode["episode_id"] = self.next_episode_id
        current_episode["trajectory_id"] = self.next_trajectory_id
        self.next_episode_id += 1
        self.next_trajectory_id += 1

        agent_state = self.sim.get_agent(self.agent_id).get_state()
        current_episode["start_position"] = agent_state.position.tolist()
        current_rotation = agent_state.rotation
        current_episode["start_rotation"] = [current_rotation.w, current_rotation.x, current_rotation.y, current_rotation.z]
        current_episode["reference_path"] = [agent_state.position.tolist()] # Start path with current position
        current_episode["action_sequence"] = []
        current_episode["recording_active"] = True
        current_episode["pending_submission"] = False
        current_episode["instruction"]["instruction_text"] = "instruction incomplete" # Reset instruction

        self._emit_log(sid, f"Worker: Recording started for episode {current_episode['episode_id']}.", level='info')
        self._emit_observation(sid, self._get_observation(), '✅ 开始记录路径...', 'recording')
        self._emit_status_update(sid, 'recording', '✅ 正在记录路径...')

    def _handle_stop_recording(self, sid):
        """Handles the command to stop recording a trajectory."""
        self._emit_log(sid, f"Worker: Received STOP_RECORDING command for SID: {sid}", level='info')

        if sid not in self.client_current_episode or not self.client_current_episode[sid].get('recording_active', False):
            self._emit_error(sid, "Worker: Cannot stop recording. No active recording found.")
            self._emit_status_update(sid, 'recording', '⚠️ 未在记录中，无法停止。')
            return

        current_episode = self.client_current_episode[sid]
        current_episode["recording_active"] = False
        current_episode["pending_submission"] = True # Mark as ready for instruction submission

        # Append 'stop' action to action sequence
        current_episode["action_sequence"].append("stop")

        # Current agent position is the goal for this episode
        current_agent_state = self.sim.get_agent(self.agent_id).get_state()
        goal_position = current_agent_state.position
        
        # Ensure 'goals' list exists and add the new goal
        if not current_episode["goals"]:
            current_episode["goals"] = []
        current_episode["goals"].append({
            "position": goal_position.tolist(),
            "radius": 0.2
        })

        self._emit_log(sid, f"Worker: Recording stopped for episode {current_episode['episode_id']}, pending instruction submission.", level='info')
        self._emit_observation(sid, self._get_observation(), '✅ 停止记录路径，请提交指令', 'pending_submission')
        self._emit_status_update(sid, 'pending_submission', '✅ 已停止记录，请填写指令并提交。')


    def _handle_submit_instruction(self, sid, instruction_text):
        """Handles instruction submission and finalizes episode data."""
        self._emit_log(sid, f"Worker: Received instruction for SID {sid}: '{instruction_text}'", level='info')
        if sid not in self.client_current_episode or not self.client_current_episode[sid].get('pending_submission', False):
            self._emit_error(sid, "Worker: No active episode pending submission. Please start/stop recording first.")
            self._emit_status_update(sid, 'pending_submission', '⚠️ 未停止记录或指令已提交，无法提交。')
            return

        current_episode = self.client_current_episode[sid]
        familiarization_steps = current_episode.get('familiarization_steps', 0)
        if familiarization_steps < MIN_FAMILIARIZATION_STEPS:
            remaining_steps = max(0, MIN_FAMILIARIZATION_STEPS - familiarization_steps)
            self._emit_error(sid, "Worker: Instruction submission is locked before the familiarization step threshold is met.")
            self._emit_status_update(
                sid,
                'pending_submission',
                f'⚠️ 当前仅完成 {familiarization_steps}/{MIN_FAMILIARIZATION_STEPS} 步场景熟悉，还需 {remaining_steps} 步后才能提交指令。'
            )
            return
        
        current_episode["instruction"]["instruction_text"] = instruction_text
        
        # Calculate geodesic_distance or Euclidean distance
        start_pos = np.array(current_episode["start_position"])
        end_pos = np.array(current_episode["goals"][0]["position"])
        calculated_distance = 0.0

        if self.sim and self.sim.pathfinder.is_loaded:
            try:
                # Habitat-Sim's pathfinder expects habitat_sim.Vector3 for find_path
                path = self.sim.pathfinder.find_path(
                    habitat_sim.Vector3(start_pos[0], start_pos[1], start_pos[2]),
                    habitat_sim.Vector3(end_pos[0], end_pos[1], end_pos[2])
                )
                if path.length() > 0:
                    calculated_distance = path.geodesic_distance
                    logger.info(f"Worker: Calculated geodesic distance: {calculated_distance:.2f}m")
                else:
                    logger.warning("Worker: Pathfinder found path of length 0. Falling back to Euclidean distance.")
                    calculated_distance = np.linalg.norm(end_pos - start_pos)
            except Exception as e:
                logger.error(f"Worker: Error calculating geodesic distance: {e}. Falling back to Euclidean distance.")
                logger.error(traceback.format_exc())
                calculated_distance = np.linalg.norm(end_pos - start_pos)
        else:
            logger.warning("Worker: Pathfinder not loaded. Using Euclidean distance.")
            calculated_distance = np.linalg.norm(end_pos - start_pos)
        
        current_episode["info"]["geodesic_distance"] = float(calculated_distance)

        # Create a clean episode dictionary for saving, excluding internal state flags
        episode_to_save = {
            k: v for k, v in current_episode.items() 
            if k not in ["recording_active", "pending_submission", "initial_agent_state_in_scene"] # Exclude internal state flags
        }

        self.all_trajectories_log.append(episode_to_save)
        self._save_trajectory_data()
        
        self._emit_log(sid, f"Worker: Episode {current_episode['episode_id']} finalized and saved for SID {sid}.", level='info')
        
        # Reset current episode for this SID to allow new recording or scene switch
        scene_id_for_next = current_episode['scene_id']
        next_episode_initial_state_serialized = None
        if self.sim: # If simulator is still active, get current agent state for the next initial state
            current_agent_state_in_sim = self.sim.get_agent(self.agent_id).get_state()
            next_episode_initial_state_serialized = {
                "position": current_agent_state_in_sim.position.tolist(),
                "rotation": [current_agent_state_in_sim.rotation.w, current_agent_state_in_sim.rotation.x, current_agent_state_in_sim.rotation.y, current_agent_state_in_sim.rotation.z]
            }
        
        self.client_current_episode[sid] = {
            "episode_id": None,
            "trajectory_id": None,
            "scene_id": scene_id_for_next,
            "start_position": None,
            "start_rotation": None,
            "info": {"geodesic_distance": 0.0},
            "goals": [],
            "instruction": {"instruction_text": "no instruction yet"},
            "reference_path": [],
            "action_sequence": [], # Reset action sequence
            "recording_active": False,
            "pending_submission": False,
            "familiarization_steps": current_episode.get('familiarization_steps', 0),
            "initial_agent_state_in_scene": next_episode_initial_state_serialized
        }
        self._emit_llm_response(sid, '等待大模型分析...') # Reset LLM output after submission
        self._emit_observation(sid, self._get_observation(), f'✅ 指令已提交，路径已记录！', 'idle')
        self._emit_status_update(sid, 'idle', '✅ 指令已提交，可以开始新的记录或切换场景。')

    def _finalize_current_episode(self, sid):
        """Helper to save an ongoing episode if it wasn't explicitly saved (e.g., scene switch, disconnect, new recording start)."""
        if sid in self.client_current_episode:
            episode_data = self.client_current_episode[sid]
            
            # Check if there's an actual episode ID assigned and if it's either recording or pending submission
            if episode_data["episode_id"] is not None and \
               (episode_data.get("recording_active", False) or episode_data.get("pending_submission", False)):
                
                # If instruction is incomplete, mark it as auto-saved
                if episode_data["instruction"]["instruction_text"] == "instruction incomplete":
                    episode_data["instruction"]["instruction_text"] = "instruction incomplete (auto-saved)"
                    
                # If no explicit goal was set (e.g., recording stopped but not submitted), use the last recorded position as a default goal
                if not episode_data["goals"] and episode_data["reference_path"]:
                    last_pos = np.array(episode_data["reference_path"][-1])
                    episode_data["goals"].append({"position": last_pos.tolist(), "radius": 0.2})
                    logger.warning(f"Worker: No explicit goal set for episode {episode_data['episode_id']}, using last recorded position as goal.")

                # If recording was active but stopped abruptly, append 'stop'
                if episode_data.get("recording_active", False):
                    episode_data["action_sequence"].append("stop")
                    logger.warning(f"Worker: Auto-finalizing: Appended 'stop' to action sequence for episode {episode_data['episode_id']}")

                # Recalculate geodesic_distance for auto-saved episodes
                start_pos = np.array(episode_data["start_position"]) if episode_data["start_position"] else None
                end_pos = np.array(episode_data["goals"][0]["position"]) if episode_data["goals"] else None
                calculated_distance = 0.0
                if start_pos is not None and end_pos is not None:
                    if self.sim and self.sim.pathfinder.is_loaded:
                        try:
                            path = self.sim.pathfinder.find_path(
                                habitat_sim.Vector3(start_pos[0], start_pos[1], start_pos[2]),
                                habitat_sim.Vector3(end_pos[0], end_pos[1], end_pos[2])
                            )
                            if path.length() > 0:
                                calculated_distance = path.geodesic_distance
                            else:
                                calculated_distance = np.linalg.norm(end_pos - start_pos)
                        except Exception as e:
                            logger.error(f"Worker: Error calculating geodesic distance during auto-finalize: {e}. Falling back to Euclidean.")
                            calculated_distance = np.linalg.norm(end_pos - start_pos)
                    else:
                        calculated_distance = np.linalg.norm(end_pos - start_pos)
                
                episode_data["info"]["geodesic_distance"] = float(calculated_distance)
                
                # Create a clean episode dictionary for saving
                episode_to_save = {
                    k: v for k, v in episode_data.items() 
                    if k not in ["recording_active", "pending_submission", "initial_agent_state_in_scene"]
                }
                self.all_trajectories_log.append(episode_to_save)
                self._save_trajectory_data()
                logger.warning(f"Worker: Auto-finalizing incomplete episode {episode_data['episode_id']} for SID {sid}")
            
            # Reset the client_current_episode state for this SID, allowing a fresh start
            scene_id_for_next = self.client_current_episode[sid]['scene_id'] if self.client_current_episode[sid]['scene_id'] else "unknown_scene"
            
            next_episode_initial_state_serialized = None
            if self.sim:
                current_agent_state_in_sim = self.sim.get_agent(self.agent_id).get_state()
                next_episode_initial_state_serialized = {
                    "position": current_agent_state_in_sim.position.tolist(),
                    "rotation": [current_agent_state_in_sim.rotation.w, current_agent_state_in_sim.rotation.x, current_agent_state_in_sim.rotation.y, current_agent_state_in_sim.rotation.z]
                }

            self.client_current_episode[sid] = {
                "episode_id": None,
                "trajectory_id": None,
                "scene_id": scene_id_for_next,
                "start_position": None,
                "start_rotation": None,
                "info": {"geodesic_distance": 0.0},
                "goals": [],
                "instruction": {"instruction_text": "no instruction yet"},
                "reference_path": [],
                "action_sequence": [],
                "recording_active": False,
                "pending_submission": False,
                "familiarization_steps": episode_data.get('familiarization_steps', 0),
                "initial_agent_state_in_scene": next_episode_initial_state_serialized
            }

# Initialize SceneLoader to get scene paths once at startup
scene_loader = SceneLoader()

# Global worker instance, will be started by SocketIO
simulator_worker = None

@app.route('/')
def index():
    """Provides the frontend HTML page."""
    logger.info("🌐 收到 HTTP GET / 请求，发送前端页面。")
    return render_template_string('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Habitat Simulator (调试版)</title>
    <!-- Use protocol-relative URL to avoid mixed content issues -->
    <script src="//cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh; /* Ensure body takes full viewport height */
            box-sizing: border-box;
        }
        .main-container {
            display: flex;
            gap: 20px;
            width: 100%;
            max-width: 1200px; /* Limit overall width */
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
            justify-content: center;
        }
        .left-panel, .right-panel {
            flex: 1; /* Both panels take equal width */
            min-width: 300px; /* Minimum width before wrapping */
            display: flex;
            flex-direction: column;
            /* align-items: center; */ /* Changed for left-aligned text inputs */
        }
        .left-panel {
            align-items: center; /* Canvas should be centered */
        }
        .right-panel {
            align-items: stretch; /* Stretch children to fill width */
        }

        h1 {
            color: #4CAF50;
            margin-bottom: 20px;
            width: 100%;
            text-align: center;
        }
        #viewport {
            border: 3px solid #4CAF50;
            border-radius: 10px;
            background: #000;
            max-width: 100%;
            height: auto;
            display: block; /* Remove extra space below canvas */
        }
        .controls-group {
            margin-bottom: 20px; /* Space between groups */
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
            width: 100%;
            box-sizing: border-box; /* Include padding in width */
        }
        .controls-group h3 {
            margin-top: 0;
            color: #eee;
            text-align: center;
        }
        
        #status {
            margin: 10px 0;
            padding: 10px;
            background: #333;
            border-radius: 5px;
            min-height: 20px;
            font-family: monospace;
            color: #FFFF00; /* Yellow for status */
            width: calc(100% - 20px); /* Adjusted for padding */
            box-sizing: border-box;
            text-align: center;
        }
        #debug-log {
            background: #222;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px; /* Added margin-top */
            text-align: left;
            font-family: monospace;
            font-size: 12px;
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid #444;
            color: #888; /* Gray for debug logs */
            width: 100%; /* Changed to 100% */
            box-sizing: border-box;
        }
        .info {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px; /* Added margin-top */
            text-align: left;
            width: 100%; /* Changed to 100% */
            box-sizing: border-box;
        }
        .scene-record-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px; /* Space between buttons */
            justify-content: center;
            margin-top: 10px;
        }
        .scene-btn, .record-btn, .llm-btn { /* Added .llm-btn */
            background: #ff6b35;
            border: 2px solid #ff6b35;
            color: #fff;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 15px;
            flex: 1; /* Allow buttons to grow and shrink */
            min-width: 150px; /* Minimum width for buttons */
            transition: all 0.2s;
            box-sizing: border-box;
        }
        /* Specific style for the new LLM button */
        .llm-btn {
            background: #6a5acd; /* Royal Blue */
            border-color: #6a5acd;
            flex-basis: 100%; /* Make it take full width in its container */
        }
        .scene-btn:hover:not(:disabled), .record-btn:hover:not(:disabled), .llm-btn:hover:not(:disabled) {
            background: #ff8c42;
        }
        .llm-btn:hover:not(:disabled) {
            background: #8470ff;
        }
        .scene-btn:disabled, .record-btn:disabled, .llm-btn:disabled {
            background-color: #555;
            border-color: #666;
            color: #bbb;
            cursor: not-allowed;
            opacity: 0.7;
        }

        #submitInstructionBtn {
            background: #007bff;
            border-color: #007bff;
            color: #fff;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 15px;
            margin-top: 10px;
            width: 100%;
            transition: all 0.2s;
            box-sizing: border-box;
        }
        #submitInstructionBtn:hover:not(:disabled) {
            background: #0056b3;
        }
        #submitInstructionBtn:disabled {
            background-color: #555;
            border-color: #666;
            color: #bbb;
            cursor: not-allowed;
            opacity: 0.7;
        }

        .debug-badge {
            background: #ff3333;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        #instructionInput {
            margin-top: 0px; /* Removed margin-top */
            width: 100%; /* Changed to 100% */
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #4CAF50;
            background: #2a2a2a;
            color: #fff;
            font-size: 14px;
            resize: vertical;
            box-sizing: border-box;
            min-height: 80px; /* Give it some default height */
        }
        #llm-output {
            background: #3a3a3a;
            padding: 10px;
            border-radius: 5px;
            min-height: 50px;
            color: #ADD8E6; /* Light blue for LLM output */
            text-align: left;
            font-family: monospace;
            font-size: 14px;
            word-wrap: break-word; /* Allow long text to wrap */
            white-space: pre-wrap; /* Preserve whitespace and line breaks */
        }
    </style>
</head>
<body>
    <h1>🏠 Habitat Simulator <span class="debug-badge">调试版</span></h1>
    <div id="status">🔌 正在连接...</div>
    
    <div class="main-container">
        <div class="left-panel">
            <canvas id="viewport" width="640" height="480"></canvas>
        </div>
        
        <div class="right-panel">
            <div class="controls-group">
                <h3>场景与记录</h3>
                <div id="annotation-progress" style="margin-bottom:10px; color:#ffd166; font-size:14px;">
                    当前处于场景熟悉阶段，完成足够步数后才可进入正式标注。
                </div>
                <div class="scene-record-buttons">
                    <button class="scene-btn" onclick="switchScene()" id="btnSwitchScene">🔄 切换场景 (Tab)</button>
                    <button class="record-btn" onclick="startRecording()" id="btnStartRecording">⏺️ 开始记录路径</button>
                    <button class="record-btn" onclick="stopRecording()" id="btnStopRecording">⏹️ 停止记录路径</button>
                </div>
            </div>

            <div class="controls-group">
                <h3>指令提交</h3>
                <textarea id="instructionInput" placeholder="在此输入指令文本..." rows="3" cols="60"></textarea>
                <button id="submitInstructionBtn">📝 提交指令</button>
            </div>
            
            <div class="controls-group">
                <h3>大模型分析结果</h3>
                <div class="scene-record-buttons"> <!-- Reusing this class for layout -->
                    <button class="llm-btn" id="btnTriggerLLM">🧠 请求大模型分析</button> <!-- New LLM Button -->
                </div>
                <div id="llm-output">点击“请求大模型分析”按钮以获取分析结果。</div>
            </div>

            <div id="debug-log">调试日志将显示在这里...</div>
            
            <div class="info">
                <strong>🔧 调试信息:</strong><br>
                • 此版本包含详细的调试输出<br>
                • 检查浏览器控制台 (F12) 和服务器终端日志<br>
                • 日志文件: <code>habitat_debug.log</code> (与脚本同目录)<br>
                • 轨迹数据: <code>trajectories.json</code> (与脚本同目录)<br>
                <strong>键盘控制说明:</strong><br>
                • 上方向键: 前进, 下方向键: 后退, 左方向键: 左转, 右方向键: 右转<br>
                • PageUp: 向上看, PageDown: 向下看<br>
                • Tab: 切换场景
            </div>
        </div>
    </div>

    <script>
        // Debug log function, outputs to page and browser console
        function addDebugLog(message) {
            const debugLogElement = document.getElementById('debug-log');
            const timestamp = new Date().toLocaleTimeString();
        
            // 创建新的元素并追加，而不是修改 innerHTML
            const newLogEntrySpan = document.createElement('span');
            newLogEntrySpan.textContent = `[${timestamp}] ${message}`;
            debugLogElement.appendChild(newLogEntrySpan);
            debugLogElement.appendChild(document.createElement('br')); // 添加换行符

            // 限制日志条目数量，例如只保留最后 50 条
            const maxLogEntries = 50;
            // 由于每个日志条目由一个 <span> 和一个 <br> 组成，所以要移除两倍的子元素
            while (debugLogElement.children.length > maxLogEntries * 2) {
                debugLogElement.removeChild(debugLogElement.firstChild); // 移除最旧的 <span>
                debugLogElement.removeChild(debugLogElement.firstChild); // 移除最旧的 <br>
            }

            debugLogElement.scrollTop = debugLogElement.scrollHeight; // 自动滚动到底部
            // console.log(`[FRONTEND_DEBUG] ${message}`); // 保持此行，方便在控制台查看完整日志
        }
        
        addDebugLog('🌐 页面加载完成。');
        
        // Connect WebSocket
        addDebugLog('🔌 尝试连接 WebSocket 服务器...');
        const socket = io({
            transports: ['websocket', 'polling'], // Prioritize WebSocket, fallback to polling
            timeout: 20000, // Increased timeout for initial connection, as sim init can be slow
            forceNew: true // Ensure a new connection on each refresh
        });
        
        const canvas = document.getElementById('viewport');
        const ctx = canvas.getContext('2d');
        const statusElement = document.getElementById('status');
        const llmOutputElement = document.getElementById('llm-output'); // Get LLM output display element
        const annotationProgressElement = document.getElementById('annotation-progress');

        // Recording State: 'idle', 'recording', 'pending_submission'
        let recordingState = 'idle'; 
        let familiarizationSteps = 0;
        let minFamiliarizationSteps = 200;
        let canAnnotate = false;

        // Get button elements
        const btnStartRecording = document.getElementById('btnStartRecording');
        const btnStopRecording = document.getElementById('btnStopRecording');
        const btnSwitchScene = document.getElementById('btnSwitchScene');
        const submitInstructionBtn = document.getElementById('submitInstructionBtn');
        const instructionInput = document.getElementById('instructionInput');
        const btnTriggerLLM = document.getElementById('btnTriggerLLM'); // New: Get LLM trigger button

        // No more movement buttons on the page, so this array is not needed for button state updates
        // const movementButtons = []; 

        function updateAnnotationProgress() {
            const remainingSteps = Math.max(0, minFamiliarizationSteps - familiarizationSteps);
            if (canAnnotate) {
                annotationProgressElement.textContent = `已完成场景熟悉：${familiarizationSteps}/${minFamiliarizationSteps} 步，可以开始正式标注。`;
                instructionInput.placeholder = '在此输入最终导航指令文本...';
            } else {
                annotationProgressElement.textContent = `场景熟悉中：${familiarizationSteps}/${minFamiliarizationSteps} 步，还需 ${remainingSteps} 步后才可开始正式标注。`;
                instructionInput.placeholder = `还需熟悉场景 ${remainingSteps} 步后，才可填写并提交指令`;
            }
        }

        function updateButtonStates() {
            addDebugLog(`🔄 Updating button states. Current state: ${recordingState}`);
            switch (recordingState) {
                case 'idle':
                    btnStartRecording.disabled = !canAnnotate;
                    btnStopRecording.disabled = true;
                    btnSwitchScene.disabled = false;
                    submitInstructionBtn.disabled = true;
                    instructionInput.disabled = true;
                    btnTriggerLLM.disabled = false; // LLM button always enabled when idle
                    break;
                case 'recording':
                    btnStartRecording.disabled = true;
                    btnStopRecording.disabled = false;
                    btnSwitchScene.disabled = true;
                    submitInstructionBtn.disabled = true;
                    instructionInput.disabled = true;
                    btnTriggerLLM.disabled = false; // LLM button can be pressed during recording
                    break;
                case 'pending_submission':
                    btnStartRecording.disabled = true;
                    btnStopRecording.disabled = true;
                    btnSwitchScene.disabled = true;
                    submitInstructionBtn.disabled = !canAnnotate;
                    instructionInput.disabled = !canAnnotate;
                    btnTriggerLLM.disabled = false; // LLM button can be pressed when pending submission
                    break;
            }
            updateAnnotationProgress();
        }

        // Connection events
        socket.on('connect', function() {
            addDebugLog('✅ WebSocket 连接成功！');
            statusElement.textContent = '✅ 已连接 - 正在初始化场景...';
            // LLM output is no longer reset on connect
            addDebugLog('📤 发送初始化请求到后端...');
            socket.emit('initialize');
            updateButtonStates(); // Set initial button states
        });
        
        socket.on('connect_error', function(error) {
            addDebugLog('❌ WebSocket 连接错误: ' + (error.message || error));
            statusElement.textContent = '❌ 连接错误: ' + (error.message || error);
        });
        
        socket.on('disconnect', function(reason) {
            addDebugLog('🔌 WebSocket 断开连接: ' + reason);
            statusElement.textContent = '❌ 连接断开: ' + reason;
            recordingState = 'idle'; // Reset state on disconnect
            // LLM output is no longer reset on disconnect
            updateButtonStates();
        });
        
        // Receive observation images and status
        socket.on('observation', function(data) {
            addDebugLog('🖼️ 收到图像数据。数据大小: ' + (data.image ? data.image.length : 0) + ' 字节。');
            if (data.image) {
                const img = new Image();
                img.onload = function() {
                    addDebugLog('✅ 图像加载成功，尺寸: ' + img.width + 'x' + img.height);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.onerror = function() {
                    addDebugLog('❌ 图像加载失败。检查 Base64 数据或图像格式。');
                    statusElement.textContent = '❌ 图像显示失败！';
                };
                img.src = 'data:image/jpeg;base64,' + data.image;
            } else {
                addDebugLog('❌ 收到空的图像数据。');
            }
            statusElement.textContent = data.status || '✅ 运行中';
            if (typeof data.familiarization_steps === 'number') {
                familiarizationSteps = data.familiarization_steps;
            }
            if (typeof data.min_familiarization_steps === 'number') {
                minFamiliarizationSteps = data.min_familiarization_steps;
            }
            if (typeof data.can_annotate === 'boolean') {
                canAnnotate = data.can_annotate;
            }
            if (data.recording_state) {
                recordingState = data.recording_state;
                updateButtonStates();
            } else {
                updateAnnotationProgress();
            }
        });

        // Receive general status updates (e.g., after recording start/stop)
        socket.on('status_update', function(data) {
            addDebugLog('📢 状态更新: ' + data.message);
            statusElement.textContent = data.message;
            if (typeof data.familiarization_steps === 'number') {
                familiarizationSteps = data.familiarization_steps;
            }
            if (typeof data.min_familiarization_steps === 'number') {
                minFamiliarizationSteps = data.min_familiarization_steps;
            }
            if (typeof data.can_annotate === 'boolean') {
                canAnnotate = data.can_annotate;
            }
            if (data.recording_state) {
                recordingState = data.recording_state;
                updateButtonStates();
            } else {
                updateAnnotationProgress();
            }
        });

        // Receive backend logs
        socket.on('backend_log', function(data) {
            addDebugLog('➡️ 后端日志: ' + data.message);
        });

        // Error handling (backend sent errors)
        socket.on('error', function(data) {
            addDebugLog('🚨 后端错误: ' + data.message);
            statusElement.textContent = '❌ 后端错误: ' + data.message;
        });

        // New: Receive LLM analysis results
        socket.on('llm_response', function(data) {
            addDebugLog('🧠 收到大模型分析结果。');
            llmOutputElement.textContent = data.text;
        });
        
        // Keyboard controls - remains fully functional
        document.addEventListener('keydown', function(event) {
            const key = event.key.toLowerCase();
            const validMovementKeys = ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'pageup', 'pagedown'];
            
            if (validMovementKeys.includes(key)) {
                // 只有在 'idle' 或 'recording' 状态下才允许发送移动和旋转动作
                if (recordingState === 'idle' || recordingState === 'recording') {
                    event.preventDefault(); // Prevent default scroll behavior
                    addDebugLog('⌨️ 发送动作: ' + key.toUpperCase());
                    socket.emit('action', {action: key});
                }
                else {
                    // 在 'pending_submission' 状态下，阻止动作但提供提示
                    addDebugLog('⚠️ 当前状态 (' + recordingState + ') 不允许移动或旋转。');
                    statusElement.textContent = '⚠️ 请先提交指令，或开始新的记录。'; // 给用户提示
                }
            }
            else if (event.code === 'Tab') { // Change switch scene key to Tab
                event.preventDefault(); // Prevent default browser tab behavior
                if (recordingState === 'idle') { // Only allow switching scene in idle state
                    addDebugLog('⌨️ 发送切换场景请求 (TAB)');
                    switchScene();
                } else {
                    addDebugLog('⚠️ 无法切换场景，当前正在记录或等待提交指令。');
                    statusElement.textContent = '⚠️ 无法切换场景。';
                }
            }
        });
        
        // The highlightButton function is no longer needed as there are no on-screen movement buttons
        // function highlightButton(action) {
        //     const btn = document.querySelector(`[data-action="${action}"]`);
        //     if (btn && !btn.disabled) {
        //         btn.style.background = '#4CAF50';
        //         setTimeout(() => {
        //             btn.style.background = '#333';
        //         }, 200);
        //     }
        // }
        
        // Switch scene
        function switchScene() {
            if (recordingState === 'idle') {
                statusElement.textContent = '🔄 正在切换场景...';
                // LLM output is no longer reset on scene switch
                socket.emit('switch_scene');
            } else {
                addDebugLog('⚠️ 无法切换场景，当前正在记录或等待提交指令。');
                statusElement.textContent = '⚠️ 无法切换场景。';
            }
        }

        // New: Handle start recording
        function startRecording() {
            if (recordingState === 'idle') {
                if (!canAnnotate) {
                    const remainingSteps = Math.max(0, minFamiliarizationSteps - familiarizationSteps);
                    addDebugLog(`⚠️ 还需熟悉场景 ${remainingSteps} 步后才能开始记录。`);
                    statusElement.textContent = `⚠️ 请先熟悉场景，还需 ${remainingSteps} 步。`;
                    return;
                }
                statusElement.textContent = '⏺️ 准备开始记录...';
                // LLM analysis trigger removed from here
                socket.emit('start_recording');
            } else {
                addDebugLog('⚠️ 当前状态无法开始记录。');
                statusElement.textContent = '⚠️ 当前状态无法开始记录。';
            }
        }

        // New: Handle stop recording
        function stopRecording() {
            if (recordingState === 'recording') {
                statusElement.textContent = '⏹️ 停止记录中...';
                socket.emit('stop_recording');
            } else {
                addDebugLog('⚠️ 未在记录中，无法停止。');
                statusElement.textContent = '⚠️ 未在记录中，无法停止。';
            }
        }

        // New: Handle instruction submission
        submitInstructionBtn.addEventListener('click', function() {
            if (!this.disabled) { // Check if button is enabled
                if (!canAnnotate) {
                    const remainingSteps = Math.max(0, minFamiliarizationSteps - familiarizationSteps);
                    addDebugLog(`⚠️ 还需熟悉场景 ${remainingSteps} 步后才能提交指令。`);
                    statusElement.textContent = `⚠️ 还需熟悉场景 ${remainingSteps} 步后才能提交指令。`;
                    return;
                }
                const instructionText = instructionInput.value.trim();
                if (instructionText) {
                    addDebugLog('📤 提交指令: ' + instructionText);
                    llmOutputElement.textContent = '等待大模型分析...'; // Reset LLM output after submission
                    socket.emit('submit_instruction', {instruction_text: instructionText});
                    instructionInput.value = ''; // Clear input after submission
                    statusElement.textContent = '📝 指令已发送，等待记录...';
                } else {
                    addDebugLog('⚠️ 指令文本为空，未提交。');
                    statusElement.textContent = '⚠️ 请输入指令。';
                }
            }
        });

        // New: Handle trigger LLM analysis button click
        btnTriggerLLM.addEventListener('click', function() {
            if (!this.disabled) {
                addDebugLog('📤 请求大模型分析。');
                llmOutputElement.textContent = '大模型正在分析图像...'; // Indicate analysis in progress
                socket.emit('trigger_llm_analysis');
            }
        });

        // Initial call to set button states on page load
        updateButtonStates();
    </script>
</body>
</html>
    ''')

# --- Socket.IO Event Handlers (now send commands to the worker queue) ---

@socketio.on('connect')
def handle_connect():
    """Client connected, emit initial log and ensure worker is running."""
    logger.info(f"⚡️ WebSocket client connected! SID: {request.sid}")
    # Emit a log message to the specific client that just connected
    emit('backend_log', {'message': f'服务器已连接，您的会话ID是: {request.sid}'}, room=request.sid)

    global simulator_worker
    # The worker is started globally before app.run(). This check is a safeguard.
    if simulator_worker is None or not simulator_worker._running:
        logger.warning("Simulator worker was not running, attempting to start it now. (This should ideally be started once at app startup)")
        simulator_worker = SimulatorWorker(socketio, scene_loader.scene_paths)
        simulator_worker.start()

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected."""
    logger.info(f"🔌 WebSocket client disconnected. SID: {request.sid}")
    # Finalize episode for the disconnected client
    if simulator_worker:
        simulator_worker._finalize_current_episode(request.sid)


@socketio.on('initialize')
def handle_initialize():
    """Enqueue initialize command for the worker."""
    logger.info(f"Main Thread: Received 'initialize' request from SID: {request.sid}. Enqueuing command.")
    # Put command into the worker's queue, including the client's SID
    simulator_worker.command_queue.put({'type': Command.INIT_SIM, 'sid': request.sid})
    emit('backend_log', {'message': '初始化请求已发送到模拟器工作线程。'}, room=request.sid)

@socketio.on('action')
def handle_action(data):
    """Enqueue action command for the worker."""
    action_key = data.get('action', '')
    logger.debug(f"Main Thread: Received action '{action_key}' from SID: {request.sid}. Enqueuing command.")
    simulator_worker.command_queue.put({'type': Command.EXECUTE_ACTION, 'sid': request.sid, 'action_key': action_key})
    emit('backend_log', {'message': f'动作 "{action_key.upper()}" 请求已发送到模拟器工作线程。'}, room=request.sid)

@socketio.on('switch_scene')
def handle_switch_scene():
    """Enqueue switch scene command for the worker."""
    logger.info(f"Main Thread: Received 'switch_scene' request from SID: {request.sid}. Enqueuing command.")
    simulator_worker.command_queue.put({'type': Command.SWITCH_SCENE, 'sid': request.sid})
    emit('backend_log', {'message': '切换场景请求已发送到模拟器工作线程。'}, room=request.sid)

@socketio.on('start_recording')
def handle_start_recording():
    """Enqueue start recording command for the worker."""
    logger.info(f"Main Thread: Received 'start_recording' request from SID: {request.sid}. Enqueuing command.")
    simulator_worker.command_queue.put({'type': Command.START_RECORDING, 'sid': request.sid})
    emit('backend_log', {'message': '开始记录请求已发送到模拟器工作线程。'}, room=request.sid)

@socketio.on('stop_recording')
def handle_stop_recording():
    """Enqueue stop recording command for the worker."""
    logger.info(f"Main Thread: Received 'stop_recording' request from SID: {request.sid}. Enqueuing command.")
    simulator_worker.command_queue.put({'type': Command.STOP_RECORDING, 'sid': request.sid})
    emit('backend_log', {'message': '停止记录请求已发送到模拟器工作线程。'}, room=request.sid)

@socketio.on('submit_instruction')
def handle_submit_instruction(data):
    """Enqueue submit instruction command for the worker."""
    instruction_text = data.get('instruction_text', '')
    logger.info(f"Main Thread: Received 'submit_instruction' request from SID: {request.sid} with text: '{instruction_text}'. Enqueuing command.")
    simulator_worker.command_queue.put({'type': Command.SUBMIT_INSTRUCTION, 'sid': request.sid, 'instruction_text': instruction_text})
    emit('backend_log', {'message': '提交指令请求已发送到模拟器工作线程。'}, room=request.sid)

@socketio.on('trigger_llm_analysis') # 新增 Socket.IO 事件处理器
def handle_trigger_llm_analysis():
    """Enqueue LLM analysis command for the worker."""
    logger.info(f"Main Thread: Received 'trigger_llm_analysis' request from SID: {request.sid}. Enqueuing command.")
    simulator_worker.command_queue.put({'type': Command.TRIGGER_LLM_ANALYSIS, 'sid': request.sid})
    emit('backend_log', {'message': '大模型分析请求已发送到模拟器工作线程。'}, room=request.sid)

# --- Application Startup ---
if __name__ == '__main__':
    _host = os.environ.get('NAVSPACE_HOST', '0.0.0.0')
    _port = int(os.environ.get('NAVSPACE_PORT', '8000'))
    logger.info("🚀 启动 Flask-SocketIO 服务器...")
    logger.info(f"🌐 请在浏览器中访问: http://localhost:{_port}")

    # Perform Habitat-Sim installation check once before starting the worker.
    # This ensures that any fundamental issues (like missing OSMesa) are caught early.
    def perform_habitat_install_check():
        logger.debug("🔧 运行 Habitat-Sim 安装健全性检查...")
        try:
            logger.debug(f"Habitat-Sim 版本: {habitat_sim.__version__}")
            logger.debug(f"Habitat-Sim 安装路径: {habitat_sim.__file__}")
            test_sim_cfg = habitat_sim.SimulatorConfiguration()
            logger.debug(f"  test_sim_cfg.gpu_device_id: {test_sim_cfg.gpu_device_id}")
            logger.debug(f"  test_sim_cfg.enable_physics: {test_sim_cfg.enable_physics}")
            logger.debug("✅ Habitat-Sim 基本配置对象创建成功。")
        except Exception as e:
            logger.critical(f"❌ Habitat-Sim 导入或版本检查失败: {e}")
            logger.critical(traceback.format_exc())
            logger.critical("请确保 Habitat-Sim 已正确安装并配置了无头渲染（如OSMesa）。")
            sys.exit(1) # Exit if core Habitat-Sim isn't working

    perform_habitat_install_check()

    # Initialize and start the single simulator worker thread *before* the Flask-SocketIO app runs.
    # This ensures the worker is ready when the first client connects.
    # It must be initialized with the socketio instance and scene paths.
    simulator_worker = SimulatorWorker(socketio, scene_loader.scene_paths)
    simulator_worker.start()

    # Run the Flask-SocketIO application.
    # IMPORTANT: Set debug=False in production to avoid multiple worker processes.
    socketio.run(app, host=_host, port=_port, debug=False, allow_unsafe_werkzeug=True)

    # After socketio.run exits (e.g., Ctrl+C), send stop command to worker
    logger.info("Flask-SocketIO app is shutting down. Stopping simulator worker.")
    if simulator_worker: # Ensure worker exists before trying to stop
        simulator_worker.stop()

