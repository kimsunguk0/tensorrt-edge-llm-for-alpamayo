import carla
import numpy as np
import math
import queue
import uuid
import os
import argparse
import io
import json
import threading
import matplotlib.pyplot as plt
import mediapy as mp
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

# ==========================================
# 1. NVIDIA Alpamayo-R1 기반 좌표계 설정
# ==========================================
OFFSETS = {'yaw': -90.0, 'roll': -90.0, 'pitch': 0.0}
DEFAULT_SEND_WIDTH = 576
DEFAULT_SEND_HEIGHT = 320
DEFAULT_MIN_PIXELS = 163840
DEFAULT_MAX_PIXELS = 196608
DEFAULT_PATCH_SIZE = 16
DEFAULT_MERGE_SIZE = 2
DEFAULT_PATCH_FACTOR = DEFAULT_PATCH_SIZE * DEFAULT_MERGE_SIZE

def get_corrected_transform(loc, quat):
    """
    NVIDIA Alpamayo 소스코드의 좌표계 보정 로직 반영
    - CARLA Y축 반전 (y = -y)
    - 쿼터니언 성분 보정 및 오프셋 적용
    """
    x, y, z = loc
    loc_carla = carla.Location(x=x, y=-y, z=z)
    
    # NVIDIA 코드 기준 쿼터니언 순서: [qx, qy, qz, qw]
    qx, qy, qz, qw = quat
    qx, qy, qz, qw = qx, -qy, qz, -qw
    
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return carla.Transform(
        loc_carla, 
        carla.Rotation(
            pitch=math.degrees(pitch) + OFFSETS['pitch'], 
            yaw=math.degrees(yaw) + OFFSETS['yaw'], 
            roll=math.degrees(roll) + OFFSETS['roll']
        )
    )
# def get_corrected_transform(loc, quat):
#     """
#     Y축 반전 로직이 제거된 좌표계 보정 로직
#     - CARLA 기본 좌표계(또는 입력 데이터 그대로) 유지
#     - 쿼터니언 성분 반전 삭제
#     """
#     x, y, z = loc
#     # Y축 반전 삭제: y = -y에서 y = y로 변경
#     loc_carla = carla.Location(x=x, y=y, z=z)
    
#     # 쿼터니언 성분 보정 삭제: 원본 순서 [qx, qy, qz, qw] 유지
#     qx, qy, qz, qw = quat
    
#     # Quaternion to Euler 변환 (표준 공식)
#     sinr_cosp = 2 * (qw * qx + qy * qz)
#     cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
#     roll = math.atan2(sinr_cosp, cosr_cosp)
    
#     sinp = 2 * (qw * qy - qz * qx)
#     pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    
#     siny_cosp = 2 * (qw * qz + qx * qy)
#     cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
#     yaw = math.atan2(siny_cosp, cosy_cosp)
    
#     return carla.Transform(
#         loc_carla, 
#         carla.Rotation(
#             pitch=math.degrees(pitch) + OFFSETS['pitch'], 
#             yaw=math.degrees(yaw) + OFFSETS['yaw'], 
#             roll=math.degrees(roll) + OFFSETS['roll']
#         )
#     )

def get_matrix(transform):
    return np.array(transform.get_matrix())


def round_to_factor(value, factor):
    return max(factor, int(round(value / factor) * factor))


def smart_resize_dimensions(height, width, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
                            patch_factor=DEFAULT_PATCH_FACTOR):
    """
    Qwen-style dynamic resize:
    - snap to patch_factor multiples first
    - if pixel count is too large, scale down
    - if pixel count is too small, scale up
    """
    rounded_height = round_to_factor(height, patch_factor)
    rounded_width = round_to_factor(width, patch_factor)
    rounded_pixels = rounded_height * rounded_width

    if rounded_pixels > max_pixels:
        beta = math.sqrt(rounded_pixels / max_pixels)
        target_height = max(patch_factor, int(math.floor(height / beta / patch_factor) * patch_factor))
        target_width = max(patch_factor, int(math.floor(width / beta / patch_factor) * patch_factor))
    elif rounded_pixels < min_pixels:
        beta = math.sqrt(min_pixels / rounded_pixels)
        target_height = max(patch_factor, int(math.ceil(height * beta / patch_factor) * patch_factor))
        target_width = max(patch_factor, int(math.ceil(width * beta / patch_factor) * patch_factor))
    else:
        target_height = rounded_height
        target_width = rounded_width

    return target_height, target_width


def resize_chw_image(image_chw):
    """Resize a CHW RGB image using Qwen-style smart resize and bicubic interpolation."""
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    orig_height, orig_width = image_hwc.shape[:2]
    target_height, target_width = smart_resize_dimensions(orig_height, orig_width)
    pil_image = Image.fromarray(image_hwc)
    resized = pil_image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    resized_hwc = np.asarray(resized, dtype=np.uint8)
    return np.transpose(resized_hwc, (2, 0, 1))


def resize_image_frames(image_frames, target_width, target_height):
    """
    Resize image frames from [Cam, T, C, H, W] using Qwen-style smart resize.
    target_width/target_height are kept for compatibility with the existing call sites.
    """
    num_cams, num_steps = image_frames.shape[:2]
    orig_height = int(image_frames.shape[3])
    orig_width = int(image_frames.shape[4])
    target_height, target_width = smart_resize_dimensions(orig_height, orig_width)
    resized = np.empty((num_cams, num_steps, 3, target_height, target_width), dtype=np.uint8)
    for cam_idx in range(num_cams):
        for step_idx in range(num_steps):
            resized[cam_idx, step_idx] = resize_chw_image(image_frames[cam_idx, step_idx])
    return resized


def sample_to_npz_bytes(sample, target_width, target_height):
    """
    Serialize an online sample to compressed NPZ bytes for network transfer.
    """
    resized_frames = resize_image_frames(sample["image_frames"], target_width, target_height)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        image_frames=resized_frames.astype(np.uint8),
        camera_indices=np.asarray(sample["camera_indices"], dtype=np.int32),
        ego_history_xyz=np.asarray(sample["ego_history_xyz"], dtype=np.float32),
        ego_history_rot=np.asarray(sample["ego_history_rot"], dtype=np.float32),
        relative_timestamps=np.asarray(sample["relative_timestamps"], dtype=np.float32),
        absolute_timestamps=np.asarray(sample["absolute_timestamps"], dtype=np.int64),
        t0_us=np.asarray([sample["t0_us"]], dtype=np.int64),
        fixed_delta_seconds=np.asarray([sample["fixed_delta_seconds"]], dtype=np.float32),
        clip_id=np.asarray([sample["clip_id"]], dtype="<U32"),
        camera_order=np.asarray(sample["camera_order"], dtype="<U32"),
    )
    return buffer.getvalue()


def log_online_history(sample):
    """
    Print ego history values before the sample is published to the network.
    """
    history_xyz = sample["ego_history_xyz"][0, 0]
    history_rot = sample["ego_history_rot"][0, 0]
    yaw_deg = np.degrees(np.arctan2(history_rot[:, 0, 1], history_rot[:, 0, 0]))

    print(">> ego_history_xyz (pre-send):")
    print(np.array2string(history_xyz, precision=4, suppress_small=False, max_line_width=200))
    print(f">> ego_history_xyz first={history_xyz[0].tolist()} last={history_xyz[-1].tolist()}")
    print(">> ego_history_yaw_deg (pre-send):")
    print(np.array2string(yaw_deg, precision=3, suppress_small=False, max_line_width=200))


class SampleStore:
    """
    Keeps both:
    - the newest observed sample for legacy /latest polling
    - a bounded FIFO queue of published samples for /next consumption
    """

    def __init__(self, queue_max_size):
        self._lock = threading.Lock()
        self._latest_sample = None
        self._latest_sequence = 0
        self._queue = deque(maxlen=queue_max_size)
        self._queue_sequence = 0
        self._dropped_queue_samples = 0

    def update_latest(self, sample):
        with self._lock:
            self._latest_sample = sample
            self._latest_sequence += 1
            return self._latest_sequence

    def enqueue(self, sample):
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._queue.popleft()
                self._dropped_queue_samples += 1
            self._queue_sequence += 1
            sequence = self._queue_sequence
            self._queue.append((sample, sequence))
            return sequence, len(self._queue)

    def snapshot_latest(self):
        with self._lock:
            if self._latest_sample is None:
                return None, 0
            return self._latest_sample, self._latest_sequence

    def pop_next(self):
        with self._lock:
            if not self._queue:
                return None, 0, 0
            sample, sequence = self._queue.popleft()
            return sample, sequence, len(self._queue)

    def stats(self):
        with self._lock:
            latest_t0_us = None
            latest_clip_id = None
            if self._latest_sample is not None:
                latest_t0_us = int(self._latest_sample["t0_us"])
                latest_clip_id = str(self._latest_sample["clip_id"])
            return {
                "latest_available": self._latest_sample is not None,
                "latest_sequence": self._latest_sequence,
                "queue_size": len(self._queue),
                "queue_max_size": self._queue.maxlen,
                "queued_sequence": self._queue_sequence,
                "dropped_queue_samples": self._dropped_queue_samples,
                "latest_t0_us": latest_t0_us,
                "latest_clip_id": latest_clip_id,
            }


def make_sample_http_handler(sample_store, target_width, target_height):
    class SampleRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                stats = sample_store.stats()
                payload = {
                    "status": "ok",
                    "sample_available": stats["latest_available"],
                    "latest_sequence": stats["latest_sequence"],
                    "queued_sequence": stats["queued_sequence"],
                    "queue_size": stats["queue_size"],
                    "queue_max_size": stats["queue_max_size"],
                    "dropped_queue_samples": stats["dropped_queue_samples"],
                    "target_width": target_width,
                    "target_height": target_height,
                }
                if stats["latest_available"]:
                    payload["t0_us"] = stats["latest_t0_us"]
                    payload["clip_id"] = stats["latest_clip_id"]

                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/latest":
                sample, sequence = sample_store.snapshot_latest()
                if sample is None:
                    body = json.dumps({"error": "sample_not_ready"}).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                payload = sample_to_npz_bytes(sample, target_width, target_height)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Sample-Sequence", str(sequence))
                self.send_header("X-T0-US", str(sample["t0_us"]))
                self.send_header("X-Clip-ID", str(sample["clip_id"]))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path == "/next":
                sample, sequence, remaining_queue_size = sample_store.pop_next()
                if sample is None:
                    self.send_response(204)
                    self.send_header("X-Queue-Size", "0")
                    self.end_headers()
                    return

                payload = sample_to_npz_bytes(sample, target_width, target_height)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Sample-Sequence", str(sequence))
                self.send_header("X-T0-US", str(sample["t0_us"]))
                self.send_header("X-Clip-ID", str(sample["clip_id"]))
                self.send_header("X-Queue-Size", str(remaining_queue_size))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            print(f"[sample_server] {self.address_string()} - {format % args}")

    return SampleRequestHandler

# ==========================================
# 2. Data Grabber 클래스
# ==========================================
class SimDataGrabber:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        # Keep a local copy of the fixed delta so downstream code can time‑sync
        # trajectories to CARLA's simulation clock.
        self.fixed_delta_seconds = 0.1
        
        self.hero_actor = None
        self.sensors = []
        self.sensor_queues = {}
        
        self.camera_indices = np.array([0, 1, 2, 6])
        self.clip_id = str(uuid.uuid4())[:8]
        
        # Alpamayo 표준 카메라 설정
        self.CAMERA_CONFIGS = [
            {"id": "cross_left", "trans": (2.473, 0.938, 0.918), "quat": (0.698, -0.146, 0.147, -0.686), "fov": 120},
            {"id": "front_wide", "trans": (1.697, -0.010, 1.436), "quat": (-0.499, 0.505, -0.500, 0.497), "fov": 120},
            {"id": "cross_right", "trans": (2.478, -0.955, 0.929), "quat": (0.141, -0.684, 0.701, -0.143), "fov": 120},
            {"id": "front_tele", "trans": (1.667, 0.074, 1.444), "quat": (0.495, -0.500, 0.507, -0.498), "fov": 30}
        ]
        self.camera_order = [cfg["id"] for cfg in self.CAMERA_CONFIGS]

        self.history_len = 16
        self.future_len = 64
        self.online_temporal_frames = 4
        self.buffer = deque(maxlen=self.history_len + self.future_len)

    def setup(self):
        # 0. 10Hz 동기 모드
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        
        # 1. Hero Vehicle 스폰
        bp = self.bp_lib.find('vehicle.tesla.model3')
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.hero_actor = self.world.spawn_actor(bp, spawn_point)
        
        # # Release parking brake and brake explicitly
        # self.hero_actor.apply_control(carla.VehicleControl(hand_brake=False, brake=0.0))
        
        # 2. 카메라 셋팅 (NVIDIA 보정 로직 적용)
        for config in self.CAMERA_CONFIGS:
            cam_bp = self.bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', '1920')
            cam_bp.set_attribute('image_size_y', '1080')
            cam_bp.set_attribute('fov', str(config['fov']))
            
            # NVIDIA 스타일 보정 Transform 생성
            tf = get_corrected_transform(config['trans'], config['quat'])
            
            cam_actor = self.world.spawn_actor(cam_bp, tf, attach_to=self.hero_actor)
            
            q = queue.Queue()
            cam_actor.listen(q.put)
            self.sensors.append(cam_actor)
            self.sensor_queues[config['id']] = q

        # 3. 모든 신호등 녹색으로 변경 및 고정
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # 4. Warmup & Start
        # Release parking brake and brake explicitly
        self.hero_actor.apply_control(carla.VehicleControl(hand_brake=False, brake=0.0))
        
        # Run simulation for a few ticks to ensure physics settle BEFORE autopilot
        for _ in range(10):
            self.world.tick()
        
        # Setup Traffic Manager for autopilot (if available)
        try:
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            traffic_manager.set_synchronous_mode(True)
            # Set autopilot with traffic manager
            self.hero_actor.set_autopilot(True, traffic_manager.get_port())
            print(">> Autopilot enabled with Traffic Manager")
        except Exception as e:
            # Fallback to basic autopilot if traffic manager not available
            self.hero_actor.set_autopilot(True)
            print(f">> Autopilot enabled (Traffic Manager not available: {e})")
        
        # Give autopilot a few ticks to start moving
        for _ in range(10):
            self.world.tick()

    def _capture_current_frame(self):
        self.world.tick()
        
        # Update Spectator to BEV
        spectator = self.world.get_spectator()
        ego_tf = self.hero_actor.get_transform()
        spectator.set_transform(carla.Transform(ego_tf.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        
        # Raw 데이터 수집
        images = []
        for cid in self.camera_order:
            raw = self.sensor_queues[cid].get()
            # BGRA -> RGB 변환 (NVIDIA 모델은 RGB를 기대함)
            img = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape((1080, 1920, 4))
            img = img[:, :, :3][:, :, ::-1] # BGR to RGB
            images.append(np.transpose(img, (2, 0, 1))) # C, H, W
        
        # print(self.hero_actor.get_transform())

        # self.buffer.append({
        #     'images': np.stack(images),
        #     'ego_matrix': get_matrix(self.hero_actor.get_transform()),
        #     'timestamp': int(self.world.get_snapshot().timestamp.elapsed_seconds * 1_000_000)
        # })
        
        # 1. 현재 트랜스폼 정보를 가져옴
        current_transform = self.hero_actor.get_transform()

        # 2. 요청하신 좌표 및 회전값 반전 (y = -y, yaw = -yaw)
        current_transform.location.y *= -1
        current_transform.rotation.yaw *= -1

        # 3. 수정된 트랜스폼을 사용하여 버퍼에 저장
        frame = {
            'images': np.stack(images),
            'ego_matrix': get_matrix(current_transform),
            'timestamp': int(self.world.get_snapshot().timestamp.elapsed_seconds * 1_000_000)
        }
        self.buffer.append(frame)
        return frame

    def _build_sample(self, t0_idx, history_indices, image_frame_indices, future_indices=None):
        t0_data = self.buffer[t0_idx]
        t0_inv = np.linalg.inv(t0_data['ego_matrix'])

        img_frames = np.stack([self.buffer[i]['images'] for i in image_frame_indices])
        img_frames = img_frames.transpose(1, 0, 2, 3, 4)  # [T, Cam, ...] -> [Cam, T, ...]

        h_xyz, h_rot = self._calc_rel_poses(history_indices, t0_inv)

        subset_ts = np.array([self.buffer[i]['timestamp'] for i in image_frame_indices], dtype=np.int64)
        abs_ts = np.repeat(subset_ts[:, None], len(self.camera_order), axis=1)
        abs_ts = abs_ts.T
        rel_ts = (abs_ts - t0_data['timestamp']) / 1e6

        sample = {
            'image_frames': img_frames,  # [Cam, T, C, H, W]
            'camera_indices': self.camera_indices,
            'camera_order': list(self.camera_order),
            'ego_history_xyz': h_xyz[None, None, ...],
            'ego_history_rot': h_rot[None, None, ...],
            'relative_timestamps': rel_ts,
            'absolute_timestamps': abs_ts,
            't0_us': t0_data['timestamp'],
            'fixed_delta_seconds': self.fixed_delta_seconds,
            'clip_id': self.clip_id,
        }

        if future_indices is not None:
            f_xyz, f_rot = self._calc_rel_poses(future_indices, t0_inv)
            sample['ego_future_xyz'] = f_xyz[None, None, ...]
            sample['ego_future_rot'] = f_rot[None, None, ...]

        return sample

    def fetch_dataset_sample(self):
        self._capture_current_frame()

        if len(self.buffer) < (self.history_len + self.future_len):
            return None

        t0_idx = self.history_len - 1
        image_frame_indices = range(t0_idx - (self.online_temporal_frames - 1), t0_idx + 1)
        return self._build_sample(
            t0_idx=t0_idx,
            history_indices=range(0, self.history_len),
            image_frame_indices=image_frame_indices,
            future_indices=range(t0_idx + 1, len(self.buffer)),
        )

    def fetch_online_sample(self):
        self._capture_current_frame()

        min_required = max(self.history_len, self.online_temporal_frames)
        if len(self.buffer) < min_required:
            return None

        t0_idx = len(self.buffer) - 1
        history_start = t0_idx - self.history_len + 1
        image_start = t0_idx - self.online_temporal_frames + 1
        return self._build_sample(
            t0_idx=t0_idx,
            history_indices=range(history_start, t0_idx + 1),
            image_frame_indices=range(image_start, t0_idx + 1),
            future_indices=None,
        )

    def fetch(self, mode="dataset"):
        if mode == "online":
            return self.fetch_online_sample()
        return self.fetch_dataset_sample()

    def _calc_rel_poses(self, indices, t0_inv):
        xyzs, rots = [], []
        for i in indices:
            rel = np.dot(t0_inv, self.buffer[i]['ego_matrix'])
            xyzs.append(rel[:3, 3])
            rots.append(rel[:3, :3])
        return np.array(xyzs), np.array(rots)

    # def _calc_rel_poses_flipped(self, indices, t0_inv):
    #     # codes here
            
    #     return np.array(xyzs), np.array(rots)

    def cleanup(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        for s in self.sensors: s.destroy()
        if self.hero_actor: self.hero_actor.destroy()

# ==========================================
# 3. 시각화 함수 (Dashboard)
# ==========================================
def save_dashboard(data, output_dir="outputs"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Latest temporal slice across all cameras: [Cam, T, C, H, W] -> [Cam, C, H, W]
    imgs = data['image_frames'][:, -1]
    titles = ["Cross Left", "Front Wide", "Cross Right", "Front Tele"]

    fig = plt.figure(figsize=(22, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1행: 카메라 이미지들
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.transpose(imgs[i], (1, 2, 0))) 
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.axis('off')

    # 2행: Front Tele
    ax_ft = fig.add_subplot(gs[1, 1])
    ax_ft.imshow(np.transpose(imgs[3], (1, 2, 0)))
    ax_ft.set_title(titles[3], fontsize=14, fontweight='bold')
    ax_ft.axis('off')

    # 2행 3열: Trajectory Plot (History + Future + Arrows)
    ax_plot = fig.add_subplot(gs[1, 2])
    # Squeeze batch dimensions [1, 1, T, 3] -> [T, 3]
    hist_xyz = data['ego_history_xyz'][0, 0]
    hist_rot = data['ego_history_rot'][0, 0] # (N, 3, 3)
    fut_xyz = data['ego_future_xyz'][0, 0]
    fut_rot = data['ego_future_rot'][0, 0]
    
    # 1. 과거 궤적 선 그래프
    ax_plot.plot(hist_xyz[:, 0], hist_xyz[:, 1], 'b-', alpha=0.3, linewidth=1.5)
    
    # 2. ego_history_rot을 활용한 방향 화살표 (Quiver)
    # 회전 행렬의 첫 번째 열 [R00, R10]이 Local X(정면)의 월드 방향입니다.
    u = hist_rot[:, 0, 0] # Forward vector's X
    v = hist_rot[:, 1, 0] # Forward vector's Y
    
    # Sample every 2nd point to avoid clutter, and scale arrows properly
    step = 2
    ax_plot.quiver(hist_xyz[::step, 0], hist_xyz[::step, 1], u[::step], v[::step], 
                   color='blue', scale=10, scale_units='xy', width=0.003, 
                   angles='xy', headwidth=3, headlength=4, headaxislength=3,
                   label='History Heading', alpha=0.7)

    # 3. 미래 경로 및 현재 위치
    combined_future = np.vstack([hist_xyz[-1:], fut_xyz])
    ax_plot.plot(combined_future[:, 0], combined_future[:, 1], 'r--o', 
                 label='Future (+6.4s)', markersize=4, linewidth=1.5)
    
    ax_plot.scatter(0, 0, c='black', marker='X', s=150, label='Current (t0)', zorder=10)
    
    # 4. 미래 궤적 방향 화살표 (Quiver)
    u = fut_rot[:, 0, 0] # Forward vector's X
    v = fut_rot[:, 1, 0] # Forward vector's Y
    
    # Sample every 4th point to avoid clutter
    step = 4
    ax_plot.quiver(fut_xyz[::step, 0], fut_xyz[::step, 1], u[::step], v[::step], 
                   color='red', scale=10, scale_units='xy', width=0.003,
                   angles='xy', headwidth=3, headlength=4, headaxislength=3,
                   label='Future Heading', alpha=0.7)
    
    # 그래프 스타일 설정
    ax_plot.set_aspect('equal')
    ax_plot.set_xlabel("Local X (Forward) [m]")
    ax_plot.set_ylabel("Local Y (Right) [m]")
    ax_plot.legend(loc='upper left')
    ax_plot.grid(True, alpha=0.4, linestyle=':')
    ax_plot.set_title(f"Ego Trajectory & Heading (t0: {data['t0_us']})", fontsize=14, fontweight='bold')

    save_path = f"{output_dir}/inf_{data['t0_us']}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()

def save_camera_images(data, output_dir="outputs/cameras"):
    """Save individual camera images for each scene"""
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    # Latest temporal slice across all cameras: [Cam, T, C, H, W] -> [Cam, C, H, W]
    imgs = data['image_frames'][:, -1]
    camera_names = ["cross_left", "front_wide", "cross_right", "front_tele"]
    
    t0_us = data['t0_us']
    
    # Save each camera image separately
    for i, (img, cam_name) in enumerate(zip(imgs, camera_names)):
        # Convert from [C, H, W] to [H, W, C] for display
        img_display = np.transpose(img, (1, 2, 0))
        
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.imshow(img_display)
        ax.set_title(f"{cam_name.replace('_', ' ').title()} - t0: {t0_us}", 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        save_path = f"{output_dir}/{cam_name}_{t0_us}.png"
        plt.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"  Saved 4 camera images to {output_dir}/")

def save_trajectory_plot(data, output_dir="outputs/trajectories"):
    """Save trajectory plot with history and future including heading arrows"""
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    # Squeeze batch dimensions [1, 1, T, 3] -> [T, 3]
    hist_xyz = data['ego_history_xyz'][0, 0]
    hist_rot = data['ego_history_rot'][0, 0] # (N, 3, 3)
    fut_xyz = data['ego_future_xyz'][0, 0]
    fut_rot = data['ego_future_rot'][0, 0]
    
    t0_us = data['t0_us']
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot historical trajectory line
    ax.plot(hist_xyz[:, 0], hist_xyz[:, 1], 'b-', alpha=0.5, linewidth=2, label='History Path')
    
    # 2. Historical trajectory heading arrows
    u_hist = hist_rot[:, 0, 0] # Forward vector's X
    v_hist = hist_rot[:, 1, 0] # Forward vector's Y
    
    # Sample every 2nd point to avoid clutter
    step_hist = 2
    ax.quiver(hist_xyz[::step_hist, 0], hist_xyz[::step_hist, 1], 
              u_hist[::step_hist], v_hist[::step_hist], 
              color='blue', scale=10, scale_units='xy', width=0.004, 
              angles='xy', headwidth=4, headlength=5, headaxislength=4,
              label='History Heading', alpha=0.8)
    
    # 3. Plot future trajectory line
    combined_future = np.vstack([hist_xyz[-1:], fut_xyz])
    ax.plot(combined_future[:, 0], combined_future[:, 1], 'r--', 
            linewidth=2, label='Future Path', alpha=0.7)
    
    # 4. Future trajectory heading arrows
    u_fut = fut_rot[:, 0, 0] # Forward vector's X
    v_fut = fut_rot[:, 1, 0] # Forward vector's Y
    
    # Sample every 4th point to avoid clutter
    step_fut = 4
    ax.quiver(fut_xyz[::step_fut, 0], fut_xyz[::step_fut, 1], 
              u_fut[::step_fut], v_fut[::step_fut], 
              color='red', scale=10, scale_units='xy', width=0.004,
              angles='xy', headwidth=4, headlength=5, headaxislength=4,
              label='Future Heading', alpha=0.8)
    
    # 5. Mark current position (t0)
    ax.scatter(0, 0, c='black', marker='X', s=200, label='Current (t0)', zorder=10, linewidths=2)
    
    # 6. Mark start of history
    ax.scatter(hist_xyz[0, 0], hist_xyz[0, 1], c='blue', marker='o', s=100, 
               label='History Start', zorder=9, edgecolors='darkblue', linewidths=2)
    
    # 7. Mark end of future
    ax.scatter(fut_xyz[-1, 0], fut_xyz[-1, 1], c='red', marker='s', s=100, 
               label='Future End', zorder=9, edgecolors='darkred', linewidths=2)
    
    # Graph styling
    ax.set_aspect('equal')
    ax.set_xlabel("Local X (Forward) [m]", fontsize=12)
    ax.set_ylabel("Local Y (Right) [m]", fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_title(f"Ego Trajectory & Heading\n(t0: {t0_us})", fontsize=14, fontweight='bold')
    
    save_path = f"{output_dir}/trajectory_{t0_us}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"  Saved trajectory plot to {save_path}")

def save_dashboard_inference_result(data, pred_xyz, pred_rot, extra, output_dir="outputs_inference"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Latest temporal slice across all cameras: [Cam, T, C, H, W] -> [Cam, C, H, W]
    imgs = data['image_frames'][:, -1]
    titles = ["Cross Left", "Front Wide", "Cross Right", "Front Tele"]

    fig = plt.figure(figsize=(22, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1행: 카메라 이미지들
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.transpose(imgs[i], (1, 2, 0))) 
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.axis('off')

    # 2행: Front Tele
    ax_ft = fig.add_subplot(gs[1, 1])
    ax_ft.imshow(np.transpose(imgs[3], (1, 2, 0)))
    ax_ft.set_title(titles[3], fontsize=14, fontweight='bold')
    ax_ft.axis('off')

    # 2행 1열: Chain of Causation (CoT)
    if extra and "cot" in extra:
        ax_cot = fig.add_subplot(gs[1, 0])
        ax_cot.axis('off')
        # extra["cot"] is typically [Batch, Set, Sample] strings. We assume Batch 0, Set 0.
        try:
            # Check structure of extra["cot"] + vehicle's yaw only
            cot_text = extra["cot"][0] + "\n" + "yaw: " + str(data['ego_history_rot'][0, 0, -1, 0, 2])
            # Wrap text for better visibility
            import textwrap
            wrapped_text = textwrap.fill(str(cot_text), width=45)
            ax_cot.text(0.0, 1.0, f"CoT:\n{wrapped_text}", fontsize=11, verticalalignment='top', family='monospace')
        except Exception as e:
            print(f"Error displaying CoT: {e}")

    # 2행 3열: Trajectory Plot (History + Future + Arrows)
    # set x, y range from -25 to + 25
    ax_plot = fig.add_subplot(gs[1, 2])
    ax_plot.set_xlim(-25, 25)
    ax_plot.set_ylim(-25, 25)
    # Squeeze batch dimensions [1, 1, T, 3] -> [T, 3]
    hist_xyz = data['ego_history_xyz'][0, 0]
    hist_rot = data['ego_history_rot'][0, 0] 
    fut_xyz = data['ego_future_xyz'][0, 0]
    fut_rot = data['ego_future_rot'][0, 0]
    
    # Prediction processing
    # pred_xyz: [B, Sets, Samples, Time, 3] -> [1, 1, 1, 64, 3] usually
    # We take the first sample: [Time, 3]
    if torch is not None and isinstance(pred_xyz, torch.Tensor):
        pred_xyz_np = pred_xyz.detach().cpu().float().numpy()[0, 0, 0]
    else:
        pred_xyz_np = pred_xyz[0, 0, 0]
        
    # 1. 과거 궤적 선 그래프
    ax_plot.plot(hist_xyz[:, 0], hist_xyz[:, 1], 'b-', alpha=0.3, linewidth=0.5, label='History')
    
    # 2. ego_history_rot을 활용한 방향 화살표 (Quiver)
    u = hist_rot[:, 0, 0] 
    v = hist_rot[:, 1, 0] 
    
    ax_plot.quiver(hist_xyz[:, 0], hist_xyz[:, 1], u, v, 
                   color='blue', scale=50, width=0.005)

    # 3. Ground Truth 미래 경로
    u_gt = fut_rot[:, 0, 0]
    v_gt = fut_rot[:, 1, 0]
    
    combined_future = np.vstack([hist_xyz[-1:], fut_xyz])
    ax_plot.plot(combined_future[:, 0], combined_future[:, 1], 'r--o', 
                 label='GT Future', markersize=4, linewidth=0.5)
    
    # GT Heading Arrows (Sparse: every 5th or similar if needed, but here all)
    # Using step=4 to avoid clutter
    ax_plot.quiver(fut_xyz[::4, 0], fut_xyz[::4, 1], u_gt[::4], v_gt[::4],
                   color='red', scale=50, width=0.004, alpha=0.5)
    
    # 4. Predicted Future Trajectory
    combined_pred = np.vstack([hist_xyz[-1:], pred_xyz_np])
    ax_plot.plot(combined_pred[:, 0], combined_pred[:, 1], 'g-x',
                 label='Pred Future', markersize=4, linewidth=0.7)
                 
    # Pred Heading Arrows
    if torch is not None and isinstance(pred_rot, torch.Tensor):
        pred_rot_np = pred_rot.detach().cpu().float().numpy()[0, 0, 0] # [Time, 3, 3]
    else:
        pred_rot_np = pred_rot[0, 0, 0]
        
    u_pred = pred_rot_np[:, 0, 0]
    v_pred = pred_rot_np[:, 1, 0]
    
    # Pred Heading Arrows (Sparse: every 4th)
    ax_plot.quiver(pred_xyz_np[::4, 0], pred_xyz_np[::4, 1], u_pred[::4], v_pred[::4],
                   color='green', scale=50, width=0.005, alpha=0.6)
    
    ax_plot.scatter(0, 0, c='black', marker='X', s=50, label='Current (t0)', zorder=10)
    
    # 그래프 스타일 설정
    # ax_plot.set_aspect('equal')
    ax_plot.set_xlabel("Local X (Forward) [m]")
    ax_plot.set_ylabel("Local Y (Right) [m]")
    ax_plot.legend(loc='upper left')
    ax_plot.grid(True, alpha=0.4, linestyle=':')
    ax_plot.set_title(f"Ego Trajectory (t0: {data['t0_us']})", fontsize=14, fontweight='bold')
    
    # Add CoT text if available
    if extra and "cot" in extra:
        cot_text = extra["cot"][0] # Batch 0, Set 0, Sample 0? 
        # structure is [Batch, Set, Sample] strings
        # Adjust layout to fit text? Or just print it on console is enough as requested. 
        # User only asked for plotting pred_xyz, pred_rot.
        pass

    # 저장
    save_path = f"{output_dir}/inf_{data['t0_us']}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()

    # Plot debug images (4x4 grid) using mediapy logic equivalent
    if 'image_frames' in data:
        # data["image_frames"] (4, 4, 3, H, W) -> flatten -> (16, 3, H, W) -> permute -> (16, H, W, 3)
        images = data["image_frames"].reshape(-1, *data["image_frames"].shape[2:]).transpose(0, 2, 3, 1)
        
        # Resize for "width=200" equivalent
        # Original H,W = 1080, 1920. Width=200 means scale factor ~0.104
        # mp.resize_image(img, (new_h, new_w))
        target_width = 200
        H, W, _ = images[0].shape
        scale = target_width / W
        target_height = int(H * scale)
        
        resized_images = [mp.resize_image(img, (target_height, target_width)) for img in images]
        
        # Create 4x4 grid
        # methods: mp.show_images is for display. For saving, we construct the grid manually.
        # columns=4.
        rows = []
        for i in range(0, len(resized_images), 4):
            batch = resized_images[i:i+4]
            # Pad if less than 4 (though here we exactly have 16 images)
            if len(batch) < 4:
                # Pad with black
                padding = [np.zeros_like(batch[0])] * (4 - len(batch))
                batch.extend(padding)
            rows.append(np.hstack(batch))
            
        grid_image = np.vstack(rows)
        
        debug_save_path = f"{output_dir}/img_debug_{data['t0_us']}.png"
        mp.write_image(debug_save_path, grid_image)


def export_predicted_trajectory_for_carla(data, pred_xyz, pred_rot, output_dir="traj_outputs"):
    """
    Convert Alpamayo's predicted future trajectory into a CARLA‑usable, time‑synced
    format.

    This exports a JSON file containing, for each future step:
      - absolute simulation time (seconds, CARLA world clock)
      - local (t0‑centric) xyz position in meters
      - local rotation as 3x3 rotation matrix

    The positions are expressed in the same t0‑local frame that Alpamayo predicts in.
    At runtime, you can transform these points back into CARLA world coordinates
    using the ego pose at t0 if you want to drive a vehicle or spawn a ghost actor.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fixed time step – comes from CARLA's synchronous settings (10 Hz by default).
    dt = float(data.get("fixed_delta_seconds", 0.1))
    t0 = float(data["t0_us"]) / 1e6

    # pred_xyz: [B, Sets, Samples, Time, 3] – we use the first sample.
    if torch is not None and isinstance(pred_xyz, torch.Tensor):
        pred_xyz_np = pred_xyz.detach().cpu().float().numpy()[0, 0, 0]  # [Time, 3]
    else:
        pred_xyz_np = pred_xyz[0, 0, 0]

    # pred_rot: [B, Sets, Samples, Time, 3, 3]
    if torch is not None and isinstance(pred_rot, torch.Tensor):
        pred_rot_np = pred_rot.detach().cpu().float().numpy()[0, 0, 0]  # [Time, 3, 3]
    else:
        pred_rot_np = pred_rot[0, 0, 0]

    num_steps = pred_xyz_np.shape[0]
    times = t0 + dt * (np.arange(num_steps) + 1)

    trajectory = []
    for i in range(num_steps):
        step = {
            "t_sec": float(times[i]),
            "xyz_local": pred_xyz_np[i].tolist(),
            "rot_local_3x3": pred_rot_np[i].tolist(),
        }
        trajectory.append(step)

    out_path = os.path.join(output_dir, f"traj_{data['clip_id']}_{data['t0_us']}.json")
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_id": data["clip_id"],
                "t0_us": int(data["t0_us"]),
                "fixed_delta_seconds": dt,
                "steps": trajectory,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f">> Exported CARLA‑ready predicted trajectory to: {out_path}")


def run_dataset_mode(grabber, max_frames):
    print(">> Starting simulation data collection with CARLA autopilot...")
    print(">> Autopilot is enabled. Vehicle will drive autonomously.")

    count = 0
    while count < max_frames:
        data = grabber.fetch(mode="dataset")
        if data:
            velocity = grabber.hero_actor.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            transform = grabber.hero_actor.get_transform()

            print(f">> [{count}] Processing scene t0: {data['t0_us']}")
            save_camera_images(data)
            save_trajectory_plot(data)
            save_dashboard(data)

            if count % 10 == 0:
                print(
                    f"    Vehicle speed: {speed:.2f} km/h, "
                    f"Location: ({transform.location.x:.2f}, {transform.location.y:.2f})"
                )
            count += 1


def run_server_mode(
    grabber,
    server_host,
    server_port,
    max_frames,
    target_width,
    target_height,
    log_online_history_flag,
    publish_interval_sec,
    queue_max_size,
):
    sample_store = SampleStore(queue_max_size=queue_max_size)
    handler = make_sample_http_handler(sample_store, target_width, target_height)
    server = ThreadingHTTPServer((server_host, server_port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f">> Latest-sample server listening on http://{server_host}:{server_port}")
    print(f">> Jetson can pull /latest for newest sample, /next for queued samples, and /health for status")
    print(f">> Resize target before send: {target_width}x{target_height}")
    print(f">> Queue publish interval: {publish_interval_sec:.2f}s, queue_max_size={queue_max_size}")

    count = 0
    publish_interval_us = int(publish_interval_sec * 1_000_000)
    last_queued_t0_us = None
    try:
        while max_frames < 0 or count < max_frames:
            data = grabber.fetch(mode="online")
            if data is None:
                continue

            if log_online_history_flag:
                log_online_history(data)

            latest_sequence = sample_store.update_latest(data)
            count += 1

            should_enqueue = False
            if last_queued_t0_us is None:
                should_enqueue = True
            else:
                should_enqueue = int(data["t0_us"]) - last_queued_t0_us >= publish_interval_us

            if should_enqueue:
                queued_sequence, queue_size = sample_store.enqueue(data)
                last_queued_t0_us = int(data["t0_us"])
                print(
                    f">> queued sample: queue_sequence={queued_sequence}, latest_sequence={latest_sequence}, "
                    f"t0_us={data['t0_us']}, queue_size={queue_size}"
                )

            if count == 1 or count % 10 == 0:
                velocity = grabber.hero_actor.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                print(
                    f">> [{count}] latest sample ready: "
                    f"t0_us={data['t0_us']}, speed={speed:.2f} km/h, "
                    f"image_frames={data['image_frames'].shape}, "
                    f"ego_history_xyz={data['ego_history_xyz'].shape}"
                )
    finally:
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    # ==========================================
    # 4. Command Line Arguments
    # ==========================================
    parser = argparse.ArgumentParser(description='CARLA Simulation with Autopilot')
    parser.add_argument('--mode', type=str, default='dataset', choices=['dataset', 'server'],
                        help='dataset: save local samples, server: expose latest sample over HTTP')
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA server host address (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to collect. Default: 100 in dataset mode, unlimited in server mode.')
    parser.add_argument('--server_host', type=str, default='0.0.0.0',
                        help='Bind address for the HTTP latest-sample server')
    parser.add_argument('--server_port', type=int, default=8765,
                        help='Port for the HTTP latest-sample server')
    parser.add_argument('--publish_interval_sec', type=float, default=5.0,
                        help='Queue publish interval in simulation seconds for /next samples')
    parser.add_argument('--queue_max_size', type=int, default=32,
                        help='Maximum number of queued samples kept for /next consumption')
    parser.add_argument('--send_width', type=int, default=DEFAULT_SEND_WIDTH,
                        help='Width of images sent to Jetson')
    parser.add_argument('--send_height', type=int, default=DEFAULT_SEND_HEIGHT,
                        help='Height of images sent to Jetson')
    parser.add_argument('--log_online_history', action='store_true',
                        help='Print ego_history_xyz/rot summary before publishing each online sample')
    args = parser.parse_args()

    if args.max_frames is None:
        args.max_frames = -1 if args.mode == 'server' else 100

    print(f">> Connecting to CARLA server at {args.host}:{args.port}...")
    grabber = SimDataGrabber(host=args.host, port=args.port)
    try:
        grabber.setup()
        if args.mode == 'server':
            run_server_mode(
                grabber,
                server_host=args.server_host,
                server_port=args.server_port,
                max_frames=args.max_frames,
                target_width=args.send_width,
                target_height=args.send_height,
                log_online_history_flag=args.log_online_history,
                publish_interval_sec=args.publish_interval_sec,
                queue_max_size=args.queue_max_size,
            )
        else:
            run_dataset_mode(grabber, args.max_frames)

    except KeyboardInterrupt:
        print(">> Interrupted by user")
    except Exception as e:
        print(f">> Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        grabber.cleanup()
        print(">> Cleanup completed.")
