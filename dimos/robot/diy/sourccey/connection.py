from __future__ import annotations

import base64
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable
import zmq

from dimos.agents.annotation import skill
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.spec.perception import Camera, IMU
from dimos.utils.logging_config import setup_logger

from .protobuf.generated import sourccey_pb2

logger = setup_logger()

_SLAM_INPUT_SCHEMA = "slam_input.v1"
_SLAM_OUTPUT_SCHEMA = "slam_output.v1"
_OPTICAL_ROTATION = Quaternion(-0.5, 0.5, -0.5, 0.5)
_ARM_JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]
_ALL_JOINT_NAMES = [*_ARM_JOINT_NAMES, "z.pos"]


@dataclass(slots=True)
class _CameraPacket:
    name: str
    frame_id: int
    capture_monotonic_ns: int
    image: np.ndarray


@dataclass(slots=True)
class _ImuPacket:
    capture_monotonic_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float | None = None
    my: float | None = None
    mz: float | None = None


@dataclass(slots=True)
class _SlamInputPacket:
    source: str
    host_monotonic_ns: int
    base_velocity: dict[str, float]
    cameras: dict[str, _CameraPacket]
    imu_samples: list[_ImuPacket]


@dataclass(slots=True)
class _SlamOutputPacket:
    world_x: float
    world_y: float
    world_z: float
    world_yaw_rad: float | None
    status: str
    detail: str


@dataclass(slots=True)
class _RobotObservationState:
    left_arm: tuple[float, float, float, float, float, float]
    right_arm: tuple[float, float, float, float, float, float]
    z_pos: float
    x_vel: float
    y_vel: float
    theta_vel: float

    @classmethod
    def zero(cls) -> "_RobotObservationState":
        zeros = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            left_arm=zeros,
            right_arm=zeros,
            z_pos=0.0,
            x_vel=0.0,
            y_vel=0.0,
            theta_vel=0.0,
        )


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode Sourccey JPEG frame")
    return frame


def _parse_slam_input_packet(payload: bytes) -> _SlamInputPacket:
    data = json.loads(payload.decode("utf-8"))
    if data.get("schema") != _SLAM_INPUT_SCHEMA:
        raise ValueError(f"Unsupported Sourccey input schema: {data.get('schema')}")

    cameras_payload = data.get("cameras", {})
    decoded_cameras: dict[str, _CameraPacket] = {}
    for camera_name, raw_camera in cameras_payload.items():
        if not isinstance(raw_camera, dict):
            continue
        jpeg_payload = raw_camera.get("jpeg_b64")
        if jpeg_payload is None:
            continue
        decoded_cameras[str(camera_name)] = _CameraPacket(
            name=str(camera_name),
            frame_id=int(raw_camera.get("frame_id", 0)),
            capture_monotonic_ns=int(raw_camera.get("capture_monotonic_ns", 0)),
            image=_decode_jpeg(base64.b64decode(jpeg_payload)),
        )

    base_velocity_payload = data.get("base_velocity", {})
    base_velocity = {
        "x.vel": float(base_velocity_payload.get("x.vel", 0.0)),
        "y.vel": float(base_velocity_payload.get("y.vel", 0.0)),
        "theta.vel": float(base_velocity_payload.get("theta.vel", 0.0)),
    }

    raw_imu_samples = data.get("imu_samples")
    if raw_imu_samples is None and isinstance(data.get("imu"), dict):
        raw_imu_samples = data["imu"].get("samples", [])

    imu_samples: list[_ImuPacket] = []
    if isinstance(raw_imu_samples, list):
        for sample in raw_imu_samples:
            if not isinstance(sample, dict) or "capture_monotonic_ns" not in sample:
                continue
            imu_samples.append(
                _ImuPacket(
                    capture_monotonic_ns=int(sample["capture_monotonic_ns"]),
                    ax=float(sample.get("ax", 0.0)),
                    ay=float(sample.get("ay", 0.0)),
                    az=float(sample.get("az", 0.0)),
                    gx=float(sample.get("gx", 0.0)),
                    gy=float(sample.get("gy", 0.0)),
                    gz=float(sample.get("gz", 0.0)),
                    mx=None if sample.get("mx") is None else float(sample.get("mx", 0.0)),
                    my=None if sample.get("my") is None else float(sample.get("my", 0.0)),
                    mz=None if sample.get("mz") is None else float(sample.get("mz", 0.0)),
                )
            )

    return _SlamInputPacket(
        source=str(data.get("source", "unknown")),
        host_monotonic_ns=int(data.get("host_monotonic_ns", 0)),
        base_velocity=base_velocity,
        cameras=decoded_cameras,
        imu_samples=imu_samples,
    )


def _parse_slam_output_packet(payload: bytes) -> _SlamOutputPacket:
    data = json.loads(payload.decode("utf-8"))
    if data.get("schema") != _SLAM_OUTPUT_SCHEMA:
        raise ValueError(f"Unsupported Sourccey output schema: {data.get('schema')}")

    pose = data.get("pose", {})
    yaw_raw = pose.get("yaw_rad", pose.get("yaw", pose.get("theta")))
    return _SlamOutputPacket(
        world_x=float(pose.get("x", 0.0)),
        world_y=float(pose.get("y", 0.0)),
        world_z=float(pose.get("z", 0.0)),
        world_yaw_rad=None if yaw_raw is None else _wrap_angle_rad(float(yaw_raw)),
        status=str(data.get("health", {}).get("status", "no_data")),
        detail=str(data.get("health", {}).get("detail", "")),
    )


def _quaternion_from_rpy_deg(rpy_deg: tuple[float, float, float]) -> Quaternion:
    roll_deg, pitch_deg, yaw_deg = rpy_deg
    return Quaternion.from_euler(
        Vector3(
            math.radians(float(roll_deg)),
            math.radians(float(pitch_deg)),
            math.radians(float(yaw_deg)),
        )
    )


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _blend_angle_rad(
    current_yaw_rad: float,
    target_yaw_rad: float,
    *,
    alpha: float,
    max_step_rad: float,
) -> float:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    if alpha_clamped <= 0.0:
        return _wrap_angle_rad(current_yaw_rad)
    delta = _wrap_angle_rad(float(target_yaw_rad) - float(current_yaw_rad))
    step = float(np.clip(delta * alpha_clamped, -abs(max_step_rad), abs(max_step_rad)))
    return _wrap_angle_rad(float(current_yaw_rad) + step)


def _magnetometer_yaw_rad(sample: _ImuPacket) -> float | None:
    if sample.mx is None or sample.my is None:
        return None
    mx = float(sample.mx)
    my = float(sample.my)
    if not math.isfinite(mx) or not math.isfinite(my):
        return None
    if math.hypot(mx, my) < 1e-6:
        return None
    return math.atan2(my, mx)


def _parse_robot_observation(payload: bytes) -> _RobotObservationState:
    robot_state = sourccey_pb2.SourcceyRobotState()
    robot_state.ParseFromString(payload)

    left = robot_state.left_arm_joints
    right = robot_state.right_arm_joints
    base_position = robot_state.base_position
    base_velocity = robot_state.base_velocity

    return _RobotObservationState(
        left_arm=(
            float(left.shoulder_pan),
            float(left.shoulder_lift),
            float(left.elbow_flex),
            float(left.wrist_flex),
            float(left.wrist_roll),
            float(left.gripper),
        ),
        right_arm=(
            float(right.shoulder_pan),
            float(right.shoulder_lift),
            float(right.elbow_flex),
            float(right.wrist_flex),
            float(right.wrist_roll),
            float(right.gripper),
        ),
        z_pos=float(base_position.z_pos),
        x_vel=float(base_velocity.x_vel),
        y_vel=float(base_velocity.y_vel),
        theta_vel=float(base_velocity.theta_vel),
    )


def _build_robot_action_packet(
    *,
    x_vel: float,
    y_vel: float,
    theta_vel: float,
    state: _RobotObservationState,
    untorque_left: bool = False,
    untorque_right: bool = False,
) -> bytes:
    msg = sourccey_pb2.SourcceyRobotAction()

    msg.left_arm_target_joints.shoulder_pan = float(state.left_arm[0])
    msg.left_arm_target_joints.shoulder_lift = float(state.left_arm[1])
    msg.left_arm_target_joints.elbow_flex = float(state.left_arm[2])
    msg.left_arm_target_joints.wrist_flex = float(state.left_arm[3])
    msg.left_arm_target_joints.wrist_roll = float(state.left_arm[4])
    msg.left_arm_target_joints.gripper = float(state.left_arm[5])

    msg.right_arm_target_joints.shoulder_pan = float(state.right_arm[0])
    msg.right_arm_target_joints.shoulder_lift = float(state.right_arm[1])
    msg.right_arm_target_joints.elbow_flex = float(state.right_arm[2])
    msg.right_arm_target_joints.wrist_flex = float(state.right_arm[3])
    msg.right_arm_target_joints.wrist_roll = float(state.right_arm[4])
    msg.right_arm_target_joints.gripper = float(state.right_arm[5])

    msg.base_target_position.z_pos = float(state.z_pos)
    msg.base_target_velocity.x_vel = float(x_vel)
    msg.base_target_velocity.y_vel = float(y_vel)
    msg.base_target_velocity.theta_vel = float(theta_vel)
    msg.untorque_left = bool(untorque_left)
    msg.untorque_right = bool(untorque_right)
    return msg.SerializeToString()


def _socket_connect_sub(context: zmq.Context, endpoint: str) -> zmq.Socket:
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.connect(endpoint)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    return sock


def _socket_connect_push(context: zmq.Context, endpoint: str) -> zmq.Socket:
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.connect(endpoint)
    return sock


def _socket_connect_pull(context: zmq.Context, endpoint: str) -> zmq.Socket:
    sock = context.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.connect(endpoint)
    return sock


def _safe_resize(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None or height is None or width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _build_mosaic(
    primary: np.ndarray,
    companion: np.ndarray | None,
    bottom: np.ndarray | None,
) -> np.ndarray:
    if companion is None and bottom is None:
        return primary

    base_h, base_w = primary.shape[:2]
    blank = np.zeros_like(primary)
    companion_img = companion if companion is not None else blank
    bottom_img = bottom if bottom is not None else blank

    companion_img = cv2.resize(companion_img, (base_w, base_h), interpolation=cv2.INTER_AREA)
    bottom_img = cv2.resize(bottom_img, (base_w, base_h), interpolation=cv2.INTER_AREA)

    top_row = np.hstack((primary, companion_img))
    bottom_row = np.hstack((bottom_img, blank))
    return np.vstack((top_row, bottom_row))


def _camera_info_from_frame(
    frame: np.ndarray,
    *,
    fov_deg: float,
    axis: str,
    frame_id: str,
) -> CameraInfo:
    height, width = frame.shape[:2]
    return CameraInfo.from_fov(
        fov_deg=fov_deg,
        width=width,
        height=height,
        axis=axis,
        frame_id=frame_id,
    )


class SourcceyConnectionConfig(ModuleConfig):
    slam_input_endpoint: str = Field(
        default_factory=lambda m: f"tcp://{m['g'].robot_ip or '127.0.0.1'}:5560"
    )
    slam_output_endpoint: str | None = "tcp://127.0.0.1:5561"
    command_endpoint: str = Field(
        default_factory=lambda m: f"tcp://{m['g'].robot_ip or '127.0.0.1'}:5555"
    )
    observation_endpoint: str = Field(
        default_factory=lambda m: f"tcp://{m['g'].robot_ip or '127.0.0.1'}:5556"
    )
    poll_timeout_ms: int = 100
    robot_state_ready_timeout_s: float = 1.5
    cmd_vel_timeout_s: float = 0.35
    max_linear_speed_m_s: float = 1.0
    max_strafe_speed_m_s: float = 1.0
    max_angular_speed_rad_s: float = 1.2
    allow_unsafe_base_control_without_state: bool = False
    primary_camera_key: str = "front_left"
    companion_camera_key: str = "front_right"
    bottom_camera_key: str = "bottom"
    publish_mosaic_as_color_image: bool = False
    resize_width: int | None = None
    resize_height: int | None = None
    camera_fov_deg: float = 78.0
    bottom_camera_fov_deg: float = 110.0
    camera_fov_axis: str = "horizontal"
    slam_output_stale_after_s: float = 0.75
    untorque_arms_during_base_control: bool = True
    packet_pose_fresh_window_s: float = 0.35
    imu_heading_correction_enabled: bool = True
    imu_heading_alpha: float = 0.18
    imu_heading_max_step_deg: float = 12.0
    slam_heading_alpha: float = 0.75
    slam_heading_max_step_deg: float = 20.0
    # Approximate multi-camera rig geometry for Sourccey.
    # These defaults are intentionally non-zero because the front cameras are not parallel,
    # and the bottom camera is pitched downward relative to the base.
    front_camera_xyz_m: tuple[float, float, float] = (0.08, 0.038, 0.16)
    front_camera_rpy_deg: tuple[float, float, float] = (0.0, -20.0, 15.0)
    companion_camera_xyz_m: tuple[float, float, float] = (0.08, -0.038, 0.16)
    companion_camera_rpy_deg: tuple[float, float, float] = (0.0, -20.0, -15.0)
    bottom_camera_xyz_m: tuple[float, float, float] = (0.04, 0.0, 0.05)
    bottom_camera_rpy_deg: tuple[float, float, float] = (0.0, -58.0, 0.0)


class SourcceyConnection(Module, Camera, IMU):
    dedicated_worker = True

    config: SourcceyConnectionConfig

    cmd_vel: In[Twist]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]
    companion_image: Out[Image]
    bottom_image: Out[Image]
    odom: Out[PoseStamped]
    imu: Out[Imu]
    joint_state: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context: zmq.Context | None = None
        self._input_socket: zmq.Socket | None = None
        self._output_socket: zmq.Socket | None = None
        self._command_socket: zmq.Socket | None = None
        self._observation_socket: zmq.Socket | None = None
        self._stop_event = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self._cmd_stop_timer: threading.Timer | None = None
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()

        self._dead_reckon_xy = np.zeros(2, dtype=np.float64)
        self._yaw_rad = 0.0
        self._last_packet_wall_ts: float | None = None
        self._last_slam_position_xy: tuple[float, float] | None = None
        self._last_slam_yaw_rad: float | None = None
        self._last_slam_wall_ts: float | None = None
        self._latest_robot_state: _RobotObservationState | None = None
        self._latest_color_image: Image | None = None
        self._last_observation_wall_ts: float | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._context = zmq.Context()
        self._input_socket = _socket_connect_sub(self._context, self.config.slam_input_endpoint)
        if self.config.slam_output_endpoint:
            self._output_socket = _socket_connect_sub(self._context, self.config.slam_output_endpoint)
        self._command_socket = _socket_connect_push(self._context, self.config.command_endpoint)
        self._observation_socket = _socket_connect_pull(self._context, self.config.observation_endpoint)

        self._prime_robot_state(timeout_s=self.config.robot_state_ready_timeout_s)

        self._stop_event.clear()
        self._pump_thread = threading.Thread(
            target=self._pump_loop,
            name="sourccey-dimos-pump",
            daemon=True,
        )
        self._pump_thread.start()
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        logger.info(
            "SourcceyConnection started",
            input_endpoint=self.config.slam_input_endpoint,
            output_endpoint=self.config.slam_output_endpoint,
            command_endpoint=self.config.command_endpoint,
            observation_endpoint=self.config.observation_endpoint,
            mosaic=self.config.publish_mosaic_as_color_image,
        )

    @rpc
    def stop(self) -> None:
        self._cancel_cmd_stop_timer()
        self._send_stop_command()
        self._stop_event.set()
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            self._pump_thread = None

        for sock_name in (
            "_input_socket",
            "_output_socket",
            "_command_socket",
            "_observation_socket",
        ):
            sock = getattr(self, sock_name)
            if sock is not None:
                sock.close(0)
                setattr(self, sock_name, None)
        if self._context is not None:
            self._context.term()
            self._context = None
        super().stop()

    def _on_cmd_vel(self, msg: Twist) -> None:
        logger.info(
            "SourcceyConnection received cmd_vel",
            linear_x=round(float(msg.linear.x), 4),
            linear_y=round(float(msg.linear.y), 4),
            angular_z=round(float(msg.angular.z), 4),
        )
        self.move(msg)


    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        vx = float(
            np.clip(
                twist.linear.x,
                -self.config.max_linear_speed_m_s,
                self.config.max_linear_speed_m_s,
            )
        )
        vy = float(
            np.clip(
                twist.linear.y,
                -self.config.max_strafe_speed_m_s,
                self.config.max_strafe_speed_m_s,
            )
        )
        wz = float(
            np.clip(
                twist.angular.z,
                -self.config.max_angular_speed_rad_s,
                self.config.max_angular_speed_rad_s,
            )
        )

        state = self._get_robot_state()
        if state is None and not self.config.allow_unsafe_base_control_without_state:
            logger.warning("Ignoring Sourccey cmd_vel until a real robot state packet has been received")
            return False
        if state is None:
            state = _RobotObservationState.zero()

        payload = _build_robot_action_packet(
            x_vel=vx,
            y_vel=vy,
            theta_vel=wz,
            state=state,
            untorque_left=bool(self.config.untorque_arms_during_base_control),
            untorque_right=bool(self.config.untorque_arms_during_base_control),
        )
        logger.info(
            "Sending Sourccey base command",
            x_vel=round(vx, 4),
            y_vel=round(vy, 4),
            theta_vel=round(wz, 4),
        )
        if not self._send_command_payload(payload):
            return False

        self._cancel_cmd_stop_timer()
        timeout = duration if duration > 0 else float(self.config.cmd_vel_timeout_s)
        if timeout > 0.0:
            self._cmd_stop_timer = threading.Timer(timeout, self._send_stop_command)
            self._cmd_stop_timer.daemon = True
            self._cmd_stop_timer.start()
        return True

    @rpc
    def get_robot_state(self) -> dict[str, float] | None:
        state = self._get_robot_state()
        if state is None:
            return None
        return {
            "left_shoulder_pan.pos": state.left_arm[0],
            "left_shoulder_lift.pos": state.left_arm[1],
            "left_elbow_flex.pos": state.left_arm[2],
            "left_wrist_flex.pos": state.left_arm[3],
            "left_wrist_roll.pos": state.left_arm[4],
            "left_gripper.pos": state.left_arm[5],
            "right_shoulder_pan.pos": state.right_arm[0],
            "right_shoulder_lift.pos": state.right_arm[1],
            "right_elbow_flex.pos": state.right_arm[2],
            "right_wrist_flex.pos": state.right_arm[3],
            "right_wrist_roll.pos": state.right_arm[4],
            "right_gripper.pos": state.right_arm[5],
            "z.pos": state.z_pos,
            "x.vel": state.x_vel,
            "y.vel": state.y_vel,
            "theta.vel": state.theta_vel,
        }

    @skill
    def observe(self) -> Image | None:
        return self._latest_color_image

    @skill
    def move_velocity(self, x: float, y: float = 0.0, yaw: float = 0.0, duration: float = 0.0) -> str:
        success = self.move(
            Twist(
                linear=Vector3(x, y, 0.0),
                angular=Vector3(0.0, 0.0, yaw),
            ),
            duration=duration,
        )
        return (
            f"Started Sourccey velocity command x={x:.3f} y={y:.3f} yaw={yaw:.3f} duration={duration:.3f}s"
            if success
            else "Sourccey velocity command was rejected because no safe robot state is available yet"
        )

    def _pump_loop(self) -> None:
        assert self._input_socket is not None
        poller = zmq.Poller()
        poller.register(self._input_socket, zmq.POLLIN)
        if self._output_socket is not None:
            poller.register(self._output_socket, zmq.POLLIN)
        if self._observation_socket is not None:
            poller.register(self._observation_socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                events = dict(poller.poll(self.config.poll_timeout_ms))
                if self._observation_socket is not None and self._observation_socket in events:
                    self._drain_observation_socket()
                if self._output_socket is not None and self._output_socket in events:
                    self._drain_output_socket()
                if self._input_socket in events:
                    self._drain_input_socket()
            except Exception as exc:
                logger.warning("SourcceyConnection pump error", error=str(exc))
                time.sleep(0.05)

    def _prime_robot_state(self, timeout_s: float) -> None:
        if self._observation_socket is None or timeout_s <= 0.0:
            return
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and not self._stop_event.is_set():
            remaining_ms = int(max(1.0, min(100.0, (deadline - time.monotonic()) * 1000.0)))
            if not self._observation_socket.poll(remaining_ms):
                continue
            try:
                latest_payload = self._observation_socket.recv()
                state = _parse_robot_observation(latest_payload)
            except Exception as exc:
                logger.debug("Ignoring invalid initial Sourccey observation", error=str(exc))
                continue
            self._set_robot_state(state)
            self._publish_joint_state(state)
            logger.info("SourcceyConnection received initial robot state")
            return
        logger.warning(
            "SourcceyConnection did not receive an initial robot state before timeout; cmd_vel will stay guarded until one arrives"
        )

    def _drain_input_socket(self) -> None:
        if self._input_socket is None:
            return
        latest_payload: bytes | None = None
        while self._input_socket.poll(0):
            latest_payload = self._input_socket.recv()
        if latest_payload is None:
            return
        packet = _parse_slam_input_packet(latest_payload)
        self._publish_packet(packet)

    def _drain_output_socket(self) -> None:
        if self._output_socket is None:
            return
        while self._output_socket.poll(0):
            try:
                slam_output = _parse_slam_output_packet(self._output_socket.recv())
            except Exception as exc:
                logger.debug("Ignoring invalid Sourccey SLAM output", error=str(exc))
                continue
            self._last_slam_position_xy = (
                float(slam_output.world_x),
                float(slam_output.world_z),
            )
            if slam_output.world_yaw_rad is not None:
                self._last_slam_yaw_rad = float(slam_output.world_yaw_rad)
            self._last_slam_wall_ts = time.time()

    def _drain_observation_socket(self) -> None:
        if self._observation_socket is None:
            return
        latest_payload: bytes | None = None
        while self._observation_socket.poll(0):
            latest_payload = self._observation_socket.recv()
        if latest_payload is None:
            return
        state = _parse_robot_observation(latest_payload)
        self._set_robot_state(state)
        self._publish_joint_state(state)
        self._publish_observation_pose(state)

    def _set_robot_state(self, state: _RobotObservationState) -> None:
        with self._state_lock:
            self._latest_robot_state = state

    def _get_robot_state(self) -> _RobotObservationState | None:
        with self._state_lock:
            return self._latest_robot_state

    def _publish_joint_state(self, state: _RobotObservationState) -> None:
        positions = [*state.left_arm, *state.right_arm, float(state.z_pos)]
        velocities = [0.0] * len(positions)
        self.joint_state.publish(
            JointState(
                ts=time.time(),
                frame_id="base_link",
                name=list(_ALL_JOINT_NAMES),
                position=positions,
                velocity=velocities,
                effort=[],
            )
        )

    def _send_command_payload(self, payload: bytes) -> bool:
        if self._command_socket is None:
            logger.warning("Sourccey command socket is not connected")
            return False
        try:
            with self._command_lock:
                self._command_socket.send(payload)
            return True
        except Exception as exc:
            logger.error("Failed to send Sourccey command", error=str(exc))
            return False

    def _cancel_cmd_stop_timer(self) -> None:
        if self._cmd_stop_timer is not None:
            self._cmd_stop_timer.cancel()
            self._cmd_stop_timer = None

    def _send_stop_command(self) -> None:
        state = self._get_robot_state()
        if state is None and not self.config.allow_unsafe_base_control_without_state:
            return
        if state is None:
            state = _RobotObservationState.zero()
        payload = _build_robot_action_packet(
            x_vel=0.0,
            y_vel=0.0,
            theta_vel=0.0,
            state=state,
            untorque_left=bool(self.config.untorque_arms_during_base_control),
            untorque_right=bool(self.config.untorque_arms_during_base_control),
        )
        self._send_command_payload(payload)

    def _apply_absolute_heading_corrections(
        self,
        *,
        wall_ts: float,
        imu_heading_rad: float | None = None,
    ) -> None:
        slam_heading_is_fresh = (
            self._last_slam_yaw_rad is not None
            and self._last_slam_wall_ts is not None
            and (wall_ts - self._last_slam_wall_ts) <= float(self.config.slam_output_stale_after_s)
        )
        if slam_heading_is_fresh:
            self._yaw_rad = _blend_angle_rad(
                self._yaw_rad,
                float(self._last_slam_yaw_rad),
                alpha=float(self.config.slam_heading_alpha),
                max_step_rad=math.radians(float(self.config.slam_heading_max_step_deg)),
            )
            return
        if not bool(self.config.imu_heading_correction_enabled) or imu_heading_rad is None:
            self._yaw_rad = _wrap_angle_rad(self._yaw_rad)
            return
        self._yaw_rad = _blend_angle_rad(
            self._yaw_rad,
            imu_heading_rad,
            alpha=float(self.config.imu_heading_alpha),
            max_step_rad=math.radians(float(self.config.imu_heading_max_step_deg)),
        )

    def _publish_observation_pose(self, state: _RobotObservationState) -> None:
        wall_ts = time.time()
        dt = 0.0
        if self._last_observation_wall_ts is not None:
            dt = max(0.0, min(wall_ts - self._last_observation_wall_ts, 0.25))
        self._last_observation_wall_ts = wall_ts

        packet_pose_is_fresh = (
            self._last_packet_wall_ts is not None
            and (wall_ts - self._last_packet_wall_ts) <= float(self.config.packet_pose_fresh_window_s)
        )
        if packet_pose_is_fresh:
            self.odom.publish(
                PoseStamped(
                    ts=wall_ts,
                    frame_id="world",
                    position=[float(self._dead_reckon_xy[0]), float(self._dead_reckon_xy[1]), 0.0],
                    orientation=Quaternion.from_euler(Vector3(0.0, 0.0, self._yaw_rad)),
                )
            )
            return

        theta_rate = float(state.theta_vel)
        self._yaw_rad += theta_rate * dt
        self._apply_absolute_heading_corrections(wall_ts=wall_ts)

        slam_is_fresh = (
            self._last_slam_position_xy is not None
            and self._last_slam_wall_ts is not None
            and (wall_ts - self._last_slam_wall_ts) <= float(self.config.slam_output_stale_after_s)
        )
        if slam_is_fresh:
            self._dead_reckon_xy = np.asarray(self._last_slam_position_xy, dtype=np.float64)
        else:
            vx = float(state.x_vel)
            vy = float(state.y_vel)
            cos_yaw = math.cos(self._yaw_rad)
            sin_yaw = math.sin(self._yaw_rad)
            dx = (vx * cos_yaw) - (vy * sin_yaw)
            dy = (vx * sin_yaw) + (vy * cos_yaw)
            self._dead_reckon_xy += np.asarray((dx * dt, dy * dt), dtype=np.float64)

        pose = PoseStamped(
            ts=wall_ts,
            frame_id="world",
            position=[float(self._dead_reckon_xy[0]), float(self._dead_reckon_xy[1]), 0.0],
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, self._yaw_rad)),
        )
        self.odom.publish(pose)

    def _publish_packet(self, packet: _SlamInputPacket) -> None:
        primary_frame = self._get_camera(packet, self.config.primary_camera_key)
        if primary_frame is None:
            return

        companion_frame = self._get_camera(packet, self.config.companion_camera_key)
        bottom_frame = self._get_camera(packet, self.config.bottom_camera_key)

        primary_image = _safe_resize(
            primary_frame.image,
            self.config.resize_width,
            self.config.resize_height,
        )
        companion_image = (
            _safe_resize(
                companion_frame.image,
                self.config.resize_width,
                self.config.resize_height,
            )
            if companion_frame is not None
            else None
        )
        bottom_image = (
            _safe_resize(
                bottom_frame.image,
                self.config.resize_width,
                self.config.resize_height,
            )
            if bottom_frame is not None
            else None
        )

        color_frame = (
            _build_mosaic(primary_image, companion_image, bottom_image)
            if self.config.publish_mosaic_as_color_image
            else primary_image
        )
        wall_ts = time.time()

        color_msg = Image.from_numpy(
            color_frame,
            format=ImageFormat.BGR,
            frame_id="camera_optical",
            ts=wall_ts,
        )
        self._latest_color_image = color_msg
        self.color_image.publish(color_msg)
        self.camera_info.publish(
            _camera_info_from_frame(
                color_frame,
                fov_deg=self.config.camera_fov_deg,
                axis=self.config.camera_fov_axis,
                frame_id="camera_optical",
            ).with_ts(wall_ts)
        )

        if companion_image is not None:
            self.companion_image.publish(
                Image.from_numpy(
                    companion_image,
                    format=ImageFormat.BGR,
                    frame_id="companion_camera_optical",
                    ts=wall_ts,
                )
            )
        if bottom_image is not None:
            self.bottom_image.publish(
                Image.from_numpy(
                    bottom_image,
                    format=ImageFormat.BGR,
                    frame_id="bottom_camera_optical",
                    ts=wall_ts,
                )
            )

        pose = self._resolve_pose(packet, wall_ts)
        self._publish_pose_and_tf(pose)

        if packet.imu_samples:
            latest = packet.imu_samples[-1]
            self.imu.publish(
                Imu(
                    angular_velocity=Vector3(latest.gx, latest.gy, latest.gz),
                    linear_acceleration=Vector3(latest.ax, latest.ay, latest.az),
                    orientation=pose.orientation,
                    frame_id="imu_link",
                    ts=wall_ts,
                )
            )

    def _resolve_pose(self, packet: _SlamInputPacket, wall_ts: float) -> PoseStamped:
        dt = 0.0
        if self._last_packet_wall_ts is not None:
            dt = max(0.0, min(wall_ts - self._last_packet_wall_ts, 0.5))
        self._last_packet_wall_ts = wall_ts

        theta_rate = 0.0
        if packet.imu_samples:
            theta_rate = float(packet.imu_samples[-1].gz)
        elif packet.base_velocity:
            theta_rate = float(packet.base_velocity.get("theta.vel", 0.0))
        self._yaw_rad += theta_rate * dt
        self._apply_absolute_heading_corrections(
            wall_ts=wall_ts,
            imu_heading_rad=(
                _magnetometer_yaw_rad(packet.imu_samples[-1])
                if packet.imu_samples
                else None
            )
        )

        slam_is_fresh = (
            self._last_slam_position_xy is not None
            and self._last_slam_wall_ts is not None
            and (wall_ts - self._last_slam_wall_ts) <= float(self.config.slam_output_stale_after_s)
        )
        if slam_is_fresh:
            self._dead_reckon_xy = np.asarray(self._last_slam_position_xy, dtype=np.float64)
        else:
            vx = float(packet.base_velocity.get("x.vel", 0.0))
            vy = float(packet.base_velocity.get("y.vel", 0.0))
            cos_yaw = math.cos(self._yaw_rad)
            sin_yaw = math.sin(self._yaw_rad)
            dx = (vx * cos_yaw) - (vy * sin_yaw)
            dy = (vx * sin_yaw) + (vy * cos_yaw)
            self._dead_reckon_xy += np.asarray((dx * dt, dy * dt), dtype=np.float64)

        orientation = Quaternion.from_euler(Vector3(0.0, 0.0, self._yaw_rad))
        return PoseStamped(
            ts=wall_ts,
            frame_id="world",
            position=[float(self._dead_reckon_xy[0]), float(self._dead_reckon_xy[1]), 0.0],
            orientation=orientation,
        )

    def _publish_pose_and_tf(self, pose: PoseStamped) -> None:
        self.odom.publish(pose)

        self.tf.publish(
            Transform.from_pose("base_link", pose),
            Transform(
                translation=Vector3(*self.config.front_camera_xyz_m),
                rotation=_quaternion_from_rpy_deg(self.config.front_camera_rpy_deg),
                frame_id="base_link",
                child_frame_id="camera_link",
                ts=pose.ts,
            ),
            Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=_OPTICAL_ROTATION,
                frame_id="camera_link",
                child_frame_id="camera_optical",
                ts=pose.ts,
            ),
            Transform(
                translation=Vector3(*self.config.companion_camera_xyz_m),
                rotation=_quaternion_from_rpy_deg(self.config.companion_camera_rpy_deg),
                frame_id="base_link",
                child_frame_id="companion_camera_link",
                ts=pose.ts,
            ),
            Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=_OPTICAL_ROTATION,
                frame_id="companion_camera_link",
                child_frame_id="companion_camera_optical",
                ts=pose.ts,
            ),
            Transform(
                translation=Vector3(*self.config.bottom_camera_xyz_m),
                rotation=_quaternion_from_rpy_deg(self.config.bottom_camera_rpy_deg),
                frame_id="base_link",
                child_frame_id="bottom_camera_link",
                ts=pose.ts,
            ),
            Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=_OPTICAL_ROTATION,
                frame_id="bottom_camera_link",
                child_frame_id="bottom_camera_optical",
                ts=pose.ts,
            ),
        )

    @staticmethod
    def _get_camera(packet: _SlamInputPacket, preferred_key: str) -> _CameraPacket | None:
        if preferred_key in packet.cameras:
            return packet.cameras[preferred_key]
        if packet.cameras:
            return next(iter(packet.cameras.values()))
        return None



