from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Literal

import zmq

from dimos.utils.logging_config import setup_logger

from .connection import (
    _RobotObservationState,
    _build_robot_action_packet,
    _parse_robot_observation,
    _parse_slam_output_packet,
    _socket_connect_pull,
    _socket_connect_push,
    _socket_connect_sub,
)

logger = setup_logger()

Side = Literal["left", "right"]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_COMMAND_PORT = 5555
_DEFAULT_OBSERVATION_PORT = 5556
_DEFAULT_SLAM_OUTPUT_PORT = 5561


def _normalize_yaw_rad(yaw_rad: float) -> float:
    return math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))


def _as_endpoint(
    explicit: str | None,
    *,
    address: str | Path | None,
    default_port: int,
) -> str:
    if explicit:
        raw = str(explicit).strip()
    elif address is not None:
        raw = str(address).strip()
    else:
        raw = _DEFAULT_HOST

    if raw.startswith("tcp://"):
        return raw

    host = raw
    port = default_port
    if ":" in raw and raw.count(":") == 1:
        maybe_host, maybe_port = raw.rsplit(":", 1)
        if maybe_port.isdigit():
            host = maybe_host
            port = int(maybe_port)

    if not host:
        host = _DEFAULT_HOST
    return f"tcp://{host}:{port}"


class SourcceyHardwareSession:
    """Shared command/state bridge for Sourccey's full-body protobuf transport.

    Safety rule: this session is now BASE-ONLY for control. We still read arm
    state for telemetry, but every outgoing packet force-untorques both arms.
    No DimOS module should be able to torque or move Sourccey's arms through
    this session.
    """

    def __init__(
        self,
        *,
        command_endpoint: str,
        observation_endpoint: str,
        slam_output_endpoint: str | None,
        ready_timeout_s: float = 1.5,
    ) -> None:
        self.command_endpoint = command_endpoint
        self.observation_endpoint = observation_endpoint
        self.slam_output_endpoint = slam_output_endpoint
        self.ready_timeout_s = ready_timeout_s

        self._ref_count = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._context: zmq.Context | None = None
        self._command_socket: zmq.Socket | None = None
        self._observation_socket: zmq.Socket | None = None
        self._slam_output_socket: zmq.Socket | None = None

        self._latest_observation: _RobotObservationState | None = None
        self._latest_slam_xy: tuple[float, float] | None = None
        self._integrated_pose = [0.0, 0.0, 0.0]
        self._last_integration_monotonic: float | None = None

        self._base_enabled = False
        self._base_command = [0.0, 0.0, 0.0]
        self._last_error: str = ""

    def acquire(self) -> "SourcceyHardwareSession":
        with self._lock:
            self._ref_count += 1
            first = self._ref_count == 1
        if first:
            self._start()
        return self

    def release(self) -> None:
        with self._lock:
            if self._ref_count <= 0:
                return
            self._ref_count -= 1
            should_stop = self._ref_count == 0
        if should_stop:
            self._stop()

    def state_ready(self) -> bool:
        with self._lock:
            return self._latest_observation is not None

    def read_error(self) -> str:
        with self._lock:
            return self._last_error

    def read_observation(self) -> _RobotObservationState | None:
        with self._lock:
            return self._latest_observation

    def read_base_velocities(self) -> list[float]:
        with self._lock:
            if self._latest_observation is not None:
                obs = self._latest_observation
                return [obs.x_vel, obs.y_vel, obs.theta_vel]
            return list(self._base_command)

    def read_odometry(self) -> list[float] | None:
        with self._lock:
            self._integrate_locked(time.monotonic())
            if self._latest_observation is None and self._latest_slam_xy is None:
                return None
            return list(self._integrated_pose)

    def read_arm_state(
        self, side: Side
    ) -> tuple[float, float, float, float, float, float] | None:
        with self._lock:
            if self._latest_observation is None:
                return None
            return (
                self._latest_observation.left_arm
                if side == "left"
                else self._latest_observation.right_arm
            )

    def write_base_enable(self, enable: bool) -> None:
        with self._lock:
            self._base_enabled = bool(enable)
            if not self._base_enabled:
                self._base_command = [0.0, 0.0, 0.0]
            self._send_locked()

    def write_base_command(self, velocities: list[float]) -> bool:
        if len(velocities) != 3:
            return False
        with self._lock:
            self._base_command = [float(v) for v in velocities]
            self._send_locked()
        return True

    def write_arm_enable(self, side: Side, enable: bool) -> None:
        del side, enable
        with self._lock:
            self._last_error = "Sourccey arm actuation is disabled in DimOS"

    def write_arm_positions(
        self,
        side: Side,
        positions: list[float],
        *,
        gripper: float | None = None,
    ) -> bool:
        del side, positions, gripper
        with self._lock:
            self._last_error = "Sourccey arm actuation is disabled in DimOS"
        return False

    def write_gripper(self, side: Side, position: float) -> bool:
        del side, position
        with self._lock:
            self._last_error = "Sourccey arm actuation is disabled in DimOS"
        return False

    def _start(self) -> None:
        self._stop_event.clear()
        self._context = zmq.Context()
        self._command_socket = _socket_connect_push(self._context, self.command_endpoint)
        self._observation_socket = _socket_connect_pull(self._context, self.observation_endpoint)
        if self.slam_output_endpoint:
            self._slam_output_socket = _socket_connect_sub(self._context, self.slam_output_endpoint)
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="sourccey-hardware-session",
            daemon=True,
        )
        self._thread.start()
        self._wait_for_first_state()

    def _stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        sockets = (
            self._command_socket,
            self._observation_socket,
            self._slam_output_socket,
        )
        for sock in sockets:
            if sock is not None:
                try:
                    sock.close(0)
                except Exception:
                    pass
        self._command_socket = None
        self._observation_socket = None
        self._slam_output_socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
        self._context = None

    def _wait_for_first_state(self) -> None:
        deadline = time.monotonic() + max(0.0, float(self.ready_timeout_s))
        while time.monotonic() < deadline:
            if self.state_ready():
                return
            time.sleep(0.02)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            did_work = False
            if self._observation_socket is not None:
                try:
                    payload = self._observation_socket.recv(flags=zmq.NOBLOCK)
                    observation = _parse_robot_observation(payload)
                    with self._lock:
                        self._integrate_locked(time.monotonic())
                        self._latest_observation = observation
                        self._last_error = ""
                    did_work = True
                except zmq.Again:
                    pass
                except Exception as exc:
                    with self._lock:
                        self._last_error = str(exc)
                    logger.warning("Sourccey observation parse failed: %s", exc)

            if self._slam_output_socket is not None:
                try:
                    payload = self._slam_output_socket.recv(flags=zmq.NOBLOCK)
                    slam_output = _parse_slam_output_packet(payload)
                    with self._lock:
                        self._integrate_locked(time.monotonic())
                        self._latest_slam_xy = (slam_output.world_x, slam_output.world_y)
                        self._integrated_pose[0] = slam_output.world_x
                        self._integrated_pose[1] = slam_output.world_y
                        self._last_error = ""
                    did_work = True
                except zmq.Again:
                    pass
                except Exception as exc:
                    with self._lock:
                        self._last_error = str(exc)
                    logger.warning("Sourccey slam_output parse failed: %s", exc)

            if not did_work:
                time.sleep(0.01)

    def _integrate_locked(self, now_monotonic: float) -> None:
        if self._last_integration_monotonic is None:
            self._last_integration_monotonic = now_monotonic
            return

        dt = max(0.0, now_monotonic - self._last_integration_monotonic)
        self._last_integration_monotonic = now_monotonic
        if dt <= 0.0 or self._latest_observation is None:
            return

        heading = float(self._integrated_pose[2])
        vx = float(self._latest_observation.x_vel)
        vy = float(self._latest_observation.y_vel)
        wz = float(self._latest_observation.theta_vel)

        self._integrated_pose[0] += (vx * math.cos(heading) - vy * math.sin(heading)) * dt
        self._integrated_pose[1] += (vx * math.sin(heading) + vy * math.cos(heading)) * dt
        self._integrated_pose[2] = _normalize_yaw_rad(heading + wz * dt)

        if self._latest_slam_xy is not None:
            self._integrated_pose[0] = self._latest_slam_xy[0]
            self._integrated_pose[1] = self._latest_slam_xy[1]

    def _current_arm_target_locked(self, side: Side) -> tuple[float, float, float, float, float, float]:
        if self._latest_observation is not None:
            return (
                self._latest_observation.left_arm
                if side == "left"
                else self._latest_observation.right_arm
            )
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _send_locked(self) -> None:
        if self._command_socket is None:
            return
        if self._latest_observation is None:
            self._last_error = "waiting for observation state"
            return

        base_state = self._latest_observation or _RobotObservationState.zero()

        merged_state = _RobotObservationState(
            left_arm=base_state.left_arm,
            right_arm=base_state.right_arm,
            z_pos=float(base_state.z_pos),
            x_vel=base_state.x_vel,
            y_vel=base_state.y_vel,
            theta_vel=base_state.theta_vel,
        )
        vx, vy, wz = self._base_command if self._base_enabled else [0.0, 0.0, 0.0]
        payload = _build_robot_action_packet(
            x_vel=float(vx),
            y_vel=float(vy),
            theta_vel=float(wz),
            state=merged_state,
            untorque_left=True,
            untorque_right=True,
        )
        try:
            self._command_socket.send(payload, flags=zmq.NOBLOCK)
            self._last_error = ""
        except zmq.Again:
            self._last_error = "command socket busy"
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Sourccey command send failed: %s", exc)


_SESSION_REGISTRY_LOCK = threading.Lock()
_SESSION_REGISTRY: dict[tuple[str, str, str | None], SourcceyHardwareSession] = {}


def get_sourccey_session(
    *,
    address: str | Path | None = None,
    command_endpoint: str | None = None,
    observation_endpoint: str | None = None,
    slam_output_endpoint: str | None = None,
    ready_timeout_s: float = 1.5,
) -> SourcceyHardwareSession:
    resolved_command = _as_endpoint(
        command_endpoint,
        address=address,
        default_port=_DEFAULT_COMMAND_PORT,
    )
    resolved_observation = _as_endpoint(
        observation_endpoint,
        address=address,
        default_port=_DEFAULT_OBSERVATION_PORT,
    )
    resolved_slam_output = (
        None
        if slam_output_endpoint is None
        else _as_endpoint(
            slam_output_endpoint,
            address=address,
            default_port=_DEFAULT_SLAM_OUTPUT_PORT,
        )
    )
    key = (resolved_command, resolved_observation, resolved_slam_output)
    with _SESSION_REGISTRY_LOCK:
        session = _SESSION_REGISTRY.get(key)
        if session is None:
            session = SourcceyHardwareSession(
                command_endpoint=resolved_command,
                observation_endpoint=resolved_observation,
                slam_output_endpoint=resolved_slam_output,
                ready_timeout_s=ready_timeout_s,
            )
            _SESSION_REGISTRY[key] = session
    return session.acquire()
