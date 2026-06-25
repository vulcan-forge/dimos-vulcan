from __future__ import annotations

import math
import re
import threading
import time
from typing import Any

from pydantic import Field

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SourcceySurveyRotationConfig(ModuleConfig):
    startup_delay_s: float = 1.5
    command_rate_hz: float = 14.0
    total_sweep_deg: float = 120.0
    turn_step_deg: float = 15.0
    turn_speed_rad_s: float = 0.80
    turn_calibration_multiplier: float = 1.33
    settle_duration_s: float = 0.9
    capture_hold_s: float = 1.6
    micro_parallax_enabled: bool = True
    micro_parallax_distance_m: float = 0.05
    micro_parallax_speed_m_s: float = 0.08
    micro_parallax_settle_s: float = 0.8
    micro_parallax_capture_s: float = 1.0
    micro_parallax_return_to_origin: bool = True
    direction: str = Field(default="left")
    segment_sequence: str = "left:60,right:120,left:60"
    repeat: bool = False
    log_prefix: str = "SourcceySurveyScan"


class SourcceySurveyRotation(Module):
    config: SourcceySurveyRotationConfig

    cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="sourccey-survey-rotation", daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        self.cmd_vel.publish(Twist.zero())
        super().stop()

    def _run_loop(self) -> None:
        time.sleep(max(0.0, float(self.config.startup_delay_s)))

        command_rate_hz = max(1.0, float(self.config.command_rate_hz))
        period = 1.0 / command_rate_hz
        turn_speed = max(0.05, abs(float(self.config.turn_speed_rad_s)))
        turn_step_deg = max(0.1, float(self.config.turn_step_deg))
        turn_calibration_multiplier = max(0.1, float(self.config.turn_calibration_multiplier))
        settle_duration_s = max(0.0, float(self.config.settle_duration_s))
        capture_hold_s = max(0.0, float(self.config.capture_hold_s))
        micro_parallax_enabled = bool(self.config.micro_parallax_enabled)
        micro_parallax_distance_m = max(0.0, float(self.config.micro_parallax_distance_m))
        micro_parallax_speed_m_s = max(0.01, float(self.config.micro_parallax_speed_m_s))
        micro_parallax_settle_s = max(0.0, float(self.config.micro_parallax_settle_s))
        micro_parallax_capture_s = max(0.0, float(self.config.micro_parallax_capture_s))
        micro_parallax_return_to_origin = bool(self.config.micro_parallax_return_to_origin)
        segment_sequence = self._parse_segment_sequence()

        while not self._stop_event.is_set():
            step_index = 0
            commanded_total_deg = sum(requested_deg * turn_calibration_multiplier for _, requested_deg in segment_sequence)
            logger.info(
                "%s starting segment_sequence=%s turn_step_deg=%.1f turn_speed_rad_s=%.2f calibration=%.2f settle_duration_s=%.2f capture_hold_s=%.2f commanded_total_deg=%.1f micro_parallax_enabled=%s micro_parallax_distance_m=%.3f micro_parallax_speed_m_s=%.3f",
                self.config.log_prefix,
                self.config.segment_sequence,
                turn_step_deg,
                turn_speed,
                turn_calibration_multiplier,
                settle_duration_s,
                capture_hold_s,
                commanded_total_deg,
                micro_parallax_enabled,
                micro_parallax_distance_m,
                micro_parallax_speed_m_s,
            )

            for segment_index, (direction_name, requested_segment_deg) in enumerate(segment_sequence, start=1):
                if self._stop_event.is_set():
                    break
                direction = -1.0 if direction_name.startswith("r") else 1.0
                remaining_requested_deg = requested_segment_deg
                logger.info(
                    "%s segment=%s/%s direction=%s requested_segment_deg=%.1f",
                    self.config.log_prefix,
                    segment_index,
                    len(segment_sequence),
                    direction_name,
                    requested_segment_deg,
                )
                while not self._stop_event.is_set() and remaining_requested_deg > 1e-6:
                    step_index += 1
                    requested_step_deg = min(turn_step_deg, remaining_requested_deg)
                    commanded_step_deg = requested_step_deg * turn_calibration_multiplier
                    turn_duration_s = math.radians(commanded_step_deg) / turn_speed
                    twist = Twist(
                        linear=Vector3(0.0, 0.0, 0.0),
                        angular=Vector3(0.0, 0.0, direction * turn_speed),
                    )

                    logger.info(
                        "%s step=%s requested_turn_deg=%.1f commanded_turn_deg=%.1f turn_duration_s=%.2f remaining_requested_after_step_deg=%.1f",
                        self.config.log_prefix,
                        step_index,
                        requested_step_deg,
                        commanded_step_deg,
                        turn_duration_s,
                        max(0.0, remaining_requested_deg - requested_step_deg),
                    )

                    turn_deadline = time.monotonic() + turn_duration_s
                    while not self._stop_event.is_set() and time.monotonic() < turn_deadline:
                        self.cmd_vel.publish(twist)
                        time.sleep(period)

                    self.cmd_vel.publish(Twist.zero())

                    settle_deadline = time.monotonic() + settle_duration_s
                    while not self._stop_event.is_set() and time.monotonic() < settle_deadline:
                        time.sleep(min(0.1, period))

                    capture_deadline = time.monotonic() + capture_hold_s
                    while not self._stop_event.is_set() and time.monotonic() < capture_deadline:
                        time.sleep(min(0.1, period))

                    if micro_parallax_enabled and micro_parallax_distance_m > 1e-6:
                        self._run_micro_parallax(
                            period=period,
                            distance_m=micro_parallax_distance_m,
                            speed_m_s=micro_parallax_speed_m_s,
                            settle_s=micro_parallax_settle_s,
                            capture_s=micro_parallax_capture_s,
                            return_to_origin=micro_parallax_return_to_origin,
                        )

                    remaining_requested_deg -= requested_step_deg

            self.cmd_vel.publish(Twist.zero())
            logger.info(
                "%s complete segment_sequence=%s calibration=%.2f executed_steps=%s",
                self.config.log_prefix,
                self.config.segment_sequence,
                turn_calibration_multiplier,
                step_index,
            )

            if not self.config.repeat:
                break

    def _pause(self, duration_s: float, period: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_s)
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.1, period))

    def _drive_linear(self, linear_x: float, duration_s: float, period: float) -> None:
        twist = Twist(
            linear=Vector3(linear_x, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0),
        )
        deadline = time.monotonic() + max(0.0, duration_s)
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            self.cmd_vel.publish(twist)
            time.sleep(period)
        self.cmd_vel.publish(Twist.zero())

    def _run_micro_parallax(
        self,
        *,
        period: float,
        distance_m: float,
        speed_m_s: float,
        settle_s: float,
        capture_s: float,
        return_to_origin: bool,
    ) -> None:
        drive_duration_s = distance_m / max(0.01, speed_m_s)
        logger.info(
            "%s micro_parallax distance_m=%.3f speed_m_s=%.3f drive_duration_s=%.2f return_to_origin=%s",
            self.config.log_prefix,
            distance_m,
            speed_m_s,
            drive_duration_s,
            return_to_origin,
        )
        self._drive_linear(speed_m_s, drive_duration_s, period)
        self._pause(settle_s, period)
        self._pause(capture_s, period)
        if return_to_origin:
            self._drive_linear(-speed_m_s, drive_duration_s, period)
            self._pause(settle_s, period)

    def _parse_segment_sequence(self) -> list[tuple[str, float]]:
        raw = str(self.config.segment_sequence).strip()
        if raw:
            parsed: list[tuple[str, float]] = []
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                match = re.fullmatch(r"(left|right)\s*:\s*([0-9]+(?:\.[0-9]+)?)", token, flags=re.IGNORECASE)
                if match is None:
                    raise ValueError(
                        f"Invalid segment_sequence token '{token}'. Expected entries like 'left:60,right:120,left:60'."
                    )
                parsed.append((match.group(1).lower(), max(0.0, float(match.group(2)))))
            if parsed:
                return parsed

        default_direction = str(self.config.direction).lower()
        default_direction = "right" if default_direction.startswith("r") else "left"
        return [(default_direction, max(0.0, float(self.config.total_sweep_deg)))]
