# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disabled Sourccey arm adapter.

Sourccey's arms are intentionally non-controllable from DimOS after a previous
hardware incident. This adapter remains only so the failure mode is explicit
and safe if someone tries to wire it back in.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

from dimos.hardware.manipulators.spec import (
    ControlMode,
    JointLimits,
    ManipulatorInfo,
)
from dimos.robot.diy.sourccey.hardware_session import Side, SourcceyHardwareSession, get_sourccey_session

if TYPE_CHECKING:
    from dimos.hardware.manipulators.registry import AdapterRegistry


class SourcceyArmAdapter:
    def __init__(
        self,
        dof: int = 5,
        address: str | Path | None = None,
        hardware_id: str = "left_arm",
        side: str | None = None,
        command_endpoint: str | None = None,
        observation_endpoint: str | None = None,
        ready_timeout_s: float = 1.5,
        servo_dt_s: float = 0.08,
        **_: object,
    ) -> None:
        if dof not in (5, 6):
            raise ValueError(f"Sourccey arm expects 5 arm joints (+ optional gripper), got {dof}")
        self._dof = dof
        self._address = address
        self._hardware_id = hardware_id
        self._side: Side = self._infer_side(hardware_id, side)
        self._command_endpoint = command_endpoint
        self._observation_endpoint = observation_endpoint
        self._ready_timeout_s = ready_timeout_s
        self._servo_dt_s = max(0.01, float(servo_dt_s))
        self._session: SourcceyHardwareSession | None = None
        self._enabled = False
        self._connected = False
        self._control_mode = ControlMode.POSITION
        self._last_read_positions: list[float] | None = None
        self._last_read_ts: float | None = None

    @staticmethod
    def _infer_side(hardware_id: str, side: str | None) -> Side:
        if side in {"left", "right"}:
            return side
        return "right" if "right" in hardware_id.lower() else "left"

    def connect(self) -> bool:
        self._connected = False
        return False

    def disconnect(self) -> None:
        self.write_stop()
        if self._session is not None:
            self._session.release()
            self._session = None
        self._connected = False
        self._enabled = False

    def is_connected(self) -> bool:
        return self._connected

    def activate(self) -> bool:
        return False

    def deactivate(self) -> bool:
        return False

    def get_info(self) -> ManipulatorInfo:
        return ManipulatorInfo(vendor="Sourccey", model=f"{self._side}_arm", dof=self._dof)

    def get_dof(self) -> int:
        return self._dof

    def get_limits(self) -> JointLimits:
        lower = [-math.pi] * min(self._dof, 5)
        upper = [math.pi] * min(self._dof, 5)
        velocity = [2.0] * min(self._dof, 5)
        if self._dof == 6:
            lower.append(0.0)
            upper.append(1.0)
            velocity.append(1.0)
        return JointLimits(
            position_lower=lower,
            position_upper=upper,
            velocity_max=velocity,
        )

    def set_control_mode(self, mode: ControlMode) -> bool:
        self._control_mode = mode
        return True

    def get_control_mode(self) -> ControlMode:
        return self._control_mode

    def read_joint_positions(self) -> list[float]:
        full_state = self._current_arm_state()
        if full_state is None:
            return [0.0] * self._dof
        if self._dof == 6:
            return list(full_state)
        return list(full_state[: self._dof])

    def read_joint_velocities(self) -> list[float]:
        now = time.monotonic()
        positions = self.read_joint_positions()
        if self._last_read_positions is None or self._last_read_ts is None:
            self._last_read_positions = list(positions)
            self._last_read_ts = now
            return [0.0] * len(positions)

        dt = max(now - self._last_read_ts, 1e-6)
        velocities = [(pos - prev) / dt for pos, prev in zip(positions, self._last_read_positions)]
        self._last_read_positions = list(positions)
        self._last_read_ts = now
        return velocities

    def read_joint_efforts(self) -> list[float]:
        return [0.0] * self._dof

    def read_state(self) -> dict[str, int]:
        return {
            "state": 0 if self._enabled else 1,
            "mode": list(ControlMode).index(self._control_mode),
        }

    def read_error(self) -> tuple[int, str]:
        if self._session is None:
            return 1, "not connected"
        error = self._session.read_error()
        return (0, "") if not error else (1, error)

    def write_joint_positions(
        self,
        positions: list[float],
        velocity: float = 1.0,
    ) -> bool:
        del positions, velocity
        return False

    def write_joint_velocities(self, velocities: list[float]) -> bool:
        del velocities
        return False

    def write_stop(self) -> bool:
        return False

    def write_enable(self, enable: bool) -> bool:
        del enable
        self._enabled = False
        return False

    def read_enabled(self) -> bool:
        return self._enabled

    def write_clear_errors(self) -> bool:
        return True

    def read_cartesian_position(self) -> dict[str, float] | None:
        return None

    def write_cartesian_position(
        self,
        pose: dict[str, float],
        velocity: float = 1.0,
    ) -> bool:
        del pose, velocity
        return False

    def read_gripper_position(self) -> float | None:
        state = self._current_arm_state()
        if state is None:
            return 0.0
        return float(state[5])

    def write_gripper_position(self, position: float) -> bool:
        del position
        return False

    def read_force_torque(self) -> list[float] | None:
        return None

    def _current_arm_state(self) -> tuple[float, float, float, float, float, float] | None:
        if self._session is None:
            return None
        return self._session.read_arm_state(self._side)


def register(registry: AdapterRegistry) -> None:
    # Intentionally do not register a live Sourccey manipulator adapter.
    # Arms must remain non-controllable from DimOS.
    del registry
