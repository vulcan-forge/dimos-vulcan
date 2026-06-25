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

"""Sourccey holonomic-base adapter over the native protobuf/ZMQ transport."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dimos.robot.diy.sourccey.hardware_session import SourcceyHardwareSession, get_sourccey_session

if TYPE_CHECKING:
    from dimos.hardware.drive_trains.registry import TwistBaseAdapterRegistry


class SourcceyTwistBaseAdapter:
    def __init__(
        self,
        dof: int = 3,
        address: str | Path | None = None,
        hardware_id: str = "sourccey",
        command_endpoint: str | None = None,
        observation_endpoint: str | None = None,
        slam_output_endpoint: str | None = None,
        ready_timeout_s: float = 1.5,
        **_: object,
    ) -> None:
        if dof != 3:
            raise ValueError(f"Sourccey base expects 3 DOF [vx, vy, wz], got {dof}")
        self._dof = dof
        self._hardware_id = hardware_id
        self._address = address
        self._command_endpoint = command_endpoint
        self._observation_endpoint = observation_endpoint
        self._slam_output_endpoint = slam_output_endpoint
        self._ready_timeout_s = ready_timeout_s
        self._session: SourcceyHardwareSession | None = None
        self._enabled = False
        self._connected = False
        self._last_command = [0.0, 0.0, 0.0]

    def connect(self) -> bool:
        self._session = get_sourccey_session(
            address=self._address,
            command_endpoint=self._command_endpoint,
            observation_endpoint=self._observation_endpoint,
            slam_output_endpoint=self._slam_output_endpoint,
            ready_timeout_s=self._ready_timeout_s,
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.write_stop()
        if self._session is not None:
            self._session.release()
            self._session = None
        self._connected = False
        self._enabled = False
        self._last_command = [0.0, 0.0, 0.0]

    def is_connected(self) -> bool:
        return self._connected

    def get_dof(self) -> int:
        return self._dof

    def read_velocities(self) -> list[float]:
        if self._session is None:
            return list(self._last_command)
        return self._session.read_base_velocities()

    def read_odometry(self) -> list[float] | None:
        if self._session is None:
            return None
        return self._session.read_odometry()

    def write_velocities(self, velocities: list[float]) -> bool:
        if len(velocities) != self._dof or not self._enabled or self._session is None:
            return False
        self._last_command = [float(v) for v in velocities]
        return self._session.write_base_command(self._last_command)

    def write_stop(self) -> bool:
        self._last_command = [0.0, 0.0, 0.0]
        if self._session is None:
            return False
        return self._session.write_base_command(self._last_command)

    def write_enable(self, enable: bool) -> bool:
        self._enabled = bool(enable)
        if self._session is not None:
            self._session.write_base_enable(self._enabled)
        return True

    def read_enabled(self) -> bool:
        return self._enabled


def register(registry: TwistBaseAdapterRegistry) -> None:
    registry.register("sourccey", SourcceyTwistBaseAdapter)
