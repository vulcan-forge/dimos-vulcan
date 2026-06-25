from __future__ import annotations

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


class SourcceyMotionTestConfig(ModuleConfig):
    startup_delay_s: float = 1.5
    command_rate_hz: float = 12.0
    move_duration_s: float = 0.75
    settle_duration_s: float = 1.0
    linear_x: float = 0.95
    linear_y: float = 0.0
    angular_z: float = 0.0
    repeat: bool = False
    label: str = Field(default="forward")


class SourcceyMotionTest(Module):
    config: SourcceyMotionTestConfig

    cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="sourccey-motion-test", daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self.cmd_vel.publish(Twist.zero())
        super().stop()

    def _run_loop(self) -> None:
        time.sleep(max(0.0, float(self.config.startup_delay_s)))
        period = 1.0 / max(1.0, float(self.config.command_rate_hz))
        active_twist = Twist(
            linear=Vector3(float(self.config.linear_x), float(self.config.linear_y), 0.0),
            angular=Vector3(0.0, 0.0, float(self.config.angular_z)),
        )
        while not self._stop_event.is_set():
            deadline = time.monotonic() + max(0.0, float(self.config.move_duration_s))
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                self.cmd_vel.publish(active_twist)
                time.sleep(period)
            self.cmd_vel.publish(Twist.zero())
            settle_deadline = time.monotonic() + max(0.0, float(self.config.settle_duration_s))
            while not self._stop_event.is_set() and time.monotonic() < settle_deadline:
                time.sleep(min(0.1, period))
            if not self.config.repeat:
                break


