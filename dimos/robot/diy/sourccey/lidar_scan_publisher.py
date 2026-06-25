from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

from pydantic import Field

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.utils.logging_config import setup_logger

from .lidar_types import PlanarLidarScan

logger = setup_logger()


class SourcceyLidarScanPublisherConfig(ModuleConfig):
    host: str = Field(default_factory=lambda m: m["g"].robot_ip or "127.0.0.1")
    port: int = 8765
    frame_id: str = "base_lidar"
    reconnect_delay_s: float = 0.5
    connect_timeout_s: float = 5.0


class SourcceyLidarScanPublisher(Module):
    dedicated_worker = True

    config: SourcceyLidarScanPublisherConfig

    scan: Out[PlanarLidarScan]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sourccey-lidar-scan-publisher",
            daemon=True,
        )
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        super().stop()

    def _run_loop(self) -> None:
        reconnect_delay_s = max(0.1, float(self.config.reconnect_delay_s))
        address = (self.config.host, int(self.config.port))
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(address, timeout=float(self.config.connect_timeout_s)) as sock:
                    file_obj = sock.makefile("r", encoding="utf-8")
                    logger.info("Connected to Sourccey LiDAR stream at tcp://%s:%s", *address)
                    for line in file_obj:
                        if self._stop_event.is_set():
                            break
                        payload = json.loads(line)
                        scan = PlanarLidarScan.from_points(
                            ts=float(payload.get("ts", time.time())),
                            frame_id=self.config.frame_id,
                            rpm=float(payload.get("rpm", 0.0)),
                            points=list(payload.get("points", [])),
                        )
                        self.scan.publish(scan)
            except Exception as exc:
                logger.warning("LiDAR scan stream disconnected: %s", exc)
                time.sleep(reconnect_delay_s)

