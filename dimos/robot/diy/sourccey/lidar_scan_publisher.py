from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

from pydantic import Field
import serial

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.utils.logging_config import setup_logger

from .lidar_types import PlanarLidarScan

logger = setup_logger()

_POINTS_PER_PACKET = 12
_PACKET_LEN = 47
_HEADER_BYTE = 0x54
_VER_LEN_BYTE = 0x2C


def _read_exact(ser: serial.Serial, length: int) -> bytes:
    payload = bytearray()
    while len(payload) < length:
        chunk = ser.read(length - len(payload))
        if not chunk:
            raise TimeoutError("Timed out waiting for LiDAR bytes.")
        payload.extend(chunk)
    return bytes(payload)


def _read_packet(ser: serial.Serial) -> bytes:
    while True:
        first = ser.read(1)
        if not first:
            raise TimeoutError("Timed out waiting for LiDAR header.")
        if first[0] != _HEADER_BYTE:
            continue
        second = ser.read(1)
        if not second:
            raise TimeoutError("Timed out waiting for LiDAR length byte.")
        if second[0] != _VER_LEN_BYTE:
            continue
        remainder = _read_exact(ser, _PACKET_LEN - 2)
        return first + second + remainder


def _u16(lo: int, hi: int) -> int:
    return lo | (hi << 8)


def _parse_packet(packet: bytes) -> tuple[float, float, list[tuple[float, float, int]]]:
    speed_deg_s = float(_u16(packet[2], packet[3]))
    start_angle_deg = _u16(packet[4], packet[5]) / 100.0
    end_angle_deg = _u16(packet[42], packet[43]) / 100.0

    span_deg = end_angle_deg - start_angle_deg
    if span_deg < 0:
        span_deg += 360.0

    points: list[tuple[float, float, int]] = []
    for idx in range(_POINTS_PER_PACKET):
        offset = 6 + idx * 3
        distance_mm = _u16(packet[offset], packet[offset + 1])
        confidence = int(packet[offset + 2])
        if distance_mm <= 0:
            continue

        if _POINTS_PER_PACKET == 1:
            angle_deg = start_angle_deg
        else:
            angle_deg = (start_angle_deg + span_deg * idx / (_POINTS_PER_PACKET - 1)) % 360.0
        points.append((angle_deg, distance_mm / 1000.0, confidence))

    return speed_deg_s, start_angle_deg, points


class SourcceyLidarScanPublisherConfig(ModuleConfig):
    source: str = "serial"
    host: str = Field(default_factory=lambda m: m["g"].robot_ip or "127.0.0.1")
    port: int = 8765
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 230400
    serial_timeout_s: float = 1.0
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
        source = str(self.config.source).strip().lower()
        if source == "tcp":
            self._run_tcp_loop()
            return
        if source != "serial":
            logger.warning("Unknown Sourccey LiDAR source '%s'; falling back to serial", source)
        self._run_serial_loop()

    def _run_tcp_loop(self) -> None:
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
                logger.warning("LiDAR scan TCP stream disconnected: %s", exc)
                time.sleep(reconnect_delay_s)

    def _run_serial_loop(self) -> None:
        reconnect_delay_s = max(0.1, float(self.config.reconnect_delay_s))
        while not self._stop_event.is_set():
            try:
                with serial.Serial(
                    self.config.serial_port,
                    int(self.config.serial_baud),
                    timeout=float(self.config.serial_timeout_s),
                ) as ser:
                    logger.info(
                        "Connected to Sourccey LiDAR serial device at %s @ %s baud",
                        self.config.serial_port,
                        self.config.serial_baud,
                    )
                    revolution_points: list[tuple[float, float, int]] = []
                    previous_angle: float | None = None
                    latest_rpm = 0.0

                    while not self._stop_event.is_set():
                        packet = _read_packet(ser)
                        speed_deg_s, start_angle_deg, packet_points = _parse_packet(packet)
                        latest_rpm = speed_deg_s / 360.0 * 60.0

                        if previous_angle is not None and start_angle_deg + 2.0 < previous_angle and revolution_points:
                            self.scan.publish(
                                PlanarLidarScan.from_points(
                                    ts=time.time(),
                                    frame_id=self.config.frame_id,
                                    rpm=latest_rpm,
                                    points=revolution_points,
                                )
                            )
                            revolution_points = []

                        revolution_points.extend(packet_points)
                        previous_angle = start_angle_deg
            except Exception as exc:
                logger.warning("LiDAR serial stream disconnected: %s", exc)
                time.sleep(reconnect_delay_s)
