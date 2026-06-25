from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SourcceySurveyRecorderConfig(ModuleConfig):
    session_root: str = Field(default_factory=lambda: str(Path(os.environ.get("DIMOS_HOME", ".")) / "artifacts" / "sourccey_survey"))
    save_interval_s: float = 0.75
    poll_interval_s: float = 0.10
    min_translation_m: float = 0.03
    min_rotation_deg: float = 5.0
    jpeg_quality: int = 88
    save_companion: bool = True
    save_bottom: bool = True
    log_prefix: str = "SourcceySurvey"


class SourcceySurveyRecorder(Module):
    dedicated_worker = True

    config: SourcceySurveyRecorderConfig

    color_image: In[Image]
    companion_image: In[Image]
    bottom_image: In[Image]
    camera_info: In[CameraInfo]
    odom: In[PoseStamped]
    imu: In[Imu]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._latest_primary: Image | None = None
        self._latest_companion: Image | None = None
        self._latest_bottom: Image | None = None
        self._latest_camera_info: CameraInfo | None = None
        self._latest_odom: PoseStamped | None = None
        self._latest_imu: Imu | None = None

        self._session_dir: Path | None = None
        self._session_manifest_path: Path | None = None
        self._saved_count = 0
        self._last_save_time = 0.0
        self._last_saved_position: tuple[float, float, float] | None = None
        self._last_saved_yaw_deg: float | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._prepare_session_dir()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(Disposable(self.companion_image.subscribe(self._on_companion_image)))
        self.register_disposable(Disposable(self.bottom_image.subscribe(self._on_bottom_image)))
        self.register_disposable(Disposable(self.camera_info.subscribe(self._on_camera_info)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.imu.subscribe(self._on_imu)))
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, name="sourccey-survey-recorder", daemon=True)
        self._thread.start()
        logger.info("%s started session_dir=%s", self.config.log_prefix, self._session_dir)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        logger.info("%s stopped saved_frames=%s session_dir=%s", self.config.log_prefix, self._saved_count, self._session_dir)
        super().stop()

    @rpc
    def latest_session_dir(self) -> str | None:
        return None if self._session_dir is None else str(self._session_dir)

    @rpc
    def session_status(self) -> dict[str, Any]:
        with self._lock:
            odom = self._latest_odom
            imu = self._latest_imu
            return {
                "session_dir": None if self._session_dir is None else str(self._session_dir),
                "saved_frames": self._saved_count,
                "has_primary": self._latest_primary is not None,
                "has_companion": self._latest_companion is not None,
                "has_bottom": self._latest_bottom is not None,
                "has_camera_info": self._latest_camera_info is not None,
                "has_odom": odom is not None,
                "has_imu": imu is not None,
                "latest_pose": None if odom is None else {
                    "x": float(odom.x),
                    "y": float(odom.y),
                    "z": float(odom.z),
                    "yaw_deg": float(odom.yaw) * 180.0 / np.pi,
                },
            }

    def _on_color_image(self, msg: Image) -> None:
        with self._lock:
            self._latest_primary = msg.copy()

    def _on_companion_image(self, msg: Image) -> None:
        with self._lock:
            self._latest_companion = msg.copy()

    def _on_bottom_image(self, msg: Image) -> None:
        with self._lock:
            self._latest_bottom = msg.copy()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = msg

    def _on_imu(self, msg: Imu) -> None:
        with self._lock:
            self._latest_imu = msg

    def _prepare_session_dir(self) -> None:
        root = Path(self.config.session_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_dir = root / f"session_{stamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._session_manifest_path = self._session_dir / "manifest.jsonl"
        (self._session_dir / "primary").mkdir(exist_ok=True)
        (self._session_dir / "companion").mkdir(exist_ok=True)
        (self._session_dir / "bottom").mkdir(exist_ok=True)

    def _record_loop(self) -> None:
        poll_interval = max(float(self.config.poll_interval_s), 0.05)
        while not self._stop_event.wait(poll_interval):
            snapshot = self._snapshot()
            if snapshot is None:
                continue
            if not self._should_save(snapshot):
                continue
            self._save_snapshot(snapshot)

    def _snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            primary = self._latest_primary.copy() if self._latest_primary is not None else None
            companion = self._latest_companion.copy() if self._latest_companion is not None else None
            bottom = self._latest_bottom.copy() if self._latest_bottom is not None else None
            camera_info = self._latest_camera_info
            odom = self._latest_odom
            imu = self._latest_imu

        if primary is None or odom is None:
            return None
        return {
            "primary": primary,
            "companion": companion,
            "bottom": bottom,
            "camera_info": camera_info,
            "odom": odom,
            "imu": imu,
            "wall_time": time.time(),
        }

    def _should_save(self, snapshot: dict[str, Any]) -> bool:
        now = float(snapshot["wall_time"])
        if self._saved_count == 0:
            return True
        if now - self._last_save_time < float(self.config.save_interval_s):
            return False

        odom = snapshot["odom"]
        current_position = (float(odom.x), float(odom.y), float(odom.z))
        current_yaw_deg = float(odom.yaw) * 180.0 / np.pi

        moved_enough = False
        if self._last_saved_position is not None:
            moved_enough = (
                np.linalg.norm(np.asarray(current_position) - np.asarray(self._last_saved_position))
                >= float(self.config.min_translation_m)
            )

        rotated_enough = False
        if self._last_saved_yaw_deg is not None:
            rotated_enough = abs(current_yaw_deg - self._last_saved_yaw_deg) >= float(self.config.min_rotation_deg)

        return moved_enough or rotated_enough

    def _save_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self._session_dir is None or self._session_manifest_path is None:
            return

        frame_index = self._saved_count
        stem = f"frame_{frame_index:05d}"
        primary_path = self._session_dir / "primary" / f"{stem}.jpg"
        companion_path = self._session_dir / "companion" / f"{stem}.jpg"
        bottom_path = self._session_dir / "bottom" / f"{stem}.jpg"

        primary = snapshot["primary"].to_opencv()
        self._write_jpeg(primary_path, primary)

        companion_saved = False
        if self.config.save_companion and snapshot["companion"] is not None:
            self._write_jpeg(companion_path, snapshot["companion"].to_opencv())
            companion_saved = True

        bottom_saved = False
        if self.config.save_bottom and snapshot["bottom"] is not None:
            self._write_jpeg(bottom_path, snapshot["bottom"].to_opencv())
            bottom_saved = True

        odom = snapshot["odom"]
        imu = snapshot["imu"]
        camera_info = snapshot["camera_info"]

        row = {
            "frame_index": frame_index,
            "timestamp": float(snapshot["wall_time"]),
            "primary_path": str(primary_path.name),
            "companion_path": str(companion_path.name) if companion_saved else None,
            "bottom_path": str(bottom_path.name) if bottom_saved else None,
            "pose": {
                "x": float(odom.x),
                "y": float(odom.y),
                "z": float(odom.z),
                "yaw_deg": float(odom.yaw) * 180.0 / np.pi,
            },
            "imu": None if imu is None else {
                "gyro": [float(imu.angular_velocity.x), float(imu.angular_velocity.y), float(imu.angular_velocity.z)],
                "accel": [float(imu.linear_acceleration.x), float(imu.linear_acceleration.y), float(imu.linear_acceleration.z)],
            },
            "camera_info": None if camera_info is None else {
                "width": int(camera_info.width),
                "height": int(camera_info.height),
                "frame_id": str(camera_info.frame_id),
                "K": list(camera_info.K),
                "D": list(camera_info.D),
            },
        }
        with self._session_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        self._saved_count += 1
        self._last_save_time = float(snapshot["wall_time"])
        self._last_saved_position = (float(odom.x), float(odom.y), float(odom.z))
        self._last_saved_yaw_deg = float(odom.yaw) * 180.0 / np.pi
        logger.info("%s saved %s pose=(%.2f, %.2f, %.2f) yaw=%.1fdeg", self.config.log_prefix, stem, float(odom.x), float(odom.y), float(odom.z), self._last_saved_yaw_deg)

    def _write_jpeg(self, path: Path, frame: Any) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(self.config.jpeg_quality, 1, 100))]
        cv2.imwrite(str(path), frame, params)
