from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class _ImageStats:
    count: int = 0
    width: int = 0
    height: int = 0
    ts: float = 0.0
    brightness: float = 0.0
    sharpness: float = 0.0


class SourcceySensorDebugConfig(ModuleConfig):
    status_interval_s: float = 1.0
    stale_after_s: float = 2.0
    include_image_metrics: bool = True
    log_prefix: str = Field(default='SourcceyDebug')


class SourcceySensorDebugMonitor(Module):
    config: SourcceySensorDebugConfig

    color_image: In[Image]
    companion_image: In[Image]
    bottom_image: In[Image]
    camera_info: In[CameraInfo]
    odom: In[PoseStamped]
    imu: In[Imu]
    joint_state: In[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._primary = _ImageStats()
        self._companion = _ImageStats()
        self._bottom = _ImageStats()
        self._camera_info_ts = 0.0

        self._odom_count = 0
        self._odom_ts = 0.0
        self._odom_xyz = (0.0, 0.0, 0.0)
        self._odom_yaw_deg = 0.0

        self._imu_count = 0
        self._imu_ts = 0.0
        self._imu_gyro = (0.0, 0.0, 0.0)
        self._imu_accel = (0.0, 0.0, 0.0)

        self._joint_count = 0
        self._joint_ts = 0.0
        self._joint_name_count = 0
        self._last_rate_sample_counts = {
            'primary': 0,
            'companion': 0,
            'bottom': 0,
            'odom': 0,
            'imu': 0,
            'joint': 0,
        }
        self._last_rate_sample_monotonic = time.monotonic()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(Disposable(self.companion_image.subscribe(self._on_companion_image)))
        self.register_disposable(Disposable(self.bottom_image.subscribe(self._on_bottom_image)))
        self.register_disposable(Disposable(self.camera_info.subscribe(self._on_camera_info)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.imu.subscribe(self._on_imu)))
        self.register_disposable(Disposable(self.joint_state.subscribe(self._on_joint_state)))

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._status_loop, name='sourccey-sensor-debug', daemon=True)
        self._thread.start()
        logger.info('%s monitor started', self.config.log_prefix)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        logger.info('%s monitor stopped', self.config.log_prefix)
        super().stop()

    @rpc
    def latest_status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_snapshot_locked(now=time.time())

    @skill
    def sensor_summary(self) -> str:
        status = self.latest_status()
        return self._format_status(status)

    def _on_color_image(self, msg: Image) -> None:
        self._record_image(self._primary, msg)

    def _on_companion_image(self, msg: Image) -> None:
        self._record_image(self._companion, msg)

    def _on_bottom_image(self, msg: Image) -> None:
        self._record_image(self._bottom, msg)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._camera_info_ts = float(getattr(msg, 'ts', 0.0) or time.time())

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._lock:
            self._odom_count += 1
            self._odom_ts = float(msg.ts)
            self._odom_xyz = (float(msg.x), float(msg.y), float(msg.z))
            self._odom_yaw_deg = float(msg.yaw) * 180.0 / 3.141592653589793

    def _on_imu(self, msg: Imu) -> None:
        with self._lock:
            self._imu_count += 1
            self._imu_ts = float(msg.ts)
            self._imu_gyro = (
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            )
            self._imu_accel = (
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            )

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            self._joint_count += 1
            self._joint_ts = float(msg.ts)
            self._joint_name_count = len(msg.name)

    def _record_image(self, stats: _ImageStats, msg: Image) -> None:
        with self._lock:
            stats.count += 1
            stats.width = int(msg.width)
            stats.height = int(msg.height)
            stats.ts = float(msg.ts)
            if self.config.include_image_metrics:
                stats.brightness = float(msg.brightness)
                stats.sharpness = float(msg.sharpness)

    def _status_loop(self) -> None:
        interval = max(float(self.config.status_interval_s), 0.2)
        while not self._stop_event.wait(interval):
            now = time.time()
            now_monotonic = time.monotonic()
            with self._lock:
                status = self._status_snapshot_locked(now=now)
                dt = max(now_monotonic - self._last_rate_sample_monotonic, interval * 0.5, 1e-3)
                status['rates_hz'] = {
                    'primary': self._compute_rate_hz(status['primary']['count'], self._last_rate_sample_counts['primary'], dt),
                    'companion': self._compute_rate_hz(status['companion']['count'], self._last_rate_sample_counts['companion'], dt),
                    'bottom': self._compute_rate_hz(status['bottom']['count'], self._last_rate_sample_counts['bottom'], dt),
                    'odom': self._compute_rate_hz(status['odom']['count'], self._last_rate_sample_counts['odom'], dt),
                    'imu': self._compute_rate_hz(status['imu']['count'], self._last_rate_sample_counts['imu'], dt),
                    'joint': self._compute_rate_hz(status['joint_state']['count'], self._last_rate_sample_counts['joint'], dt),
                }
                self._last_rate_sample_counts = {
                    'primary': status['primary']['count'],
                    'companion': status['companion']['count'],
                    'bottom': status['bottom']['count'],
                    'odom': status['odom']['count'],
                    'imu': status['imu']['count'],
                    'joint': status['joint_state']['count'],
                }
                self._last_rate_sample_monotonic = now_monotonic
            logger.info(self._format_status(status))

    def _compute_rate_hz(self, current_count: int, previous_count: int, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        delta = max(0, int(current_count) - int(previous_count))
        return delta / dt

    def _status_snapshot_locked(self, now: float) -> dict[str, Any]:
        return {
            'primary': self._image_snapshot(self._primary, now),
            'companion': self._image_snapshot(self._companion, now),
            'bottom': self._image_snapshot(self._bottom, now),
            'camera_info_age_s': self._age(now, self._camera_info_ts),
            'odom': {
                'count': self._odom_count,
                'age_s': self._age(now, self._odom_ts),
                'xyz': self._odom_xyz,
                'yaw_deg': self._odom_yaw_deg,
            },
            'imu': {
                'count': self._imu_count,
                'age_s': self._age(now, self._imu_ts),
                'gyro': self._imu_gyro,
                'accel': self._imu_accel,
            },
            'joint_state': {
                'count': self._joint_count,
                'age_s': self._age(now, self._joint_ts),
                'joint_names': self._joint_name_count,
            },
        }

    def _image_snapshot(self, stats: _ImageStats, now: float) -> dict[str, Any]:
        return {
            'count': stats.count,
            'size': (stats.width, stats.height),
            'age_s': self._age(now, stats.ts),
            'brightness': stats.brightness,
            'sharpness': stats.sharpness,
        }

    def _age(self, now: float, ts: float) -> float | None:
        if ts <= 0.0:
            return None
        return max(0.0, now - ts)

    def _fresh_label(self, age_s: float | None) -> str:
        if age_s is None:
            return 'missing'
        if age_s <= float(self.config.stale_after_s):
            return 'fresh'
        return 'stale'

    def _fmt_image(self, name: str, entry: dict[str, Any], rate_hz: float | None) -> str:
        width, height = entry['size']
        age_s = entry['age_s']
        label = self._fresh_label(age_s)
        rate_text = f"{rate_hz:.1f}Hz" if rate_hz is not None else '?'
        size_text = f"{width}x{height}" if width > 0 and height > 0 else 'none'
        metrics = ''
        if self.config.include_image_metrics and age_s is not None:
            metrics = f" bright={entry['brightness']:.2f} sharp={entry['sharpness']:.2f}"
        age_text = 'n/a' if age_s is None else f"{age_s:.2f}s"
        return f"{name}={label}@{rate_text} age={age_text} size={size_text}{metrics}"

    def _format_status(self, status: dict[str, Any]) -> str:
        rates = status.get('rates_hz', {})
        primary = self._fmt_image('primary', status['primary'], rates.get('primary'))
        companion = self._fmt_image('companion', status['companion'], rates.get('companion'))
        bottom = self._fmt_image('bottom', status['bottom'], rates.get('bottom'))

        odom = status['odom']
        odom_age = 'n/a' if odom['age_s'] is None else f"{odom['age_s']:.2f}s"
        odom_fresh = self._fresh_label(odom['age_s'])

        imu = status['imu']
        imu_age = 'n/a' if imu['age_s'] is None else f"{imu['age_s']:.2f}s"
        imu_fresh = self._fresh_label(imu['age_s'])

        joint = status['joint_state']
        joint_age = 'n/a' if joint['age_s'] is None else f"{joint['age_s']:.2f}s"
        joint_fresh = self._fresh_label(joint['age_s'])

        return (
            f"{self.config.log_prefix}: "
            f"{primary} | {companion} | {bottom} | "
            f"odom={odom_fresh}@{rates.get('odom', 0.0):.1f}Hz age={odom_age} "
            f"xyz=({odom['xyz'][0]:.2f},{odom['xyz'][1]:.2f},{odom['xyz'][2]:.2f}) yaw={odom['yaw_deg']:.1f}deg | "
            f"imu={imu_fresh}@{rates.get('imu', 0.0):.1f}Hz age={imu_age} "
            f"gyro=({imu['gyro'][0]:.3f},{imu['gyro'][1]:.3f},{imu['gyro'][2]:.3f}) "
            f"accel=({imu['accel'][0]:.2f},{imu['accel'][1]:.2f},{imu['accel'][2]:.2f}) | "
            f"joint_state={joint_fresh}@{rates.get('joint', 0.0):.1f}Hz age={joint_age} joints={joint['joint_names']} | "
            f"camera_info_age={'n/a' if status['camera_info_age_s'] is None else format(status['camera_info_age_s'], '.2f') + 's'}"
        )
