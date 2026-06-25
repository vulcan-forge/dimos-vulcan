from __future__ import annotations

from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

from .lidar_geometry import detect_stop_zone
from .lidar_types import PlanarLidarScan, StopZoneConfig, StopZoneState

logger = setup_logger()


class SourcceyLidarSafetyGateConfig(ModuleConfig):
    forward_angle_deg: float = 180.0
    min_distance_m: float = 0.12
    tripwire_distance_m: float = 0.105
    tripwire_half_width_m: float = 0.525
    tripwire_thickness_m: float = 0.12
    min_points_to_trigger: int = 6
    min_confidence: int = 0


class SourcceyLidarSafetyGate(Module):
    config: SourcceyLidarSafetyGateConfig

    scan: In[PlanarLidarScan]
    cmd_vel_in: In[Twist]

    cmd_vel: Out[Twist]
    stop_zone: Out[StopZoneState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_state = StopZoneState(
            ts=0.0,
            blocked=False,
            blocking_points=0,
            threshold_points=max(int(self.config.min_points_to_trigger), 1),
            nearest_blocking_distance_m=None,
        )

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))
        self.register_disposable(Disposable(self.cmd_vel_in.subscribe(self._on_cmd_vel)))

    def _stop_config(self) -> StopZoneConfig:
        return StopZoneConfig(
            forward_angle_deg=float(self.config.forward_angle_deg),
            min_distance_m=float(self.config.min_distance_m),
            tripwire_distance_m=float(self.config.tripwire_distance_m),
            tripwire_half_width_m=float(self.config.tripwire_half_width_m),
            tripwire_thickness_m=float(self.config.tripwire_thickness_m),
            min_points_to_trigger=int(self.config.min_points_to_trigger),
            min_confidence=int(self.config.min_confidence),
        )

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        self._latest_state = detect_stop_zone(scan, cfg=self._stop_config())
        self.stop_zone.publish(self._latest_state)

    def _on_cmd_vel(self, cmd_vel: Twist) -> None:
        if not self._latest_state.blocked or cmd_vel.linear.x <= 0.0:
            logger.info(
                "LidarSafetyGate pass-through",
                blocked=self._latest_state.blocked,
                linear_x=round(float(cmd_vel.linear.x), 4),
                linear_y=round(float(cmd_vel.linear.y), 4),
                angular_z=round(float(cmd_vel.angular.z), 4),
            )
            self.cmd_vel.publish(cmd_vel)
            return

        logger.warning(
            "LidarSafetyGate blocked forward motion",
            blocking_points=int(self._latest_state.blocking_points),
            nearest_blocking_distance_m=self._latest_state.nearest_blocking_distance_m,
            linear_x=round(float(cmd_vel.linear.x), 4),
            linear_y=round(float(cmd_vel.linear.y), 4),
            angular_z=round(float(cmd_vel.angular.z), 4),
        )
        self.cmd_vel.publish(
            Twist(
                linear=Vector3(0.0, cmd_vel.linear.y, cmd_vel.linear.z),
                angular=Vector3(cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z),
            )
        )
