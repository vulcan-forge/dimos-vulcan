from __future__ import annotations

import time
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

# Console rate-limit: at most one of each tagged line per this many seconds.
_DBG_MIN_INTERVAL_S = 2.0


class SourcceyLidarSafetyGateConfig(ModuleConfig):
    forward_angle_deg: float = 180.0
    # Ignore returns inside the robot's own footprint (body / arms / lidar mount).
    min_distance_m: float = 0.20
    # Stop when a real obstacle is ~0.28 m ahead (zone: [min_distance, ~0.36] m).
    tripwire_distance_m: float = 0.28
    # Half-width of the stop box. MUST stay narrower than the robot's own arms,
    # which sit ~0.26-0.40 m off-center within the depth band and otherwise read
    # as ~25 permanent "blocking" points that veto all forward motion (the arms
    # are outside the explorer's 35-deg forward cone, so the explorer drives while
    # the gate vetoes — robot never moves). 0.20 m keeps a tight central collision
    # guard for the base while ignoring the arms.
    tripwire_half_width_m: float = 0.20
    tripwire_thickness_m: float = 0.16
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
        self._dbg_last_wall_ts: dict[str, float] = {}

    def _dbg(self, tag: str, **fields: Any) -> None:
        """Log ``[safety_gate.<tag>]`` at most once per ``_DBG_MIN_INTERVAL_S`` per tag."""
        now = time.time()
        if (now - self._dbg_last_wall_ts.get(tag, 0.0)) < _DBG_MIN_INTERVAL_S:
            return
        self._dbg_last_wall_ts[tag] = now
        logger.info(f"[safety_gate.{tag}]", **fields)

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
        self._dbg(
            "scan",
            blocked=bool(self._latest_state.blocked),
            blocking_points=int(self._latest_state.blocking_points),
            threshold_points=int(self._latest_state.threshold_points),
            nearest_blocking_distance_m=self._latest_state.nearest_blocking_distance_m,
        )

    def _on_cmd_vel(self, cmd_vel: Twist) -> None:
        if not self._latest_state.blocked or cmd_vel.linear.x <= 0.0:
            self._dbg(
                "passthrough",
                blocked=self._latest_state.blocked,
                linear_x=round(float(cmd_vel.linear.x), 4),
                linear_y=round(float(cmd_vel.linear.y), 4),
                angular_z=round(float(cmd_vel.angular.z), 4),
            )
            self.cmd_vel.publish(cmd_vel)
            return

        now = time.time()
        if (now - self._dbg_last_wall_ts.get("blocked_msg", 0.0)) >= _DBG_MIN_INTERVAL_S:
            self._dbg_last_wall_ts["blocked_msg"] = now
            logger.warning(
                ">>> CURRENTLY SAFETY BLOCKED <<< forward motion vetoed, robot should turn",
                blocking_points=int(self._latest_state.blocking_points),
                nearest_blocking_distance_m=self._latest_state.nearest_blocking_distance_m,
                requested_linear_x=round(float(cmd_vel.linear.x), 4),
            )
        self.cmd_vel.publish(
            Twist(
                linear=Vector3(0.0, cmd_vel.linear.y, cmd_vel.linear.z),
                angular=Vector3(cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z),
            )
        )
