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
from .run_trace import trace_event

logger = setup_logger()

# Console rate-limit: at most one of each tagged line per this many seconds.
_DBG_MIN_INTERVAL_S = 2.0


class SourcceyLidarSafetyGateConfig(ModuleConfig):
    forward_angle_deg: float = 270.0
    # Ignore returns inside the robot's own footprint (body / arms / lidar mount).
    min_distance_m: float = 0.03
    # Stop when a real obstacle is ~0.28 m ahead (zone: [min_distance, ~0.36] m).
    tripwire_distance_m: float = 0.14
    # Half-width of the stop box. MUST stay narrower than the robot's own arms,
    # which sit ~0.26-0.40 m off-center within the depth band and otherwise read
    # as ~25 permanent "blocking" points that veto all forward motion (the arms
    # are outside the explorer's 35-deg forward cone, so the explorer drives while
    # the gate vetoes — robot never moves). 0.20 m keeps a tight central collision
    # guard for the base while ignoring the arms.
    tripwire_half_width_m: float = 0.28
    tripwire_thickness_m: float = 0.12
    min_points_to_trigger: int = 8
    min_confidence: int = 0
    debug_enabled: bool = False
    debug_min_interval_s: float = _DBG_MIN_INTERVAL_S


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
        self._last_traced_blocked: bool | None = None

    def _dbg(self, tag: str, **fields: Any) -> None:
        """Rate-limited ``[safety_gate.<tag>]`` debug logging."""
        if not bool(self.config.debug_enabled):
            return
        now = time.time()
        min_interval_s = max(float(self.config.debug_min_interval_s), 0.0)
        if (now - self._dbg_last_wall_ts.get(tag, 0.0)) < min_interval_s:
            return
        self._dbg_last_wall_ts[tag] = now
        logger.info(f"[safety_gate.{tag}]", **fields)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))
        self.register_disposable(Disposable(self.cmd_vel_in.subscribe(self._on_cmd_vel)))
        trace_event(
            "lidar_safety_gate",
            "start",
            forward_angle_deg=float(self.config.forward_angle_deg),
            min_distance_m=float(self.config.min_distance_m),
            tripwire_distance_m=float(self.config.tripwire_distance_m),
            tripwire_half_width_m=float(self.config.tripwire_half_width_m),
            tripwire_thickness_m=float(self.config.tripwire_thickness_m),
            min_points_to_trigger=int(self.config.min_points_to_trigger),
        )

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
        blocked = bool(self._latest_state.blocked)
        if self._last_traced_blocked is None or blocked != self._last_traced_blocked:
            trace_event(
                "lidar_safety_gate",
                "stop_zone_changed",
                blocked=blocked,
                blocking_points=int(self._latest_state.blocking_points),
                threshold_points=int(self._latest_state.threshold_points),
                nearest_blocking_distance_m=self._latest_state.nearest_blocking_distance_m,
            )
            self._last_traced_blocked = blocked
        self._dbg(
            "scan",
            blocked=blocked,
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
        trace_event(
            "lidar_safety_gate",
            "forward_motion_veto",
            blocking_points=int(self._latest_state.blocking_points),
            nearest_blocking_distance_m=self._latest_state.nearest_blocking_distance_m,
            requested_linear_x=round(float(cmd_vel.linear.x), 4),
            requested_linear_y=round(float(cmd_vel.linear.y), 4),
            requested_angular_z=round(float(cmd_vel.angular.z), 4),
        )
        self.cmd_vel.publish(
            Twist(
                linear=Vector3(0.0, cmd_vel.linear.y, cmd_vel.linear.z),
                angular=Vector3(cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z),
            )
        )
