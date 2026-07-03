# Copyright 2025-2026 Dimensional Inc.
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

"""Autonomous "wander and map" behavior for the Sourccey robot.

The :class:`SourcceyWanderExplorer` drives the robot forward, watches the
planar LiDAR for walls/obstacles ahead, and — when blocked or when the area
directly ahead has already been mapped — rotates toward the most open *and
least explored* direction before continuing. It keeps doing this until the
occupancy grid shows there are no meaningful frontiers left (the room is
mapped), then stops.

Design notes
------------
* This module is purely *reactive* and self-contained. It emits ``cmd_vel``
  which is expected to pass through :class:`SourcceyLidarSafetyGate` (the
  hard, last-resort collision stop) before reaching the robot. The wander
  module reacts to walls well before the safety gate trips, and it also
  subscribes to the gate's ``stop_zone`` so that if the gate *does* veto
  forward motion the explorer turns away instead of grinding into the veto.
* "Intelligence" comes from the occupancy grid: candidate headings are scored
  by how clear they are *and* how much unexplored (``UNKNOWN``) space lies
  ahead, so the robot biases its turns toward unmapped territory.
* The sign that maps ``angular.z`` to a change in reported yaw is *calibrated
  online* (Sourccey inverts yaw sign for state estimation), so the turn loop
  self-corrects and always rotates toward its target heading.
* Velocity units: ``x.vel`` / ``theta.vel`` on the Sourccey base are normalized
  wheel throttles in [-1, 1], NOT m/s — see ``sourccey.py``. A heavy base needs
  ~0.8+ throttle to *start* from a standstill, so DRIVE uses a brief full-throttle
  "kick" to break static friction, then settles to a gentler cruise.
"""

from __future__ import annotations

from enum import Enum
import math
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool
import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.utils.logging_config import setup_logger

from .lidar_geometry import normalize_angle_deg
from .lidar_types import PlanarLidarScan, StopZoneState
from .run_trace import start_new_run, trace_event

logger = setup_logger()

# Console rate-limit: at most one of each tagged debug line per this many seconds.
_DBG_MIN_INTERVAL_S = 2.0


class WanderState(str, Enum):
    IDLE = "idle"
    DRIVE = "drive"
    TURN = "turn"
    SETTLE = "settle"
    DONE = "done"


class SourcceyWanderExplorerConfig(ModuleConfig):
    # --- LiDAR scan interpretation (must match the point-cloud adapter) ---
    forward_angle_deg: float = 270.0
    valid_angle_half_width_deg: float = 90.0
    invert_lateral_axis: bool = False
    min_confidence: int = 5
    scan_max_distance_m: float = 8.0
    # Ignore returns closer than this: they are the robot's own body / arms /
    # lidar housing, not real obstacles. The safety gate handles true close range.
    # Matches the point-cloud adapter's min_range_m so map and reactive views agree.
    scan_min_range_m: float = 0.22

    # --- Driving speeds / loop rate ---
    # x.vel/theta.vel are normalized wheel throttles in [-1, 1] (not m/s/rad-s).
    # Teleop only ever drives the base at 0.8..1.0; below ~0.7 a standstill base
    # won't break static friction. Cruise is deliberately ~half the kick: once
    # already rolling, kinetic friction is far lower, so a gentler steady throttle
    # keeps it moving even though it could never *start* at that value.
    # Cruise must stay high enough to actually keep the heavy base rolling — too
    # low and it stalls between nudges, which reads as long pauses + creeping.
    cruise_speed_m_s: float = 0.40
    turn_speed_rad_s: float = 0.9
    control_hz: float = 12.0
    # Static-friction kick: on entering DRIVE from a dead stop, command
    # kick_speed_m_s for kick_duration_s to physically break the heavy base
    # loose (it cannot start at a gentle throttle), then settle to cruise.
    kick_speed_m_s: float = 1.0
    kick_duration_s: float = 0.8
    # While already rolling, a frequent firm push keeps it gliding so it never
    # stalls into a long pause. Set re_kick_period_s=0 to disable.
    re_kick_speed_m_s: float = 0.85
    re_kick_duration_s: float = 0.7
    re_kick_period_s: float = 2.0

    # --- Reactive obstacle handling (live scan, robot frame) ---
    turn_trigger_distance_m: float = 0.32
    resume_clearance_m: float = 0.42
    corridor_half_width_m: float = 0.28
    forward_cone_half_deg: float = 35.0

    # --- Velocity-based lookahead ("can I stop before I hit it?") ---
    # Effective turn-trigger distance grows with the speed about to be commanded:
    #   trigger = clamp(margin + speed_mps * reaction_s, turn_trigger, resume - 0.1)
    # so the faster it intends to move, the earlier it decides to turn. Kept below
    # resume_clearance_m to preserve drive<->turn hysteresis.
    lookahead_enabled: bool = True
    throttle_to_mps: float = 0.5  # full throttle (1.0) ~= 0.5 m/s of real travel
    reaction_time_s: float = 1.2
    lookahead_margin_m: float = 0.2

    # --- Intelligent heading selection (occupancy grid) ---
    occupancy_threshold: int = 60
    heading_candidates: int = 36
    heading_lookahead_m: float = 3.0
    min_heading_clearance_m: float = 0.5
    clearance_weight: float = 1.0
    # Bias strongly toward unexplored space and only lightly penalize having to
    # turn far, so the robot is willing to swing around to a barely-mapped area
    # rather than keep nudging toward the nearest open (already-seen) direction.
    explore_weight: float = 3.5
    turn_penalty_weight: float = 0.25

    # --- Avoid re-committing to directions just explored ---
    # The robot remembers the headings it recently drove off in and penalizes
    # picking similar ones again, so it spreads into unmapped parts of the room
    # instead of oscillating back into the same already-explored direction.
    revisit_avoid_enabled: bool = True
    revisit_memory_size: int = 6
    revisit_penalty_weight: float = 6.0
    revisit_sigma_deg: float = 45.0
    # Two committed headings within this angle are treated as "the same way".
    revisit_dedup_deg: float = 25.0

    # --- Turn control ---
    # A turn now completes when the robot has actually rotated to the chosen
    # heading (within align_tolerance), NOT merely when the path ahead opens up.
    # Because localized_pose is frozen by the mapping adapter during turns, yaw
    # progress is estimated by integrating the commanded angular rate:
    #   est_rate (rad/s) = turn_speed_rad_s * turn_rate_est_scale
    # and resynced to the real pose whenever a fresh one arrives.
    align_tolerance_deg: float = 18.0
    # Always rotate at least this much before resuming (prevents bolting forward
    # after a tiny turn, which made it keep going the same way).
    min_turn_deg: float = 80.0
    turn_rate_est_scale: float = 1.0
    max_turn_s: float = 6.0
    heading_refresh_s: float = 1.8
    turn_burst_s: float = 0.35
    turn_settle_pause_s: float = 0.65
    turn_burst_mapping_enabled: bool = True
    turn_burst_mapping_min_wait_s: float = 1.0
    turn_burst_mapping_max_wait_s: float = 3.5
    drive_burst_s: float = 0.65
    drive_burst_mapping_enabled: bool = True
    drive_burst_mapping_min_wait_s: float = 1.0
    drive_burst_mapping_max_wait_s: float = 3.5
    # After a turn completes, stop and wait for the offboard mapper to accept a
    # fresh stationary scan before driving again. This trades speed for cleaner
    # map stitching and avoids smearing a just-turned snapshot onto stale pose.
    post_turn_mapping_enabled: bool = True
    post_turn_mapping_min_wait_s: float = 0.8
    post_turn_mapping_max_wait_s: float = 2.5
    post_turn_require_pose_update: bool = True
    post_turn_require_costmap_update: bool = True
    post_turn_require_scan_update: bool = True
    post_turn_require_localized_commit: bool = False
    settle_min_pose_updates: int = 2
    settle_min_costmap_updates: int = 2
    settle_min_scan_updates: int = 3
    # --- Turn-only mapping test mode ---
    # For debugging map stitching, bypass all forward motion and simply rotate in
    # fixed increments, stop, and wait for the mapper to commit a stationary
    # snapshot before rotating again.
    turn_only_snapshot_mode: bool = False
    turn_only_step_deg: float = 45.0

    # --- Redirect when the area straight ahead is already mapped ---
    redirect_when_explored: bool = True
    redirect_after_s: float = 6.0
    min_explore_fraction: float = 0.05

    # --- Completion ("the room is mapped") ---
    completion_enabled: bool = True
    frontier_done_threshold: int = 8
    min_free_cells_for_done: int = 400
    done_after_s: float = 20.0

    # --- Lifecycle ---
    auto_start: bool = False
    start_delay_s: float = 3.0
    pose_stale_after_s: float = 2.0
    startup_min_free_cells: int = 120
    startup_stabilization_s: float = 1.5
    startup_require_costmap: bool = True
    log_every_ticks: int = 30
    heartbeat_s: float = 3.0
    debug_enabled: bool = False
    debug_min_interval_s: float = _DBG_MIN_INTERVAL_S


# ---------------------------------------------------------------------------
# Pure decision helpers (no module state — unit tested directly)
# ---------------------------------------------------------------------------


def angle_diff(target_rad: float, source_rad: float) -> float:
    """Smallest signed angle (radians) to rotate from ``source`` to ``target``."""
    return math.atan2(math.sin(target_rad - source_rad), math.cos(target_rad - source_rad))


def scan_to_local_xy(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_confidence: int,
    min_distance_m: float = 0.0,
) -> np.ndarray:
    """Convert a planar scan to robot-frame points ``(forward_x, lateral_y)``.

    Matches the convention used by ``SourcceyLidarPointCloudAdapter`` so that
    "forward" and "left/right" agree with what is fed into the map. Returns at
    most one point per beam; ``min_distance_m`` drops near-field self-hits.
    """
    out: list[tuple[float, float]] = []
    for angle_deg, distance_m, confidence in zip(
        scan.angles_deg, scan.distances_m, scan.confidences, strict=False
    ):
        if int(confidence) < int(min_confidence):
            continue
        distance = float(distance_m)
        if not math.isfinite(distance) or distance <= 0.0 or distance > float(max_distance_m):
            continue
        if distance < float(min_distance_m):
            continue
        delta_deg = normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        if abs(delta_deg) > float(valid_angle_half_width_deg):
            continue
        theta = math.radians(delta_deg)
        forward_m = distance * math.cos(theta)
        lateral_m = distance * math.sin(theta)
        if invert_lateral_axis:
            lateral_m = -lateral_m
        out.append((forward_m, lateral_m))
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def forward_clearance(
    points_local_xy: np.ndarray,
    *,
    corridor_half_width_m: float,
    forward_cone_half_deg: float,
    max_distance_m: float,
) -> float:
    """Distance to the nearest obstacle in the corridor straight ahead.

    A point counts if it is in front of the robot, within the lateral corridor
    (robot width + margin) and inside the forward cone. Returns ``max_distance``
    when nothing is ahead.
    """
    if points_local_xy.shape[0] == 0:
        return float(max_distance_m)
    fwd = points_local_xy[:, 0]
    lat = points_local_xy[:, 1]
    cone_rad = math.radians(float(forward_cone_half_deg))
    bearing = np.arctan2(lat, fwd)
    mask = (
        (fwd > 0.0) & (np.abs(lat) <= float(corridor_half_width_m)) & (np.abs(bearing) <= cone_rad)
    )
    if not bool(np.any(mask)):
        return float(max_distance_m)
    return float(min(float(np.min(fwd[mask])), float(max_distance_m)))


def costmap_raymarch(
    grid: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    resolution: float,
    start_x: float,
    start_y: float,
    heading_rad: float,
    lookahead_m: float,
    occupancy_threshold: int,
) -> tuple[float, float]:
    """March a ray across the occupancy grid from ``start`` along ``heading``.

    Returns ``(clearance_m, unknown_fraction)`` where ``clearance_m`` is the
    distance until the first occupied cell (capped at ``lookahead_m``) and
    ``unknown_fraction`` is the share of traversed cells that are unexplored.
    """
    if grid.size == 0 or resolution <= 0.0:
        return float(lookahead_m), 1.0

    height, width = grid.shape
    step = float(resolution)
    n_steps = max(int(float(lookahead_m) / step), 1)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    unknown = 0
    traversed = 0
    clearance = float(lookahead_m)
    for i in range(1, n_steps + 1):
        dist = i * step
        wx = start_x + dist * cos_h
        wy = start_y + dist * sin_h
        gx = int((wx - origin_x) / step)
        gy = int((wy - origin_y) / step)
        if not (0 <= gx < width and 0 <= gy < height):
            # Off the known map: treat as unexplored open space.
            unknown += 1
            traversed += 1
            continue
        value = int(grid[gy, gx])
        traversed += 1
        if value == int(CostValues.UNKNOWN):
            unknown += 1
        elif value >= int(occupancy_threshold):
            clearance = dist
            break
    unknown_fraction = (unknown / traversed) if traversed > 0 else 1.0
    return clearance, unknown_fraction


def sector_clearance(
    points_local_xy: np.ndarray,
    *,
    center_bearing_rad: float,
    sector_half_width_rad: float,
    max_distance_m: float,
) -> float:
    """Nearest obstacle range within an angular sector of the live scan.

    Bearings are in the robot frame (0 = straight ahead). Returns
    ``max_distance`` when the sector is empty.
    """
    if points_local_xy.shape[0] == 0:
        return float(max_distance_m)
    fwd = points_local_xy[:, 0]
    lat = points_local_xy[:, 1]
    bearing = np.arctan2(lat, fwd)
    delta = np.arctan2(np.sin(bearing - center_bearing_rad), np.cos(bearing - center_bearing_rad))
    mask = np.abs(delta) <= float(sector_half_width_rad)
    if not bool(np.any(mask)):
        return float(max_distance_m)
    ranges = np.hypot(fwd[mask], lat[mask])
    return float(min(float(np.min(ranges)), float(max_distance_m)))


def stopping_trigger_distance(
    *,
    planned_throttle: float,
    throttle_to_mps: float,
    reaction_time_s: float,
    margin_m: float,
    floor_m: float,
    ceiling_m: float,
) -> float:
    """Velocity-scaled distance at which to start turning before an obstacle.

    Converts the throttle about to be commanded into an estimated real speed and
    returns ``margin + speed * reaction_time``, clamped to ``[floor, ceiling]``.
    Faster intended motion -> earlier turn. The ceiling keeps it below the
    resume-clearance so drive<->turn hysteresis is preserved.
    """
    speed_mps = abs(float(planned_throttle)) * float(throttle_to_mps)
    raw = float(margin_m) + speed_mps * float(reaction_time_s)
    return float(min(max(raw, float(floor_m)), float(ceiling_m)))


def select_world_heading(
    grid: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    resolution: float,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    n_candidates: int,
    lookahead_m: float,
    occupancy_threshold: int,
    min_clearance_m: float,
    clearance_weight: float,
    explore_weight: float,
    turn_penalty_weight: float,
    scan_points_local: np.ndarray | None = None,
    scan_fov_half_deg: float = 90.0,
    sector_half_deg: float = 12.0,
    avoid_headings: list[float] | None = None,
    avoid_penalty_weight: float = 0.0,
    avoid_sigma_rad: float = 0.785,
) -> tuple[float, float, float]:
    """Pick the best world-frame heading to drive toward.

    Scores headings spanning the full circle by clearance, unexplored gain and
    a small penalty on how far the robot must rotate. When ``scan_points_local``
    is supplied, candidate headings inside the LiDAR field of view also respect
    the live scan, so a direction the map thinks is clear but the LiDAR sees
    blocked is not chosen. ``avoid_headings`` (world-frame radians) are directions
    the robot recently committed to; candidates near them are penalized (Gaussian,
    width ``avoid_sigma_rad``) so it spreads toward unexplored areas instead of
    re-picking a direction it already went. Returns
    ``(best_heading_rad, best_unknown_fraction, best_clearance_m)``.
    """
    n = max(int(n_candidates), 4)
    best_heading = float(robot_yaw)
    best_score = -math.inf
    best_unknown = 0.0
    best_clearance = 0.0
    # Fallback if every candidate is below the clearance floor: keep the most open
    # direction that we have not recently explored (revisit-penalized openness).
    fallback_heading = float(robot_yaw)
    fallback_metric = -math.inf
    fov_half_rad = math.radians(float(scan_fov_half_deg))
    sector_half_rad = math.radians(float(sector_half_deg))
    two_sigma_sq = 2.0 * float(avoid_sigma_rad) * float(avoid_sigma_rad)

    for i in range(n):
        rel = -math.pi + (2.0 * math.pi * i) / n
        heading = math.atan2(math.sin(robot_yaw + rel), math.cos(robot_yaw + rel))
        clearance, unknown_fraction = costmap_raymarch(
            grid,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            start_x=robot_x,
            start_y=robot_y,
            heading_rad=heading,
            lookahead_m=lookahead_m,
            occupancy_threshold=occupancy_threshold,
        )
        if scan_points_local is not None and abs(rel) <= fov_half_rad:
            clearance = min(
                clearance,
                sector_clearance(
                    scan_points_local,
                    center_bearing_rad=rel,
                    sector_half_width_rad=sector_half_rad,
                    max_distance_m=float(lookahead_m),
                ),
            )

        avoid_penalty = 0.0
        if avoid_headings and float(avoid_penalty_weight) > 0.0 and two_sigma_sq > 0.0:
            for prior in avoid_headings:
                delta = angle_diff(heading, float(prior))
                avoid_penalty += math.exp(-(delta * delta) / two_sigma_sq)
            avoid_penalty *= float(avoid_penalty_weight)

        # Fallback metric: most open, minus how much we've already been this way.
        fallback_metric_candidate = clearance - avoid_penalty
        if fallback_metric_candidate > fallback_metric:
            fallback_metric = fallback_metric_candidate
            fallback_heading = heading

        if clearance < float(min_clearance_m):
            continue

        score = (
            float(clearance_weight) * min(clearance, float(lookahead_m))
            + float(explore_weight) * unknown_fraction * float(lookahead_m)
            - float(turn_penalty_weight) * abs(rel)
            - avoid_penalty
        )
        if score > best_score:
            best_score = score
            best_heading = heading
            best_unknown = unknown_fraction
            best_clearance = clearance

    if best_score == -math.inf:
        return fallback_heading, 0.0, 0.0
    return best_heading, best_unknown, best_clearance


def frontier_cell_count(grid: np.ndarray) -> int:
    """Count free cells that border unexplored space (exploration frontier)."""
    if grid.size == 0:
        return 0
    free = grid == int(CostValues.FREE)
    unknown = grid == int(CostValues.UNKNOWN)
    adjacent_unknown = np.zeros_like(free)
    adjacent_unknown[:-1, :] |= unknown[1:, :]
    adjacent_unknown[1:, :] |= unknown[:-1, :]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    return int(np.sum(free & adjacent_unknown))


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class SourcceyWanderExplorer(Module):
    """Reactive, costmap-aware autonomous explorer for Sourccey."""

    config: SourcceyWanderExplorerConfig

    scan: In[PlanarLidarScan]
    odom: In[PoseStamped]
    localized_pose: In[PoseStamped]
    global_costmap: In[OccupancyGrid]
    explore_cmd: In[Bool]
    stop_explore_cmd: In[Bool]
    # Fed by SourcceyLidarSafetyGate: when the hard collision guard trips, the
    # explorer reacts by turning away instead of grinding into the veto forever.
    stop_zone: In[StopZoneState]

    cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._control_thread: threading.Thread | None = None

        self._latest_scan: PlanarLidarScan | None = None
        self._latest_pose: PoseStamped | None = None
        self._latest_localized_pose: PoseStamped | None = None
        self._latest_costmap: OccupancyGrid | None = None
        self._latest_scan_seq = 0
        self._latest_pose_seq = 0
        self._latest_localized_pose_seq = 0
        self._latest_costmap_seq = 0
        self._safety_blocked = False
        self._last_stop_zone_wall_ts = 0.0
        self._last_pose_wall_ts = 0.0
        self._last_scan_wall_ts = 0.0
        self._last_heartbeat_wall_ts = 0.0
        self._frontier_count = 0
        self._free_cells = 0

        self._state = WanderState.IDLE
        self._target_heading: float | None = None
        # World-frame headings (radians) the robot recently committed to driving,
        # newest last. Used to penalize re-picking an already-explored direction.
        self._recent_headings: list[float] = []
        # Persistent spin direction used when pose is unavailable (reactive bounce).
        self._spin_dir = 1.0
        self._turn_started_wall_ts = 0.0
        self._heading_refreshed_wall_ts = 0.0
        self._turn_burst_started_wall_ts = 0.0
        self._turn_pause_until_wall_ts = 0.0
        # Turn-progress tracking (pose is frozen mid-turn, so estimate yaw by
        # integrating the commanded angular rate; resync to pose when fresh).
        self._turn_est_yaw: float | None = None
        self._turn_accum_rad = 0.0
        self._turn_required_rad = math.radians(float(self.config.min_turn_deg))
        self._turn_last_tick_wall_ts = 0.0
        self._settle_started_wall_ts = 0.0
        self._settle_scan_seq_baseline = 0
        self._settle_pose_seq_baseline = 0
        self._settle_localized_pose_seq_baseline = 0
        self._settle_costmap_seq_baseline = 0
        self._settle_resume_state = WanderState.DRIVE
        self._settle_reason = "startup"
        self._settle_min_wait_s = 0.0
        self._settle_max_wait_s = 0.0
        self._drive_segment_started_wall_ts = 0.0
        self._explored_since_wall_ts: float | None = None
        self._done_since_wall_ts: float | None = None
        self._started_wall_ts = 0.0
        self._startup_ready_since_wall_ts: float | None = None
        # Startup stabilization is a ONE-TIME gate. Once satisfied, it must never
        # re-trigger: during turns the mapping adapter holds pose updates, so pose
        # legitimately goes stale and would otherwise reset the gate forever,
        # deadlocking the robot into "stabilizing" while it tries to turn.
        self._startup_complete = False
        # Static-friction kick: while now < this wall-clock ts, drive at _kick_speed.
        self._drive_kick_until_wall_ts = 0.0
        # Throttle to use during the current kick window (full breakaway vs gentle).
        self._kick_speed = 0.0
        # Next wall-clock ts at which to re-apply a gentle anti-stall push.
        self._next_rekick_wall_ts = 0.0
        # True while the last command was a forward drive (used to time the kick).
        self._was_driving = False

        # Online calibration of "+angular.z -> +yaw".
        self._yaw_cmd_sign = 1.0
        self._prev_yaw: float | None = None
        self._prev_yaw_wall_ts = 0.0
        self._prev_cmd_wz = 0.0
        self._tick = 0

        # Rate-limited debug logging: at most one message per tag per interval.
        self._dbg_last_wall_ts: dict[str, float] = {}

    def _dbg(self, tag: str, **fields: Any) -> None:
        """Rate-limited ``[wander.<tag>]`` debug logging.

        Enable via ``debug_enabled=True`` in the blueprint/config when chasing
        wander or mapping issues.
        """
        if not bool(self.config.debug_enabled):
            return
        now = time.time()
        min_interval_s = max(float(self.config.debug_min_interval_s), 0.0)
        if (now - self._dbg_last_wall_ts.get(tag, 0.0)) < min_interval_s:
            return
        self._dbg_last_wall_ts[tag] = now
        logger.info(f"[wander.{tag}]", **fields)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.localized_pose.subscribe(self._on_localized_pose)))
        self.register_disposable(Disposable(self.global_costmap.subscribe(self._on_costmap)))
        self.register_disposable(Disposable(self.explore_cmd.subscribe(self._on_explore_cmd)))
        self.register_disposable(
            Disposable(self.stop_explore_cmd.subscribe(self._on_stop_explore_cmd))
        )
        self.register_disposable(Disposable(self.stop_zone.subscribe(self._on_stop_zone)))

        self._started_wall_ts = time.time()
        if bool(self.config.auto_start):
            self._state = (
                WanderState.TURN
                if bool(self.config.turn_only_snapshot_mode)
                else WanderState.DRIVE
            )

        self._stop_event.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        session = start_new_run(
            component="wander_explorer",
            label="sourccey_lidar_mapping_offboard_explore",
        )
        logger.info(
            "SourcceyWanderExplorer control loop started",
            auto_start=bool(self.config.auto_start),
            initial_state=self._state.value,
            start_delay_s=float(self.config.start_delay_s),
        )
        trace_event(
            "wander_explorer",
            "start",
            auto_start=bool(self.config.auto_start),
            initial_state=self._state.value,
            start_delay_s=float(self.config.start_delay_s),
            turn_only_snapshot_mode=bool(self.config.turn_only_snapshot_mode),
            turn_only_step_deg=float(self.config.turn_only_step_deg),
            run_id=session["run_id"],
            run_dir=session["run_dir"],
        )

    @rpc
    def stop(self) -> None:
        trace_event("wander_explorer", "stop_requested", state=self._state.value)
        self._stop_event.set()
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        self._control_thread = None
        self._publish_stop()
        super().stop()

    # --- subscriptions ---

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        with self._lock:
            self._latest_scan = scan
            self._latest_scan_seq += 1
            self._last_scan_wall_ts = time.time()
        self._dbg("rx_scan", beams=len(getattr(scan, "angles_deg", []) or []))

    def _on_odom(self, pose: PoseStamped) -> None:
        with self._lock:
            self._latest_pose = pose
            self._last_pose_wall_ts = time.time()
            self._latest_pose_seq += 1
        self._dbg(
            "rx_odom",
            x=round(float(pose.x), 3),
            y=round(float(pose.y), 3),
            yaw_deg=round(math.degrees(float(pose.yaw)), 1),
        )

    def _on_localized_pose(self, pose: PoseStamped) -> None:
        with self._lock:
            self._latest_localized_pose = pose
            self._latest_localized_pose_seq += 1
        self._dbg(
            "rx_localized_pose",
            x=round(float(pose.x), 3),
            y=round(float(pose.y), 3),
            yaw_deg=round(math.degrees(float(pose.yaw)), 1),
        )

    def _on_stop_zone(self, state: StopZoneState) -> None:
        with self._lock:
            self._safety_blocked = bool(state.blocked)
            self._last_stop_zone_wall_ts = time.time()
        self._dbg(
            "rx_stop_zone",
            blocked=bool(state.blocked),
            blocking_points=int(state.blocking_points),
            nearest_blocking_distance_m=state.nearest_blocking_distance_m,
        )

    def _on_costmap(self, grid: OccupancyGrid) -> None:
        frontier = frontier_cell_count(grid.grid)
        free = int(np.sum(grid.grid == int(CostValues.FREE))) if grid.grid.size else 0
        with self._lock:
            self._latest_costmap = grid
            self._latest_costmap_seq += 1
            self._frontier_count = frontier
            self._free_cells = free
        self._dbg("rx_costmap", frontier_cells=frontier, free_cells=free, shape=tuple(grid.grid.shape))

    def _on_explore_cmd(self, msg: Bool) -> None:
        self._dbg("rx_explore_cmd", data=bool(msg.data))
        if not bool(msg.data):
            return
        with self._lock:
            self._state = WanderState.DRIVE
            self._target_heading = None
            self._explored_since_wall_ts = None
            self._done_since_wall_ts = None
            self._started_wall_ts = time.time()
        logger.info("SourcceyWanderExplorer exploration started via command")

    def _on_stop_explore_cmd(self, msg: Bool) -> None:
        self._dbg("rx_stop_explore_cmd", data=bool(msg.data))
        if not bool(msg.data):
            return
        with self._lock:
            self._state = WanderState.IDLE
        self._publish_stop()
        logger.info("SourcceyWanderExplorer exploration stopped via command")

    # --- control loop ---

    def _control_loop(self) -> None:
        period = 1.0 / max(float(self.config.control_hz), 1.0)
        while not self._stop_event.wait(period):
            try:
                self._control_tick()
            except Exception:
                logger.exception("SourcceyWanderExplorer control tick failed")

    def _control_tick(self) -> None:
        now = time.time()
        with self._lock:
            state = self._state
            scan = self._latest_scan
            pose = self._latest_pose
            costmap = self._latest_costmap
            frontier_count = self._frontier_count
            free_cells = self._free_cells
            scan_seq = self._latest_scan_seq
            pose_seq = self._latest_pose_seq
            costmap_seq = self._latest_costmap_seq
            last_pose_wall = self._last_pose_wall_ts
            last_scan_wall = self._last_scan_wall_ts
            safety_blocked_raw = self._safety_blocked
            last_stop_zone_wall = self._last_stop_zone_wall_ts

        # Only trust a recent stop-zone reading; a stale one shouldn't pin us.
        safety_blocked = bool(safety_blocked_raw) and (
            (now - last_stop_zone_wall) <= float(self.config.pose_stale_after_s)
        )

        pose_fresh = pose is not None and (now - last_pose_wall) <= float(
            self.config.pose_stale_after_s
        )
        scan_fresh = scan is not None and (now - last_scan_wall) <= float(
            self.config.pose_stale_after_s
        )
        self._dbg(
            "tick",
            state=state.value,
            pose_fresh=pose_fresh,
            scan_fresh=scan_fresh,
            safety_blocked=safety_blocked,
            pose_age_s=None if pose is None else round(now - last_pose_wall, 2),
            scan_age_s=None if scan is None else round(now - last_scan_wall, 2),
            have_costmap=costmap is not None,
            free_cells=free_cells,
            frontier_cells=frontier_count,
            startup_complete=self._startup_complete,
            since_start_s=round(now - self._started_wall_ts, 2),
        )

        if state in (WanderState.IDLE, WanderState.DONE):
            self._dbg("gate", reason="idle_or_done", state=state.value)
            self._heartbeat(
                now,
                state=state.value,
                note="not exploring — send 'Start Explore' or set auto_start=True",
            )
            return

        if bool(self.config.auto_start) and (now - self._started_wall_ts) < float(
            self.config.start_delay_s
        ):
            self._dbg(
                "gate",
                reason="startup_delay",
                since_start_s=round(now - self._started_wall_ts, 2),
                start_delay_s=float(self.config.start_delay_s),
            )
            self._publish_stop()
            self._heartbeat(now, state=state.value, note="startup delay")
            return

        # One-time startup stabilization. Once latched complete it never re-gates,
        # so turns (which legitimately stale the pose) can't deadlock the robot.
        if bool(self.config.auto_start) and not self._startup_complete:
            require_costmap = bool(self.config.startup_require_costmap)
            startup_ready = (
                pose_fresh
                and (scan_fresh or not require_costmap)
                and ((costmap is not None) or not require_costmap)
                and (
                    free_cells >= int(self.config.startup_min_free_cells)
                    or not require_costmap
                )
            )
            if not startup_ready:
                self._startup_ready_since_wall_ts = None
                self._dbg(
                    "gate",
                    reason="startup_not_ready",
                    pose_fresh=pose_fresh,
                    scan_fresh=scan_fresh,
                    have_costmap=costmap is not None,
                    free_cells=free_cells,
                    require_costmap=require_costmap,
                    need_free_cells=int(self.config.startup_min_free_cells),
                )
                self._publish_stop()
                self._heartbeat(
                    now,
                    state=state.value,
                    note="waiting for stable startup map/pose",
                    pose_fresh=pose_fresh,
                    scan_fresh=scan_fresh,
                    have_costmap=costmap is not None,
                    free_cells=free_cells,
                    require_costmap=require_costmap,
                )
                return
            if self._startup_ready_since_wall_ts is None:
                self._startup_ready_since_wall_ts = now
                self._dbg("gate", reason="startup_stabilizing_begin")
                self._publish_stop()
                self._heartbeat(now, state=state.value, note="startup stabilization")
                return
            if (now - self._startup_ready_since_wall_ts) < float(
                self.config.startup_stabilization_s
            ):
                self._dbg(
                    "gate",
                    reason="startup_stabilizing",
                    stable_for_s=round(now - self._startup_ready_since_wall_ts, 2),
                    need_s=float(self.config.startup_stabilization_s),
                )
                self._publish_stop()
                self._heartbeat(now, state=state.value, note="startup stabilization")
                return
            # Stabilized once — latch it so turns (which stale the pose) never
            # drag us back into the startup gate.
            self._startup_complete = True
            self._dbg("gate", reason="startup_complete")

        # The live scan is the essential obstacle sense; pose/costmap only add
        # "intelligent" heading selection. Keep wandering reactively even when the
        # mapping adapter stops publishing pose during turn stabilization.
        if not scan_fresh:
            self._dbg(
                "gate",
                reason="scan_not_fresh",
                have_scan=scan is not None,
                scan_age_s=None if scan is None else round(now - last_scan_wall, 2),
                stale_after_s=float(self.config.pose_stale_after_s),
            )
            self._heartbeat(
                now,
                state=state.value,
                note="waiting for fresh scan",
                have_pose=pose is not None,
                pose_age_s=None if pose is None else round(now - last_pose_wall, 2),
                have_scan=scan is not None,
                scan_age_s=None if scan is None else round(now - last_scan_wall, 2),
                have_costmap=costmap is not None,
            )
            return

        if pose_fresh and pose is not None:
            self._update_yaw_calibration(float(pose.yaw), now)

        # Completion check: the room is mapped when no frontiers remain for a while.
        if self._check_completion(frontier_count, free_cells, now):
            self._dbg("gate", reason="completion_done")
            return

        local_points = self._scan_points(scan)
        clearance = forward_clearance(
            local_points,
            corridor_half_width_m=float(self.config.corridor_half_width_m),
            forward_cone_half_deg=float(self.config.forward_cone_half_deg),
            max_distance_m=float(self.config.scan_max_distance_m),
        )
        effective_trigger = self._effective_turn_trigger()
        self._dbg(
            "clearance",
            state=state.value,
            forward_clearance_m=round(float(clearance), 3),
            effective_trigger_m=round(effective_trigger, 3),
            scan_points=int(local_points.shape[0]),
            safety_blocked=safety_blocked,
            resume_clearance_m=float(self.config.resume_clearance_m),
        )

        if state == WanderState.DRIVE:
            self._tick_drive(
                pose,
                pose_fresh,
                costmap,
                clearance,
                effective_trigger,
                safety_blocked,
                scan_seq,
                pose_seq,
                costmap_seq,
                now,
            )
        elif state == WanderState.TURN:
            if (
                bool(self.config.turn_only_snapshot_mode)
                and self._target_heading is None
                and pose_fresh
                and pose is not None
            ):
                self._begin_turn(pose, pose_fresh, costmap, now, reason="snapshot_step")
            self._tick_turn(
                pose,
                pose_fresh,
                costmap,
                clearance,
                safety_blocked,
                scan_seq,
                pose_seq,
                costmap_seq,
                now,
            )
        elif state == WanderState.SETTLE:
            self._tick_settle(
                pose,
                pose_fresh,
                costmap,
                clearance,
                safety_blocked,
                scan_fresh,
                scan_seq,
                pose_seq,
                costmap_seq,
                now,
            )

    def _effective_turn_trigger(self) -> float:
        """Velocity-scaled distance at which to start turning before an obstacle."""
        if not bool(self.config.lookahead_enabled):
            return float(self.config.turn_trigger_distance_m)
        # Use cruise as the steady planned speed; keep below resume for hysteresis.
        return stopping_trigger_distance(
            planned_throttle=float(self.config.cruise_speed_m_s),
            throttle_to_mps=float(self.config.throttle_to_mps),
            reaction_time_s=float(self.config.reaction_time_s),
            margin_m=float(self.config.lookahead_margin_m),
            floor_m=float(self.config.turn_trigger_distance_m),
            ceiling_m=max(
                float(self.config.turn_trigger_distance_m),
                float(self.config.resume_clearance_m) - 0.1,
            ),
        )

    def _tick_drive(
        self,
        pose: PoseStamped | None,
        pose_fresh: bool,
        costmap: OccupancyGrid | None,
        clearance: float,
        effective_trigger: float,
        safety_blocked: bool,
        scan_seq: int,
        pose_seq: int,
        costmap_seq: int,
        now: float,
    ) -> None:
        # The hard safety gate has vetoed forward motion: don't keep pushing into
        # it — turn away to find an open direction (this is what frees the robot).
        if safety_blocked:
            self._dbg("safety", note="SAFETY BLOCKED while driving -> turning away")
            self._begin_turn(pose, pose_fresh, costmap, now, reason="safety")
            return

        wall_ahead = clearance < float(effective_trigger)

        # Only consider redirecting when actually approaching something — never
        # interrupt a wide-open straight traverse just because it's already mapped.
        explored_ahead = False
        approaching = clearance < float(self.config.heading_lookahead_m)
        if (
            bool(self.config.redirect_when_explored)
            and approaching
            and pose_fresh
            and pose is not None
            and costmap is not None
        ):
            _, unknown_fraction = costmap_raymarch(
                costmap.grid,
                origin_x=float(costmap.origin.position.x),
                origin_y=float(costmap.origin.position.y),
                resolution=float(costmap.resolution),
                start_x=float(pose.x),
                start_y=float(pose.y),
                heading_rad=float(pose.yaw),
                lookahead_m=float(self.config.heading_lookahead_m),
                occupancy_threshold=int(self.config.occupancy_threshold),
            )
            if unknown_fraction < float(self.config.min_explore_fraction):
                if self._explored_since_wall_ts is None:
                    self._explored_since_wall_ts = now
                explored_ahead = (now - self._explored_since_wall_ts) >= float(
                    self.config.redirect_after_s
                )
            else:
                self._explored_since_wall_ts = None

        self._dbg(
            "drive_decision",
            clearance_m=round(float(clearance), 3),
            effective_trigger_m=round(float(effective_trigger), 3),
            wall_ahead=wall_ahead,
            explored_ahead=explored_ahead,
        )

        if wall_ahead or explored_ahead:
            self._begin_turn(pose, pose_fresh, costmap, now, reason="wall" if wall_ahead else "explored")
            return

        if (
            bool(self.config.drive_burst_mapping_enabled)
            and float(self.config.drive_burst_s) > 0.0
            and self._drive_segment_started_wall_ts > 0.0
            and (now - self._drive_segment_started_wall_ts) >= float(self.config.drive_burst_s)
        ):
            self._begin_mapping_settle(
                now,
                scan_seq=scan_seq,
                pose_seq=pose_seq,
                costmap_seq=costmap_seq,
                resume_state=WanderState.DRIVE,
                reason="drive_burst",
                min_wait_s=float(self.config.drive_burst_mapping_min_wait_s),
                max_wait_s=float(self.config.drive_burst_mapping_max_wait_s),
            )
            self._dbg(
                "drive_burst_settle",
                elapsed_s=round(now - self._drive_segment_started_wall_ts, 3),
                clearance_m=round(float(clearance), 3),
            )
            self._publish_stop()
            self._maybe_log(clearance, "drive_burst_settle")
            return

        self._drive_forward(now)
        self._maybe_log(clearance, "drive")

    def _tick_turn(
        self,
        pose: PoseStamped | None,
        pose_fresh: bool,
        costmap: OccupancyGrid | None,
        clearance: float,
        safety_blocked: bool,
        scan_seq: int,
        pose_seq: int,
        costmap_seq: int,
        now: float,
    ) -> None:
        # Keep the estimated yaw locked to the real pose whenever one is available
        # (it's frozen during the turn, but a fresh one may slip through between
        # bursts). This corrects any drift in the integrated estimate.
        if pose_fresh and pose is not None:
            self._turn_est_yaw = float(pose.yaw)
            if self._target_heading is None and (now - self._heading_refreshed_wall_ts) >= float(
                self.config.heading_refresh_s
            ):
                self._refresh_target_heading(pose, costmap, now)
                if self._target_heading is not None and self._turn_est_yaw is not None:
                    err = angle_diff(self._target_heading, self._turn_est_yaw)
                    self._spin_dir = (1.0 if err >= 0.0 else -1.0) * self._yaw_cmd_sign

        if now < float(self._turn_pause_until_wall_ts):
            self._turn_last_tick_wall_ts = now
            self._publish_stop()
            self._maybe_log(clearance, "turn_settle")
            return

        # How much have we actually rotated, and are we pointed at the target yet?
        turned_rad = abs(self._turn_accum_rad)
        min_turn_rad = max(float(self._turn_required_rad), 0.0)
        align_rad = math.radians(float(self.config.align_tolerance_deg))
        if self._target_heading is not None and self._turn_est_yaw is not None:
            remaining_rad = abs(angle_diff(self._target_heading, self._turn_est_yaw))
        else:
            remaining_rad = None
        # Reached the chosen heading (lets a small intended turn finish early), OR
        # swept the minimum (guarantees a real direction change for big/unreachable
        # targets). EITHER, combined with a clear path, lets us resume — so it can
        # never deadlock spinning, yet never bolts forward after a tiny twitch.
        aligned = remaining_rad is not None and remaining_rad <= align_rad
        turned_enough = turned_rad >= min_turn_rad
        path_clear = clearance >= float(self.config.resume_clearance_m)
        turn_only_mode = bool(self.config.turn_only_snapshot_mode)
        turn_complete = (aligned or turned_enough) if turn_only_mode else (
            (not safety_blocked) and path_clear and (aligned or turned_enough)
        )
        if turn_complete:
            if bool(self.config.post_turn_mapping_enabled):
                self._begin_mapping_settle(
                    now,
                    scan_seq=scan_seq,
                    pose_seq=pose_seq,
                    costmap_seq=costmap_seq,
                    resume_state=WanderState.TURN if turn_only_mode else WanderState.DRIVE,
                    reason="snapshot_turn" if turn_only_mode else "post_turn",
                    min_wait_s=float(self.config.post_turn_mapping_min_wait_s),
                    max_wait_s=float(self.config.post_turn_mapping_max_wait_s),
                )
                self._dbg(
                    "resume_to_settle",
                    turned_deg=round(math.degrees(turned_rad), 1),
                    remaining_deg=None if remaining_rad is None else round(math.degrees(remaining_rad), 1),
                    clearance_m=round(float(clearance), 3),
                )
                self._publish_stop()
                self._maybe_log(clearance, "post_turn_settle")
            else:
                with self._lock:
                    if self._state == WanderState.TURN:
                        self._state = WanderState.TURN if turn_only_mode else WanderState.DRIVE
                self._explored_since_wall_ts = None
                if turn_only_mode:
                    self._target_heading = None
                    self._turn_started_wall_ts = now
                    self._turn_burst_started_wall_ts = now
                    self._turn_pause_until_wall_ts = 0.0
                    self._turn_accum_rad = 0.0
                    self._turn_last_tick_wall_ts = now
                self._dbg(
                    "resume",
                    turned_deg=round(math.degrees(turned_rad), 1),
                    remaining_deg=None if remaining_rad is None else round(math.degrees(remaining_rad), 1),
                    clearance_m=round(float(clearance), 3),
                )
                if turn_only_mode:
                    self._publish_stop()
                    self._maybe_log(clearance, "resume_turn_only")
                else:
                    self._drive_forward(now)
                    self._maybe_log(clearance, "resume_drive")
            return

        # Anti-stuck: if a turn drags on without ever satisfying resume, flip the
        # spin direction and re-pick a heading so we don't grind forever.
        if (now - self._turn_started_wall_ts) >= float(self.config.max_turn_s):
            self._spin_dir = -self._spin_dir
            self._turn_started_wall_ts = now
            self._turn_burst_started_wall_ts = now
            self._turn_pause_until_wall_ts = 0.0
            self._turn_accum_rad = 0.0
            if pose_fresh and pose is not None:
                self._refresh_target_heading(pose, costmap, now)

        if (
            float(self.config.turn_burst_s) > 0.0
            and float(self.config.turn_settle_pause_s) > 0.0
            and (now - self._turn_burst_started_wall_ts) >= float(self.config.turn_burst_s)
        ):
            if bool(self.config.turn_burst_mapping_enabled):
                self._begin_mapping_settle(
                    now,
                    scan_seq=scan_seq,
                    pose_seq=pose_seq,
                    costmap_seq=costmap_seq,
                    resume_state=WanderState.TURN,
                    reason="turn_burst",
                    min_wait_s=float(self.config.turn_burst_mapping_min_wait_s),
                    max_wait_s=float(self.config.turn_burst_mapping_max_wait_s),
                )
                self._dbg(
                    "turn_burst_settle",
                    turned_deg=round(math.degrees(abs(self._turn_accum_rad)), 1),
                    clearance_m=round(float(clearance), 3),
                )
                self._publish_stop()
                self._maybe_log(clearance, "turn_burst_settle")
            else:
                self._turn_pause_until_wall_ts = now + float(self.config.turn_settle_pause_s)
                self._turn_burst_started_wall_ts = self._turn_pause_until_wall_ts
                self._turn_last_tick_wall_ts = now
                self._publish_stop()
                self._maybe_log(clearance, "turn_pause")
            return

        # Command the turn and integrate the commanded rotation into the estimate.
        direction = self._spin_dir
        est_rate = (
            float(self.config.turn_speed_rad_s)
            * float(self.config.turn_rate_est_scale)
            * direction
            * self._yaw_cmd_sign
        )
        dt = 0.0
        if self._turn_last_tick_wall_ts > 0.0:
            dt = max(0.0, min(now - self._turn_last_tick_wall_ts, 0.5))
        self._turn_last_tick_wall_ts = now
        self._turn_accum_rad += abs(est_rate) * dt
        if self._turn_est_yaw is not None:
            self._turn_est_yaw = math.atan2(
                math.sin(self._turn_est_yaw + est_rate * dt),
                math.cos(self._turn_est_yaw + est_rate * dt),
            )

        self._dbg(
            "turning",
            turned_deg=round(math.degrees(abs(self._turn_accum_rad)), 1),
            target_deg=None if self._target_heading is None else round(math.degrees(self._target_heading), 1),
            est_yaw_deg=None if self._turn_est_yaw is None else round(math.degrees(self._turn_est_yaw), 1),
            clearance_m=round(float(clearance), 3),
        )
        self._was_driving = False
        self._publish(0.0, self.config.turn_speed_rad_s * direction)
        self._maybe_log(clearance, "turn")

    def _tick_settle(
        self,
        pose: PoseStamped | None,
        pose_fresh: bool,
        costmap: OccupancyGrid | None,
        clearance: float,
        safety_blocked: bool,
        scan_fresh: bool,
        scan_seq: int,
        pose_seq: int,
        costmap_seq: int,
        now: float,
    ) -> None:
        elapsed_s = max(0.0, now - float(self._settle_started_wall_ts))
        min_wait_s = max(float(self._settle_min_wait_s), 0.0)
        max_wait_s = max(float(self._settle_max_wait_s), min_wait_s)
        path_clear = clearance >= float(self.config.resume_clearance_m)
        scan_seq_delta = int(scan_seq) - int(self._settle_scan_seq_baseline)
        pose_seq_delta = int(pose_seq) - int(self._settle_pose_seq_baseline)
        localized_pose_seq_delta = int(self._latest_localized_pose_seq) - int(
            self._settle_localized_pose_seq_baseline
        )
        pose_ready = (not bool(self.config.post_turn_require_pose_update)) or (
            pose_fresh and pose_seq_delta >= max(int(self.config.settle_min_pose_updates), 1)
        )
        costmap_ready = (not bool(self.config.post_turn_require_costmap_update)) or (
            costmap is not None
            and (int(costmap_seq) - int(self._settle_costmap_seq_baseline))
            >= max(int(self.config.settle_min_costmap_updates), 1)
        )
        scan_ready = (not bool(self.config.post_turn_require_scan_update)) or (
            scan_fresh and scan_seq_delta >= max(int(self.config.settle_min_scan_updates), 1)
        )
        localized_ready = (not bool(self.config.post_turn_require_localized_commit)) or (
            localized_pose_seq_delta >= 1
        )
        costmap_seq_delta = int(costmap_seq) - int(self._settle_costmap_seq_baseline)

        if safety_blocked or not path_clear:
            self._dbg(
                "settle_reenter_turn",
                safety_blocked=safety_blocked,
                clearance_m=round(float(clearance), 3),
            )
            self._begin_turn(pose, pose_fresh, costmap, now, reason="wall")
            return

        if elapsed_s >= min_wait_s and scan_ready and pose_ready and costmap_ready and localized_ready:
            if self._settle_resume_state == WanderState.TURN:
                if bool(self.config.turn_only_snapshot_mode) and pose_fresh and pose is not None:
                    self._begin_turn(pose, pose_fresh, costmap, now, reason="snapshot_step")
                else:
                    with self._lock:
                        if self._state == WanderState.SETTLE:
                            self._state = WanderState.TURN
                    if bool(self.config.turn_only_snapshot_mode):
                        self._target_heading = None
                    self._turn_pause_until_wall_ts = 0.0
                    self._turn_burst_started_wall_ts = now
                    self._turn_last_tick_wall_ts = now
                    if pose_fresh and pose is not None:
                        self._turn_est_yaw = float(pose.yaw)
                logger.info(
                    "SourcceyWanderExplorer turn-burst mapping settle complete",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=scan_ready,
                    pose_ready=pose_ready,
                    costmap_ready=costmap_ready,
                    scan_seq=scan_seq,
                    pose_seq=pose_seq,
                    costmap_seq=costmap_seq,
                    scan_seq_delta=scan_seq_delta,
                    pose_seq_delta=pose_seq_delta,
                    costmap_seq_delta=costmap_seq_delta,
                    reason=self._settle_reason,
                )
                trace_event(
                    "wander_explorer",
                    "settle_complete_resume_turn",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=bool(scan_ready),
                    pose_ready=bool(pose_ready),
                    costmap_ready=bool(costmap_ready),
                    scan_seq_delta=int(scan_seq_delta),
                    pose_seq_delta=int(pose_seq_delta),
                    localized_pose_seq_delta=int(localized_pose_seq_delta),
                    costmap_seq_delta=int(costmap_seq_delta),
                    reason=str(self._settle_reason),
                )
                self._maybe_log(clearance, "resume_turn")
            else:
                with self._lock:
                    if self._state == WanderState.SETTLE:
                        self._state = WanderState.DRIVE
                self._explored_since_wall_ts = None
                logger.info(
                    "SourcceyWanderExplorer mapping settle complete",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=scan_ready,
                    pose_ready=pose_ready,
                    costmap_ready=costmap_ready,
                    scan_seq=scan_seq,
                    pose_seq=pose_seq,
                    costmap_seq=costmap_seq,
                    scan_seq_delta=scan_seq_delta,
                    pose_seq_delta=pose_seq_delta,
                    costmap_seq_delta=costmap_seq_delta,
                    reason=self._settle_reason,
                )
                trace_event(
                    "wander_explorer",
                    "settle_complete_resume_drive",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=bool(scan_ready),
                    pose_ready=bool(pose_ready),
                    costmap_ready=bool(costmap_ready),
                    scan_seq_delta=int(scan_seq_delta),
                    pose_seq_delta=int(pose_seq_delta),
                    localized_pose_seq_delta=int(localized_pose_seq_delta),
                    costmap_seq_delta=int(costmap_seq_delta),
                    reason=str(self._settle_reason),
                )
                self._drive_forward(now)
                self._maybe_log(clearance, "resume_drive")
            return

        if elapsed_s >= max_wait_s:
            if self._settle_resume_state == WanderState.TURN and not localized_ready:
                logger.warning(
                    "SourcceyWanderExplorer mapping settle waiting for trusted localized commit",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=scan_ready,
                    pose_ready=pose_ready,
                    localized_ready=localized_ready,
                    costmap_ready=costmap_ready,
                    scan_seq_delta=scan_seq_delta,
                    pose_seq_delta=pose_seq_delta,
                    localized_pose_seq_delta=localized_pose_seq_delta,
                    costmap_seq_delta=costmap_seq_delta,
                    reason=self._settle_reason,
                )
                trace_event(
                    "wander_explorer",
                    "settle_waiting_for_localized_commit",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=bool(scan_ready),
                    pose_ready=bool(pose_ready),
                    localized_ready=bool(localized_ready),
                    costmap_ready=bool(costmap_ready),
                    scan_seq_delta=int(scan_seq_delta),
                    pose_seq_delta=int(pose_seq_delta),
                    localized_pose_seq_delta=int(localized_pose_seq_delta),
                    costmap_seq_delta=int(costmap_seq_delta),
                    reason=str(self._settle_reason),
                )
                self._settle_started_wall_ts = now
                self._publish_stop()
                self._maybe_log(clearance, "wait_localized_commit")
                return
            if self._settle_resume_state == WanderState.TURN:
                if bool(self.config.turn_only_snapshot_mode) and pose_fresh and pose is not None:
                    self._begin_turn(pose, pose_fresh, costmap, now, reason="snapshot_step")
                else:
                    with self._lock:
                        if self._state == WanderState.SETTLE:
                            self._state = WanderState.TURN
                    if bool(self.config.turn_only_snapshot_mode):
                        self._target_heading = None
                    self._turn_pause_until_wall_ts = 0.0
                    self._turn_burst_started_wall_ts = now
                    self._turn_last_tick_wall_ts = now
                logger.info(
                    "SourcceyWanderExplorer turn-burst mapping settle timed out",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=scan_ready,
                    pose_ready=pose_ready,
                    costmap_ready=costmap_ready,
                    scan_seq=scan_seq,
                    pose_seq=pose_seq,
                    costmap_seq=costmap_seq,
                    scan_seq_delta=scan_seq_delta,
                    pose_seq_delta=pose_seq_delta,
                    costmap_seq_delta=costmap_seq_delta,
                    reason=self._settle_reason,
                )
                trace_event(
                    "wander_explorer",
                    "settle_timeout_resume_turn",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=bool(scan_ready),
                    pose_ready=bool(pose_ready),
                    costmap_ready=bool(costmap_ready),
                    scan_seq_delta=int(scan_seq_delta),
                    pose_seq_delta=int(pose_seq_delta),
                    localized_pose_seq_delta=int(localized_pose_seq_delta),
                    costmap_seq_delta=int(costmap_seq_delta),
                    reason=str(self._settle_reason),
                )
                self._maybe_log(clearance, "resume_turn_timeout")
            else:
                with self._lock:
                    if self._state == WanderState.SETTLE:
                        self._state = WanderState.DRIVE
                self._explored_since_wall_ts = None
                logger.info(
                    "SourcceyWanderExplorer mapping settle timed out",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=scan_ready,
                    pose_ready=pose_ready,
                    costmap_ready=costmap_ready,
                    scan_seq=scan_seq,
                    pose_seq=pose_seq,
                    costmap_seq=costmap_seq,
                    scan_seq_delta=scan_seq_delta,
                    pose_seq_delta=pose_seq_delta,
                    costmap_seq_delta=costmap_seq_delta,
                    reason=self._settle_reason,
                )
                trace_event(
                    "wander_explorer",
                    "settle_timeout_resume_drive",
                    elapsed_s=round(elapsed_s, 3),
                    scan_ready=bool(scan_ready),
                    pose_ready=bool(pose_ready),
                    costmap_ready=bool(costmap_ready),
                    scan_seq_delta=int(scan_seq_delta),
                    pose_seq_delta=int(pose_seq_delta),
                    localized_pose_seq_delta=int(localized_pose_seq_delta),
                    costmap_seq_delta=int(costmap_seq_delta),
                    reason=str(self._settle_reason),
                )
                self._drive_forward(now)
                self._maybe_log(clearance, "resume_drive_timeout")
            return

        self._publish_stop()
        self._heartbeat(
            now,
            state=self._state.value,
            note=f"waiting for mapping settle ({self._settle_reason})",
            elapsed_s=round(elapsed_s, 2),
            scan_ready=scan_ready,
            pose_ready=pose_ready,
            localized_ready=localized_ready,
            costmap_ready=costmap_ready,
            scan_seq_delta=scan_seq_delta,
            pose_seq_delta=pose_seq_delta,
            localized_pose_seq_delta=localized_pose_seq_delta,
            costmap_seq_delta=costmap_seq_delta,
            clearance_m=round(float(clearance), 3),
        )
        self._maybe_log(clearance, "settle")

    def _begin_turn(
        self,
        pose: PoseStamped | None,
        pose_fresh: bool,
        costmap: OccupancyGrid | None,
        now: float,
        *,
        reason: str,
    ) -> None:
        with self._lock:
            self._state = WanderState.TURN
        # Remember the direction we were just driving (and exploring) so the new
        # heading is chosen to avoid going that way again.
        if pose_fresh and pose is not None:
            self._record_committed_heading(float(pose.yaw))
        self._turn_started_wall_ts = now
        self._turn_burst_started_wall_ts = now
        self._turn_pause_until_wall_ts = 0.0
        self._drive_segment_started_wall_ts = 0.0
        self._explored_since_wall_ts = None
        # Reset turn-progress tracking for this new turn.
        self._turn_accum_rad = 0.0
        self._turn_last_tick_wall_ts = now
        self._turn_est_yaw = float(pose.yaw) if (pose_fresh and pose is not None) else None
        if pose_fresh and pose is not None:
            if bool(self.config.turn_only_snapshot_mode) or reason == "snapshot_step":
                step_rad = math.radians(max(float(self.config.turn_only_step_deg), 1.0))
                self._spin_dir = 1.0 * self._yaw_cmd_sign
                self._turn_required_rad = step_rad
                self._target_heading = math.atan2(
                    math.sin(float(pose.yaw) + (self._spin_dir * step_rad)),
                    math.cos(float(pose.yaw) + (self._spin_dir * step_rad)),
                )
                self._heading_refreshed_wall_ts = now
            elif reason in {"safety", "wall"}:
                self._spin_dir = self._spin_dir_from_scan()
                turn_rad = math.radians(max(float(self.config.min_turn_deg), 70.0))
                self._turn_required_rad = turn_rad
                self._target_heading = math.atan2(
                    math.sin(float(pose.yaw) + (self._spin_dir * turn_rad)),
                    math.cos(float(pose.yaw) + (self._spin_dir * turn_rad)),
                )
                self._heading_refreshed_wall_ts = now
            else:
                self._turn_required_rad = math.radians(max(float(self.config.min_turn_deg), 1.0))
                self._refresh_target_heading(pose, costmap, now)
                if self._target_heading is not None:
                    err = angle_diff(self._target_heading, float(pose.yaw))
                    self._spin_dir = (1.0 if err >= 0.0 else -1.0) * self._yaw_cmd_sign
        else:
            # No usable pose: pick the more open side from the live scan and bounce.
            self._turn_required_rad = math.radians(max(float(self.config.min_turn_deg), 1.0))
            self._target_heading = None
            self._spin_dir = self._spin_dir_from_scan()
        logger.info(
            "SourcceyWanderExplorer turning to seek open/unexplored space",
            reason=reason,
            pose_fresh=pose_fresh,
            spin_dir=self._spin_dir,
            target_heading_deg=None
            if self._target_heading is None
            else round(math.degrees(self._target_heading), 1),
        )
        trace_event(
            "wander_explorer",
            "begin_turn",
            reason=str(reason),
            pose_fresh=bool(pose_fresh),
            spin_dir=float(self._spin_dir),
            target_heading_deg=None
            if self._target_heading is None
            else round(math.degrees(self._target_heading), 1),
        )

    def _begin_mapping_settle(
        self,
        now: float,
        *,
        scan_seq: int,
        pose_seq: int,
        costmap_seq: int,
        resume_state: WanderState,
        reason: str,
        min_wait_s: float,
        max_wait_s: float,
    ) -> None:
        with self._lock:
            self._state = WanderState.SETTLE
        self._settle_started_wall_ts = now
        self._settle_scan_seq_baseline = int(scan_seq)
        self._settle_pose_seq_baseline = int(pose_seq)
        self._settle_localized_pose_seq_baseline = int(self._latest_localized_pose_seq)
        self._settle_costmap_seq_baseline = int(costmap_seq)
        self._settle_resume_state = resume_state
        self._settle_reason = str(reason)
        self._settle_min_wait_s = float(min_wait_s)
        self._settle_max_wait_s = float(max_wait_s)
        logger.info(
            "SourcceyWanderExplorer entering mapping settle",
            reason=self._settle_reason,
            resume_state=self._settle_resume_state.value,
            scan_seq_baseline=self._settle_scan_seq_baseline,
            pose_seq_baseline=self._settle_pose_seq_baseline,
            localized_pose_seq_baseline=self._settle_localized_pose_seq_baseline,
            costmap_seq_baseline=self._settle_costmap_seq_baseline,
            min_wait_s=round(self._settle_min_wait_s, 3),
            max_wait_s=round(self._settle_max_wait_s, 3),
        )
        trace_event(
            "wander_explorer",
            "begin_mapping_settle",
            reason=str(self._settle_reason),
            resume_state=self._settle_resume_state.value,
            scan_seq_baseline=int(self._settle_scan_seq_baseline),
            pose_seq_baseline=int(self._settle_pose_seq_baseline),
            localized_pose_seq_baseline=int(self._settle_localized_pose_seq_baseline),
            costmap_seq_baseline=int(self._settle_costmap_seq_baseline),
            min_wait_s=round(self._settle_min_wait_s, 3),
            max_wait_s=round(self._settle_max_wait_s, 3),
        )

    def _record_committed_heading(self, heading: float) -> None:
        """Remember a world-frame heading the robot drove off in (revisit memory).

        Collapses a near-duplicate of the most recent entry so repeatedly driving
        the same way doesn't flood the memory and over-penalize that direction.
        """
        heading = math.atan2(math.sin(heading), math.cos(heading))
        dedup_rad = math.radians(float(self.config.revisit_dedup_deg))
        if self._recent_headings and abs(
            angle_diff(heading, self._recent_headings[-1])
        ) < dedup_rad:
            self._recent_headings[-1] = heading
            return
        self._recent_headings.append(heading)
        max_n = max(int(self.config.revisit_memory_size), 1)
        if len(self._recent_headings) > max_n:
            self._recent_headings = self._recent_headings[-max_n:]
        self._dbg(
            "revisit",
            recorded_deg=round(math.degrees(heading), 1),
            memory_deg=[round(math.degrees(h), 1) for h in self._recent_headings],
        )

    def _spin_dir_from_scan(self) -> float:
        """Pick a turn direction toward the more open hemisphere of the live scan.

        ``+angular.z`` is the robot's left; with ``invert_lateral_axis`` the
        point-cloud +y already maps to left. Turn toward whichever side has more
        nearby free space; fall back to the current persistent direction.
        """
        with self._lock:
            scan = self._latest_scan
        points = self._scan_points(scan)
        if points.shape[0] == 0:
            return self._spin_dir
        lateral = points[:, 1]
        forward = points[:, 0]
        near = forward < float(self.config.turn_trigger_distance_m) * 2.0
        left_blocked = int(np.sum(near & (lateral > 0.0)))
        right_blocked = int(np.sum(near & (lateral < 0.0)))
        if left_blocked == right_blocked:
            return self._spin_dir
        # Turn away from the more blocked side (toward the more open side).
        return 1.0 if right_blocked > left_blocked else -1.0

    def _refresh_target_heading(
        self,
        pose: PoseStamped,
        costmap: OccupancyGrid | None,
        now: float,
    ) -> None:
        self._heading_refreshed_wall_ts = now
        with self._lock:
            scan = self._latest_scan
        scan_points = self._scan_points(scan)
        if costmap is None or costmap.grid.size == 0:
            # No map yet: just turn in place; resume condition (live scan) guards safety.
            self._target_heading = angle_diff(float(pose.yaw) + math.pi / 2.0, 0.0)
            return
        heading, _unknown, _clearance = select_world_heading(
            costmap.grid,
            origin_x=float(costmap.origin.position.x),
            origin_y=float(costmap.origin.position.y),
            resolution=float(costmap.resolution),
            robot_x=float(pose.x),
            robot_y=float(pose.y),
            robot_yaw=float(pose.yaw),
            n_candidates=int(self.config.heading_candidates),
            lookahead_m=float(self.config.heading_lookahead_m),
            occupancy_threshold=int(self.config.occupancy_threshold),
            min_clearance_m=float(self.config.min_heading_clearance_m),
            clearance_weight=float(self.config.clearance_weight),
            explore_weight=float(self.config.explore_weight),
            turn_penalty_weight=float(self.config.turn_penalty_weight),
            scan_points_local=scan_points if scan_points.shape[0] > 0 else None,
            scan_fov_half_deg=float(self.config.valid_angle_half_width_deg),
            avoid_headings=(
                list(self._recent_headings)
                if bool(self.config.revisit_avoid_enabled)
                else None
            ),
            avoid_penalty_weight=float(self.config.revisit_penalty_weight),
            avoid_sigma_rad=math.radians(float(self.config.revisit_sigma_deg)),
        )
        self._target_heading = heading

    def _check_completion(self, frontier_count: int, free_cells: int, now: float) -> bool:
        if not bool(self.config.completion_enabled):
            return False
        mapped = free_cells >= int(self.config.min_free_cells_for_done) and frontier_count < int(
            self.config.frontier_done_threshold
        )
        if not mapped:
            self._done_since_wall_ts = None
            return False
        if self._done_since_wall_ts is None:
            self._done_since_wall_ts = now
            return False
        if (now - self._done_since_wall_ts) < float(self.config.done_after_s):
            return False
        with self._lock:
            self._state = WanderState.DONE
        self._publish_stop()
        logger.info(
            "SourcceyWanderExplorer finished: room mapped",
            frontier_cells=frontier_count,
            free_cells=free_cells,
        )
        return True

    # --- helpers ---

    def _scan_points(self, scan: PlanarLidarScan | None) -> np.ndarray:
        if scan is None:
            return np.zeros((0, 2), dtype=np.float32)
        return scan_to_local_xy(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            invert_lateral_axis=bool(self.config.invert_lateral_axis),
            max_distance_m=float(self.config.scan_max_distance_m),
            min_confidence=int(self.config.min_confidence),
            min_distance_m=float(self.config.scan_min_range_m),
        )

    def _update_yaw_calibration(self, yaw: float, now: float) -> None:
        if self._prev_yaw is not None and abs(self._prev_cmd_wz) > 1e-3:
            dyaw = angle_diff(yaw, self._prev_yaw)
            if abs(dyaw) > 0.02:
                observed = math.copysign(1.0, dyaw) * math.copysign(1.0, self._prev_cmd_wz)
                self._yaw_cmd_sign = observed
        self._prev_yaw = yaw
        self._prev_yaw_wall_ts = now

    def _drive_forward(self, now: float) -> None:
        """Publish a forward drive command, with a static-friction kick on start.

        The first ticks after entering DRIVE use ``kick_speed_m_s`` to break the
        heavy base loose, then it settles to ``cruise_speed_m_s``. While cruising,
        a brief periodic re-kick guards against stalls without raising avg speed.
        """
        period = float(self.config.re_kick_period_s)
        if not self._was_driving:
            # Fresh start from standstill: full kick to break static friction.
            self._drive_segment_started_wall_ts = now
            self._drive_kick_until_wall_ts = now + float(self.config.kick_duration_s)
            self._kick_speed = float(self.config.kick_speed_m_s)
            self._next_rekick_wall_ts = (now + period) if period > 0.0 else math.inf
            self._was_driving = True
        elif period > 0.0 and now >= self._next_rekick_wall_ts:
            # Already rolling: a gentle, longer push guards against stalls without
            # the harsh lurch of a full breakaway kick — slow and smooth.
            self._drive_kick_until_wall_ts = now + float(self.config.re_kick_duration_s)
            self._kick_speed = float(self.config.re_kick_speed_m_s)
            self._next_rekick_wall_ts = now + period
        if now < self._drive_kick_until_wall_ts:
            self._dbg("drive_forward", mode="kick", throttle=round(self._kick_speed, 3))
            self._publish(self._kick_speed, 0.0)
        else:
            self._dbg("drive_forward", mode="cruise", throttle=float(self.config.cruise_speed_m_s))
            self._publish(float(self.config.cruise_speed_m_s), 0.0)

    def _publish(self, linear_x: float, angular_z: float) -> None:
        self._prev_cmd_wz = float(angular_z)
        cmd_key = (round(float(linear_x), 4), round(float(angular_z), 4), self._state.value)
        if getattr(self, "_last_traced_cmd_key", None) != cmd_key:
            trace_event(
                "wander_explorer",
                "publish_cmd_vel",
                linear_x=cmd_key[0],
                angular_z=cmd_key[1],
                state=cmd_key[2],
            )
            self._last_traced_cmd_key = cmd_key
        self._dbg(
            "publish",
            linear_x=round(float(linear_x), 4),
            angular_z=round(float(angular_z), 4),
        )
        self.cmd_vel.publish(
            Twist(
                linear=Vector3(float(linear_x), 0.0, 0.0),
                angular=Vector3(0.0, 0.0, float(angular_z)),
            )
        )

    def _publish_stop(self) -> None:
        self._prev_cmd_wz = 0.0
        self._was_driving = False
        cmd_key = (0.0, 0.0, self._state.value)
        if getattr(self, "_last_traced_cmd_key", None) != cmd_key:
            trace_event(
                "wander_explorer",
                "publish_cmd_vel",
                linear_x=0.0,
                angular_z=0.0,
                state=self._state.value,
            )
            self._last_traced_cmd_key = cmd_key
        self._dbg("publish_stop", linear_x=0.0, angular_z=0.0)
        self.cmd_vel.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))

    def _heartbeat(self, now: float, **fields: Any) -> None:
        if (now - self._last_heartbeat_wall_ts) < float(self.config.heartbeat_s):
            return
        self._last_heartbeat_wall_ts = now
        logger.info("SourcceyWanderExplorer heartbeat", **fields)

    def _maybe_log(self, clearance: float, phase: str) -> None:
        self._tick += 1
        if self._tick % max(int(self.config.log_every_ticks), 1) != 0:
            return
        logger.info(
            "SourcceyWanderExplorer",
            phase=phase,
            state=self._state.value,
            forward_clearance_m=round(float(clearance), 2),
            yaw_cmd_sign=self._yaw_cmd_sign,
            frontier_cells=self._frontier_count,
        )
