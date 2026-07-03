from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

from .lidar_geometry import normalize_angle_deg
from .lidar_types import PlanarLidarScan
from .run_trace import trace_event

logger = setup_logger()


class SourcceyLidarPointCloudAdapterConfig(ModuleConfig):
    forward_angle_deg: float = 270.0
    valid_angle_half_width_deg: float = 90.0
    invert_lateral_axis: bool = False
    lidar_mount_x_m: float = 0.0
    lidar_mount_y_m: float = 0.0
    max_distance_m: float = 8.0
    # Drop returns closer than this: they are the robot's own body / arms / lidar
    # housing, not real obstacles, and otherwise paint a permanent ring of phantom
    # green "remembered" points right around the robot.
    min_range_m: float = 0.22
    min_confidence: int = 0
    free_ray_step_m: float = 0.05
    free_ray_start_m: float = 0.05
    free_height_m: float = 0.0
    obstacle_height_m: float = 0.25
    frame_id: str = "world"
    odom_stale_after_s: float = 0.75
    publish_empty_clouds: bool = False
    log_every_scans: int = 30
    scan_match_enabled: bool = True
    scan_match_max_points: int = 120
    scan_match_translation_window_m: float = 0.20
    scan_match_low_motion_translation_window_m: float = 0.08
    scan_match_low_motion_trigger_m: float = 0.08
    scan_match_rotation_window_deg: float = 10.0
    scan_match_accept_score_m: float = 0.12
    scan_match_overlap_radius_m: float = 0.10
    scan_match_min_overlap_fraction: float = 0.42
    min_reference_points_for_scan_match: int = 60
    scan_match_strict_reference_points: int = 480
    scan_match_strict_accept_score_m: float = 0.03
    scan_match_strict_min_overlap_fraction: float = 0.50
    scan_match_pose_jump_reject_m: float = 0.18
    scan_match_yaw_jump_reject_deg: float = 10.0
    scan_match_pose_jump_overlap_floor: float = 0.70
    scan_match_seed_guard_translation_m: float = 0.10
    scan_match_seed_guard_yaw_deg: float = 10.0
    scan_match_seed_guard_overlap_floor: float = 0.92
    # Turn-only snapshot runs should never accept a "great overlap" fit that
    # barely rotated relative to the last trusted pose even though the odom/IMU
    # seed says the robot just made a substantial turn. Those are the false
    # same-pose commits that make the runtime map appear frozen while the robot
    # keeps moving physically.
    turn_snapshot_expected_seed_yaw_deg: float = 20.0
    turn_snapshot_min_candidate_yaw_deg: float = 16.0
    turn_snapshot_stale_translation_floor_m: float = 0.05
    turn_snapshot_stale_overlap_floor: float = 0.92
    turn_snapshot_search_half_width_deg: float = 185.0
    turn_snapshot_search_step_deg: float = 6.0
    turn_snapshot_search_refine_half_width_deg: float = 16.0
    turn_snapshot_search_refine_step_deg: float = 1.5
    turn_snapshot_search_translation_window_m: float = 0.08
    # When a stopped turn snapshot produces an extremely strong LiDAR-only
    # relocalization, trust that fit even if raw odom/IMU yaw drifted badly. This
    # avoids the "frozen map" failure where the best LiDAR match is rejected only
    # because the seed heading is wrong.
    turn_snapshot_lidar_override_overlap_floor: float = 0.90
    turn_snapshot_lidar_override_score_ceiling_m: float = 0.01
    turn_snapshot_lidar_override_max_translation_m: float = 0.12
    # Early after bootstrap, turn-only mapping can present a very sparse anchor:
    # one wall segment and little else. In that regime, normal scan matching can
    # reject every post-turn snapshot forever, leaving the global map blank. When
    # the robot is clearly turning in place and raw odom indicates a small-XY,
    # large-yaw change, allow a conservative odom-seeded growth commit so the
    # registration anchor can expand beyond the first wall.
    odom_turn_growth_enabled: bool = True
    odom_turn_growth_reference_points_max: int = 180
    odom_turn_growth_min_turn_deg: float = 20.0
    odom_turn_growth_max_translation_m: float = 0.12
    odom_turn_growth_max_score_m: float = 0.18
    odom_turn_growth_min_overlap_fraction: float = 0.10
    # Never allow raw odom "turn growth" to commit a large heading jump when the
    # best LiDAR candidate plainly disagrees. That exact conflict was bending the
    # room into detached vertical strips: odom said ~90deg, LiDAR only supported
    # ~10-15deg, and the fallback still committed the 90deg pose.
    odom_turn_growth_candidate_yaw_margin_deg: float = 35.0
    odom_turn_growth_candidate_turn_ratio_floor: float = 0.45
    odom_turn_growth_candidate_sign_min_turn_deg: float = 12.0
    # Very early in a turn-only room scan the "map" may still be just one wall.
    # Pure scan matching is underconstrained there, so a real 45deg stop-turn can
    # be rejected forever even though the robot did exactly what we asked. Allow a
    # conservative seed-growth commit while the reference is still sparse so the
    # second/third stationary turn snapshots can actually build the room outline.
    bootstrap_turn_seed_enabled: bool = True
    bootstrap_turn_seed_reference_points_max: int = 180
    bootstrap_turn_seed_min_turn_deg: float = 25.0
    bootstrap_turn_seed_max_translation_m: float = 0.10
    bootstrap_turn_seed_max_score_m: float = 0.10
    bootstrap_turn_seed_max_overlap_fraction: float = 0.55
    submap_max_points: int = 1600
    submap_local_radius_m: float = 1.25
    registered_scan_height_m: float = 0.32
    reset_context_translation_m: float = 0.35
    reset_context_rotation_deg: float = 30.0
    min_points_for_scan_match: int = 40
    max_odom_angular_speed_for_mapping_rad_s: float = 0.30
    max_odom_linear_speed_for_mapping_m_s: float = 0.20
    max_seed_odom_step_m: float = 0.45
    max_seed_odom_step_deg: float = 35.0
    mapping_holdoff_after_turn_s: float = 0.35
    mapping_requires_stationary_scans: int = 3
    stationary_keyframe_buffer_scans: int = 8
    stationary_keyframe_voxel_m: float = 0.03
    stationary_keyframe_max_points: int = 320
    stationary_commit_reset_translation_m: float = 0.05
    stationary_commit_reset_rotation_deg: float = 6.0
    # Preferred "is the robot moving" signal: the commanded velocity actually sent
    # to the base. Dead-reckoned odom deltas are unreliable (gyro bias + two odom
    # publishers interleaving), which previously pinned the mapper in a permanent
    # "turn holdoff" so it never accepted a stationary snapshot. When a cmd_vel
    # stream is present we gate on it; otherwise we fall back to the odom-speed
    # heuristic above.
    cmd_vel_gate_enabled: bool = True
    cmd_vel_active_window_s: float = 1.0
    cmd_vel_motion_epsilon: float = 1e-3
    obstacle_memory_enabled: bool = True
    obstacle_memory_decay_s: float = 3600.0
    obstacle_memory_max_points: int = 8000
    obstacle_memory_voxel_m: float = 0.04
    obstacle_memory_height_m: float = 0.32
    obstacle_memory_visible_clear_radius_m: float = 0.16
    obstacle_memory_match_overlap_floor: float = 0.85
    obstacle_memory_match_score_ceiling_m: float = 0.03
    mapping_debug_enabled: bool = False
    mapping_debug_log_every_scans: int = 5
    mapping_debug_log_every_skips: int = 3


def _scan_to_local_points(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_confidence: int,
    min_range_m: float = 0.0,
) -> list[tuple[float, float, float, int]]:
    points: list[tuple[float, float, float, int]] = []
    for angle_deg, distance_m, confidence in zip(
        scan.angles_deg,
        scan.distances_m,
        scan.confidences,
        strict=False,
    ):
        distance = float(distance_m)
        conf = int(confidence)
        if conf < int(min_confidence):
            continue
        if not math.isfinite(distance) or distance <= 0.0 or distance > float(max_distance_m):
            continue
        if distance < float(min_range_m):
            continue
        delta_deg = normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        if abs(delta_deg) > float(valid_angle_half_width_deg):
            continue
        theta_rad = math.radians(delta_deg)
        forward_m = distance * math.cos(theta_rad)
        lateral_m = distance * math.sin(theta_rad)
        if invert_lateral_axis:
            lateral_m = -lateral_m
        points.append((forward_m, lateral_m, distance, conf))
    return points


def _transform_xy(points_local_xy: np.ndarray, pose: PoseStamped) -> np.ndarray:
    if points_local_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cos_yaw = math.cos(float(pose.yaw))
    sin_yaw = math.sin(float(pose.yaw))
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = points_local_xy @ rotation.T
    world[:, 0] += float(pose.x)
    world[:, 1] += float(pose.y)
    return world


def _sensor_origin_xy(pose: PoseStamped, *, mount_x_m: float, mount_y_m: float) -> tuple[float, float]:
    cos_yaw = math.cos(float(pose.yaw))
    sin_yaw = math.sin(float(pose.yaw))
    sensor_x = float(pose.x) + (float(mount_x_m) * cos_yaw) - (float(mount_y_m) * sin_yaw)
    sensor_y = float(pose.y) + (float(mount_x_m) * sin_yaw) + (float(mount_y_m) * cos_yaw)
    return sensor_x, sensor_y


def _sensor_origin_displacement_m(
    pose_a: PoseStamped,
    pose_b: PoseStamped,
    *,
    mount_x_m: float,
    mount_y_m: float,
) -> float:
    ax, ay = _sensor_origin_xy(pose_a, mount_x_m=mount_x_m, mount_y_m=mount_y_m)
    bx, by = _sensor_origin_xy(pose_b, mount_x_m=mount_x_m, mount_y_m=mount_y_m)
    return math.hypot(float(bx) - float(ax), float(by) - float(ay))


def _transform_xy_with_mount(
    points_local_xy: np.ndarray,
    pose: PoseStamped,
    *,
    mount_x_m: float,
    mount_y_m: float,
) -> np.ndarray:
    if points_local_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    sensor_x, sensor_y = _sensor_origin_xy(pose, mount_x_m=mount_x_m, mount_y_m=mount_y_m)
    cos_yaw = math.cos(float(pose.yaw))
    sin_yaw = math.sin(float(pose.yaw))
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = points_local_xy @ rotation.T
    world[:, 0] += float(sensor_x)
    world[:, 1] += float(sensor_y)
    return world


def _ray_samples_local(
    *,
    forward_m: float,
    lateral_m: float,
    distance_m: float,
    step_m: float,
    start_m: float,
) -> np.ndarray:
    usable_start = max(float(start_m), 0.0)
    usable_step = max(float(step_m), 1e-3)
    if distance_m <= usable_start + usable_step:
        return np.zeros((0, 2), dtype=np.float32)

    sample_distances = np.arange(usable_start, distance_m, usable_step, dtype=np.float32)
    if sample_distances.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    sample_distances = sample_distances[sample_distances < float(distance_m)]
    if sample_distances.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    unit_forward = float(forward_m) / float(distance_m)
    unit_lateral = float(lateral_m) / float(distance_m)
    return np.column_stack((sample_distances * unit_forward, sample_distances * unit_lateral)).astype(
        np.float32,
        copy=False,
    )


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _transform_local_points(points_local_xy: np.ndarray, *, x: float, y: float, yaw: float) -> np.ndarray:
    if points_local_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = points_local_xy @ rotation.T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    return world


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indices]


def _voxelize_points(points_xy: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    if points_xy.size == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.float32),
        )

    voxel = max(float(voxel_m), 1e-3)
    cells = np.round(points_xy / voxel).astype(np.int32, copy=False)
    _, unique_indices = np.unique(cells, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return (
        cells[unique_indices].astype(np.int32, copy=False),
        points_xy[unique_indices].astype(np.float32, copy=False),
    )


def _scan_match_score(reference_world: np.ndarray, candidate_world: np.ndarray) -> float:
    if reference_world.size == 0 or candidate_world.size == 0:
        return float("inf")
    deltas = candidate_world[:, None, :] - reference_world[None, :, :]
    min_d2 = np.min(np.sum(deltas * deltas, axis=2), axis=1)
    return float(np.mean(np.clip(min_d2, 0.0, 0.25)))


def _scan_match_metrics(
    reference_world: np.ndarray,
    candidate_world: np.ndarray,
    *,
    overlap_radius_m: float,
) -> tuple[float, float]:
    if reference_world.size == 0 or candidate_world.size == 0:
        return float("inf"), 0.0
    deltas = candidate_world[:, None, :] - reference_world[None, :, :]
    min_d2 = np.min(np.sum(deltas * deltas, axis=2), axis=1)
    score = float(np.mean(np.clip(min_d2, 0.0, 0.25)))
    overlap_radius2 = max(float(overlap_radius_m), 1e-3) ** 2
    overlap_fraction = float(np.mean(min_d2 <= overlap_radius2))
    return score, overlap_fraction


def build_pointcloud_from_scan(
    scan: PlanarLidarScan,
    pose: PoseStamped,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    lidar_mount_x_m: float,
    lidar_mount_y_m: float,
    max_distance_m: float,
    min_confidence: int,
    free_ray_step_m: float,
    free_ray_start_m: float,
    free_height_m: float,
    obstacle_height_m: float,
    frame_id: str,
    min_range_m: float = 0.0,
) -> PointCloud2:
    local_points = _scan_to_local_points(
        scan,
        forward_angle_deg=forward_angle_deg,
        valid_angle_half_width_deg=valid_angle_half_width_deg,
        invert_lateral_axis=invert_lateral_axis,
        max_distance_m=max_distance_m,
        min_confidence=min_confidence,
        min_range_m=min_range_m,
    )
    local_hits = (
        np.asarray(local_points, dtype=np.float32)
        if local_points
        else np.zeros((0, 4), dtype=np.float32)
    )
    return build_pointcloud_from_local_hits(
        local_hits,
        pose,
        lidar_mount_x_m=lidar_mount_x_m,
        lidar_mount_y_m=lidar_mount_y_m,
        free_ray_step_m=free_ray_step_m,
        free_ray_start_m=free_ray_start_m,
        free_height_m=free_height_m,
        obstacle_height_m=obstacle_height_m,
        frame_id=frame_id,
        timestamp=float(scan.ts),
    )


def build_pointcloud_from_local_hits(
    local_hits: np.ndarray,
    pose: PoseStamped,
    *,
    lidar_mount_x_m: float,
    lidar_mount_y_m: float,
    free_ray_step_m: float,
    free_ray_start_m: float,
    free_height_m: float,
    obstacle_height_m: float,
    frame_id: str,
    timestamp: float,
) -> PointCloud2:
    if local_hits.size == 0:
        return PointCloud2.from_numpy(
            np.zeros((0, 3), dtype=np.float32),
            frame_id=frame_id,
            timestamp=float(timestamp),
        )

    free_points_local: list[np.ndarray] = []
    obstacle_points_local = np.zeros((len(local_hits), 2), dtype=np.float32)
    obstacle_intensities = np.zeros((len(local_hits),), dtype=np.float32)

    for index, (forward_m, lateral_m, distance_m, confidence) in enumerate(local_hits):
        obstacle_points_local[index, 0] = float(forward_m)
        obstacle_points_local[index, 1] = float(lateral_m)
        obstacle_intensities[index] = float(confidence)
        ray_points = _ray_samples_local(
            forward_m=float(forward_m),
            lateral_m=float(lateral_m),
            distance_m=float(distance_m),
            step_m=float(free_ray_step_m),
            start_m=float(free_ray_start_m),
        )
        if ray_points.size != 0:
            free_points_local.append(ray_points)

    free_world_xy = (
        _transform_xy_with_mount(
            np.vstack(free_points_local).astype(np.float32, copy=False),
            pose,
            mount_x_m=lidar_mount_x_m,
            mount_y_m=lidar_mount_y_m,
        )
        if free_points_local
        else np.zeros((0, 2), dtype=np.float32)
    )
    obstacle_world_xy = _transform_xy_with_mount(
        obstacle_points_local,
        pose,
        mount_x_m=lidar_mount_x_m,
        mount_y_m=lidar_mount_y_m,
    )

    free_world_xyz = (
        np.column_stack(
            (
                free_world_xy[:, 0],
                free_world_xy[:, 1],
                np.full((len(free_world_xy),), float(free_height_m), dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        if len(free_world_xy) > 0
        else np.zeros((0, 3), dtype=np.float32)
    )
    obstacle_world_xyz = np.column_stack(
        (
            obstacle_world_xy[:, 0],
            obstacle_world_xy[:, 1],
            np.full((len(obstacle_world_xy),), float(obstacle_height_m), dtype=np.float32),
        )
    ).astype(np.float32, copy=False)

    points_world_xyz = np.vstack((free_world_xyz, obstacle_world_xyz)).astype(np.float32, copy=False)
    free_intensities = np.zeros((len(free_world_xyz),), dtype=np.float32)
    intensities = np.concatenate((free_intensities, obstacle_intensities.astype(np.float32, copy=False)))

    return PointCloud2.from_numpy(
        points_world_xyz,
        frame_id=frame_id,
        timestamp=float(timestamp),
        intensities=intensities,
    )


def _merge_stationary_local_hits(
    hit_batches: list[np.ndarray],
    *,
    voxel_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not hit_batches:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    merged = np.vstack([batch for batch in hit_batches if batch.size != 0]).astype(np.float32, copy=False)
    if merged.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    voxel = max(float(voxel_m), 1e-3)
    cells = np.round(merged[:, :2] / voxel).astype(np.int64, copy=False)
    _, last_in_reversed = np.unique(cells[::-1], axis=0, return_index=True)
    keep_indices = np.sort((len(cells) - 1) - last_in_reversed)
    deduped = merged[keep_indices].astype(np.float32, copy=False)
    deduped[:, 2] = np.linalg.norm(deduped[:, :2], axis=1).astype(np.float32, copy=False)
    if deduped.shape[1] > 3:
        deduped[:, 3] = np.maximum(deduped[:, 3], 1.0)

    if len(deduped) > max_points:
        deduped = _downsample_points(deduped, max_points).astype(np.float32, copy=False)

    return deduped, deduped[:, :2].astype(np.float32, copy=False)


def build_registered_scan_from_local_points(
    local_points_xy: np.ndarray,
    pose: PoseStamped,
    *,
    z_height_m: float,
    frame_id: str,
    timestamp: float,
    lidar_mount_x_m: float,
    lidar_mount_y_m: float,
) -> PointCloud2:
    if local_points_xy.size == 0:
        return PointCloud2.from_numpy(np.zeros((0, 3), dtype=np.float32), frame_id=frame_id, timestamp=timestamp)

    sensor_x, sensor_y = _sensor_origin_xy(
        pose,
        mount_x_m=lidar_mount_x_m,
        mount_y_m=lidar_mount_y_m,
    )
    world_xy = _transform_local_points(
        local_points_xy,
        x=float(sensor_x),
        y=float(sensor_y),
        yaw=float(pose.yaw),
    )
    world_xyz = np.column_stack(
        (
            world_xy[:, 0],
            world_xy[:, 1],
            np.full((len(world_xy),), float(z_height_m), dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    return PointCloud2.from_numpy(world_xyz, frame_id=frame_id, timestamp=timestamp)


class SourcceyLidarPointCloudAdapter(Module):
    config: SourcceyLidarPointCloudAdapterConfig

    scan: In[PlanarLidarScan]
    odom: In[PoseStamped]
    # Optional: the commanded velocity reaching the base (post safety gate). When
    # connected, it is the authoritative "robot is moving" signal for gating
    # mapping (see SourcceyLidarPointCloudAdapterConfig.cmd_vel_gate_enabled).
    cmd_vel: In[Twist]
    lidar: Out[PointCloud2]
    registered_scan: Out[PointCloud2]
    localized_pose: Out[PoseStamped]
    obstacle_memory: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._prev_odom: PoseStamped | None = None
        self._scan_count = 0
        self._committed_scan_count = 0
        self._last_local_points: np.ndarray | None = None
        self._last_matched_pose: PoseStamped | None = None
        # The pose at which the current submap registration anchor was established
        # (the most recent committed pose). Unlike ``_last_matched_pose`` this is
        # NOT cleared when a single scan match is rejected, so anchor growth while
        # stationary stays pinned to the last trusted location instead of drifting
        # back onto raw odometry.
        self._anchor_pose: PoseStamped | None = None
        # Raw odom reading captured at the moment of the last commit. The scan-match
        # seed is the anchor pose composed with (current_raw_odom - odom_at_anchor),
        # so the seed tracks ALL motion (drive AND turn) since the last accepted
        # snapshot instead of a single odom message delta.
        self._odom_at_anchor: PoseStamped | None = None
        self._submap_points_world = np.zeros((0, 2), dtype=np.float32)
        self._last_turn_motion_wall_ts: float | None = None
        self._last_cmd_vel_wall_ts: float = 0.0
        self._last_motion_cmd_wall_ts: float = 0.0
        self._motion_epoch = 0
        self._last_committed_motion_epoch = 0
        self._stationary_keyframe_motion_epoch = 0
        self._stationary_scan_streak = 0
        self._stationary_keyframe_hits: list[np.ndarray] = []
        self._stationary_keyframe_committed = False
        self._obstacle_memory_points_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_memory_timestamps = np.zeros((0,), dtype=np.float64)
        self._last_pose_resolve_reason = "startup"
        # Commit classification for the most recent scan. ``None`` means the scan
        # was NOT committed (settling / holdoff / rejected / insufficient): in that
        # state nothing pose-dependent (blue map, yellow registered scan, green
        # memory update, localized pose) may be published from it.
        self._last_commit_kind: str | None = None
        self._last_reference_world_count = 0
        self._last_local_points_count = 0
        self._last_seed_pose: PoseStamped | None = None
        self._last_turn_recovery_candidate_pose: PoseStamped | None = None
        self._last_turn_recovery_candidate_score: float | None = None
        self._last_turn_recovery_candidate_overlap: float | None = None
        self._last_turn_recovery_candidate_reject_reason: str | None = None
        self._debug_skip_count = 0
        self._last_match_overlap_fraction = 0.0
        self._last_traced_skip_reason: str | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        trace_event(
            "lidar_pointcloud_adapter",
            "start",
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            min_range_m=float(self.config.min_range_m),
            max_distance_m=float(self.config.max_distance_m),
            mapping_requires_stationary_scans=int(self.config.mapping_requires_stationary_scans),
            mapping_holdoff_after_turn_s=float(self.config.mapping_holdoff_after_turn_s),
            scan_match_enabled=bool(self.config.scan_match_enabled),
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_odom(self, msg: PoseStamped) -> None:
        if self._latest_odom is not None:
            prev = self._latest_odom
            dt = max(float(msg.ts) - float(prev.ts), 1e-3)
            dyaw = _wrap_angle_rad(float(msg.yaw) - float(prev.yaw))
            angular_speed = abs(float(dyaw) / dt)
            dx = float(msg.x) - float(prev.x)
            dy = float(msg.y) - float(prev.y)
            linear_speed = math.hypot(dx, dy) / dt
            if (
                angular_speed > float(self.config.max_odom_angular_speed_for_mapping_rad_s)
                or linear_speed > float(self.config.max_odom_linear_speed_for_mapping_m_s)
            ):
                self._last_turn_motion_wall_ts = time.time()
                # Treat real odom motion as authoritative too: the previous
                # cmd_vel-only reset let a "stationary" keyframe survive while the
                # robot was still physically turning, which then splattered a stale
                # snapshot across the room once the matcher accepted it.
                self._motion_epoch += 1
                self._reset_stationary_keyframe()
        self._prev_odom = self._latest_odom
        self._latest_odom = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        now = time.time()
        self._last_cmd_vel_wall_ts = now
        eps = max(float(self.config.cmd_vel_motion_epsilon), 0.0)
        moving = (
            abs(float(msg.linear.x)) > eps
            or abs(float(msg.linear.y)) > eps
            or abs(float(msg.angular.z)) > eps
        )
        if moving:
            self._last_motion_cmd_wall_ts = now
            self._motion_epoch += 1
            self._reset_stationary_keyframe()

    def _cmd_vel_stream_active(self) -> bool:
        if not bool(self.config.cmd_vel_gate_enabled):
            return False
        return (time.time() - float(self._last_cmd_vel_wall_ts)) <= float(
            self.config.cmd_vel_active_window_s
        )

    def _hold_mapping_for_turn(self) -> bool:
        now = time.time()
        holdoff_s = float(self.config.mapping_holdoff_after_turn_s)
        # A motion command alone is not enough: the robot can still be physically
        # settling after the command stream goes back to zero. Keep holdoff active
        # until BOTH the recent motion command window and the recent odom-motion
        # window have gone quiet. This is deliberately conservative because the
        # user wants clean snapshots, not continuous in-motion updates.
        cmd_vel_hold = False
        if self._cmd_vel_stream_active():
            cmd_vel_hold = (now - float(self._last_motion_cmd_wall_ts)) < holdoff_s

        odom_hold = False
        if self._last_turn_motion_wall_ts is not None:
            odom_hold = (now - float(self._last_turn_motion_wall_ts)) < holdoff_s

        return cmd_vel_hold or odom_hold

    def _reset_stationary_keyframe(self) -> None:
        self._stationary_scan_streak = 0
        self._stationary_keyframe_hits.clear()
        self._stationary_keyframe_committed = False

    def _has_moved_since_last_commit(self) -> bool:
        odom = self._latest_odom
        odom0 = self._odom_at_anchor
        if odom is None or odom0 is None:
            return False
        dx = float(odom.x) - float(odom0.x)
        dy = float(odom.y) - float(odom0.y)
        dyaw_deg = abs(math.degrees(_wrap_angle_rad(float(odom.yaw) - float(odom0.yaw))))
        return (
            math.hypot(dx, dy) >= float(self.config.stationary_commit_reset_translation_m)
            or dyaw_deg >= float(self.config.stationary_commit_reset_rotation_deg)
        )

    def _accumulate_stationary_keyframe(
        self,
        local_hits: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        if self._hold_mapping_for_turn():
            self._reset_stationary_keyframe()
            return None, None, "turn_holdoff"

        if self._stationary_scan_streak <= 0:
            self._stationary_keyframe_motion_epoch = int(self._motion_epoch)

        self._stationary_scan_streak += 1

        if local_hits.size != 0:
            self._stationary_keyframe_hits.append(local_hits.astype(np.float32, copy=False))

        max_batches = max(
            int(self.config.stationary_keyframe_buffer_scans),
            int(self.config.mapping_requires_stationary_scans),
            1,
        )
        if len(self._stationary_keyframe_hits) > max_batches:
            self._stationary_keyframe_hits = self._stationary_keyframe_hits[-max_batches:]

        if self._stationary_keyframe_committed and self._has_moved_since_last_commit():
            self._reset_stationary_keyframe()
            if local_hits.size != 0:
                self._stationary_scan_streak = 1
                self._stationary_keyframe_hits.append(local_hits.astype(np.float32, copy=False))

        if self._stationary_keyframe_committed:
            return None, None, "stationary_committed"

        if int(self._stationary_keyframe_motion_epoch) != int(self._motion_epoch):
            self._reset_stationary_keyframe()
            return None, None, "motion_interrupted_keyframe"

        required = max(int(self.config.mapping_requires_stationary_scans), 1)
        if self._stationary_scan_streak < required:
            return None, None, "snapshot_settling"

        merged_hits, merged_points_xy = _merge_stationary_local_hits(
            self._stationary_keyframe_hits,
            voxel_m=float(self.config.stationary_keyframe_voxel_m),
            max_points=max(int(self.config.stationary_keyframe_max_points), 1),
        )
        if merged_points_xy.size == 0:
            return None, None, "insufficient_points"
        return merged_hits, merged_points_xy, "snapshot_ready"

    def _debug_log(self, tag: str, **fields: Any) -> None:
        if not bool(self.config.mapping_debug_enabled):
            return
        logger.info(f"[lidar_map.{tag}]", **fields)

    def _reference_world_points(self) -> np.ndarray:
        if self._submap_points_world.size != 0:
            return self._submap_points_world
        if self._last_local_points is None or self._last_matched_pose is None:
            return np.zeros((0, 2), dtype=np.float32)
        sensor_x, sensor_y = _sensor_origin_xy(
            self._last_matched_pose,
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        return _transform_local_points(
            _downsample_points(self._last_local_points, max(int(self.config.scan_match_max_points), 24)),
            x=float(sensor_x),
            y=float(sensor_y),
            yaw=float(self._last_matched_pose.yaw),
        )

    def _seed_pose_from_odom(self, odom: PoseStamped | None) -> PoseStamped | None:
        """Seed the scan match by composing odom motion since the last commit.

        The seed = anchor_pose ⊕ (current_raw_odom ⊖ odom_at_anchor). This tracks
        the FULL relative motion (translation and rotation) accumulated since the
        last accepted snapshot, so after driving or turning the seed lands near
        the true new pose and scan-matching only has to refine the small residual.

        Earlier this only added a single odom-message delta to the last matched
        pose (and fell back to raw odom after any rejection), so the seed lagged
        far behind the robot — the dominant reason matches kept failing.
        """
        if odom is None:
            return None
        anchor = self._anchor_pose
        odom0 = self._odom_at_anchor
        if anchor is None or odom0 is None:
            # No committed anchor yet: raw odom provisionally defines the frame.
            return odom

        dx = float(odom.x) - float(odom0.x)
        dy = float(odom.y) - float(odom0.y)
        translation_m = math.hypot(dx, dy)
        low_motion_trigger_m = max(float(self.config.scan_match_low_motion_trigger_m), 0.0)
        max_seed_step_m = max(float(self.config.max_seed_odom_step_m), 0.0)
        if translation_m <= low_motion_trigger_m:
            dx = 0.0
            dy = 0.0
        elif max_seed_step_m > 0.0 and translation_m > max_seed_step_m:
            scale = max_seed_step_m / max(translation_m, 1e-9)
            dx *= scale
            dy *= scale

        dyaw = _wrap_angle_rad(float(odom.yaw) - float(odom0.yaw))
        max_seed_step_rad = math.radians(max(float(self.config.max_seed_odom_step_deg), 0.0))
        if max_seed_step_rad > 0.0 and abs(dyaw) > max_seed_step_rad:
            dyaw = math.copysign(max_seed_step_rad, dyaw)
        # The committed map frame may have diverged from the raw odom frame by a
        # fixed rotation (the accumulated scan-match correction). Rotate the
        # odom-frame displacement into the map frame before applying it.
        theta = _wrap_angle_rad(float(anchor.yaw) - float(odom0.yaw))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        seed_x = float(anchor.x) + (cos_t * dx) - (sin_t * dy)
        seed_y = float(anchor.y) + (sin_t * dx) + (cos_t * dy)
        seed_yaw = _wrap_angle_rad(float(anchor.yaw) + dyaw)
        return PoseStamped(
            ts=float(odom.ts),
            frame_id=str(anchor.frame_id or self.config.frame_id),
            position=Vector3(seed_x, seed_y, float(anchor.z)),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, seed_yaw)),
        )

    def _clear_match_context(self, *, clear_submap: bool = False) -> None:
        self._last_local_points = None
        self._last_matched_pose = None
        if clear_submap:
            self._submap_points_world = np.zeros((0, 2), dtype=np.float32)
            # The submap *is* the anchor; if it is gone the anchor location is no
            # longer meaningful and must be re-bootstrapped from scratch.
            self._anchor_pose = None
            self._odom_at_anchor = None

    def _should_reset_match_context(self, seed_pose: PoseStamped) -> bool:
        """Hard-reset only when the robot has clearly left the mapped neighborhood.

        This is a last-resort recovery for true tracking loss / odom teleports.
        It is deliberately TRANSLATION-ONLY and measured against the anchor: a
        pure in-place rotation (turning to look around) keeps the robot inside the
        submap and must NEVER wipe the map — the previous rotation-based reset
        nuked the whole map on every normal turn.
        """
        anchor = self._anchor_pose
        if anchor is None:
            return False
        dx = float(seed_pose.x) - float(anchor.x)
        dy = float(seed_pose.y) - float(anchor.y)
        distance_m = math.hypot(dx, dy)
        return distance_m > float(self.config.reset_context_translation_m)

    def _refine_pose_with_scan_match(
        self,
        *,
        ts: float,
        seed_pose: PoseStamped,
        current_local_points: np.ndarray,
        reference_world: np.ndarray,
        translation_window_m: float | None = None,
    ) -> tuple[PoseStamped | None, float, float]:
        if reference_world.size == 0 or current_local_points.size == 0:
            return None, float("inf"), 0.0

        source_local = _downsample_points(
            current_local_points,
            max(int(self.config.scan_match_max_points), 24),
        )
        # The brute-force search evaluates ~470 candidate poses, each an
        # O(len(source) * len(reference)) distance sweep. A room-spanning submap
        # can hold a couple thousand points, so cap the reference density here to
        # keep a single snapshot match well under the settle window. The cap is
        # still dense enough (relative to the 0.08 m overlap radius) that a wall
        # outline stays continuous for overlap scoring.
        raw_reference_points = int(len(reference_world))
        reference_world = _downsample_points(
            reference_world,
            max(int(self.config.scan_match_max_points) * 5, 200),
        )
        best_x = float(seed_pose.x)
        best_y = float(seed_pose.y)
        best_yaw = float(seed_pose.yaw)
        seed_sensor_x, seed_sensor_y = _sensor_origin_xy(
            seed_pose,
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        best_score, best_overlap = _scan_match_metrics(
            reference_world,
            _transform_local_points(
                source_local,
                x=seed_sensor_x,
                y=seed_sensor_y,
                yaw=best_yaw,
            ),
            overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
        )

        if translation_window_m is None:
            translation_window = max(float(self.config.scan_match_translation_window_m), 0.02)
        else:
            translation_window = max(float(translation_window_m), 0.02)
        rotation_window = math.radians(max(float(self.config.scan_match_rotation_window_deg), 1.0))
        search_levels = (
            (translation_window, rotation_window, 3),
            (translation_window * 0.4, rotation_window * 0.4, 2),
        )

        for trans_step, yaw_step, n_steps in search_levels:
            offsets = range(-n_steps, n_steps + 1)
            for dx_idx in offsets:
                for dy_idx in offsets:
                    for dyaw_idx in offsets:
                        cand_x = best_x + dx_idx * trans_step
                        cand_y = best_y + dy_idx * trans_step
                        cand_yaw = _wrap_angle_rad(best_yaw + dyaw_idx * yaw_step)
                        sensor_x = cand_x + (
                            float(self.config.lidar_mount_x_m) * math.cos(cand_yaw)
                            - float(self.config.lidar_mount_y_m) * math.sin(cand_yaw)
                        )
                        sensor_y = cand_y + (
                            float(self.config.lidar_mount_x_m) * math.sin(cand_yaw)
                            + float(self.config.lidar_mount_y_m) * math.cos(cand_yaw)
                        )
                        score, overlap = _scan_match_metrics(
                            reference_world,
                            _transform_local_points(source_local, x=sensor_x, y=sensor_y, yaw=cand_yaw),
                            overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
                        )
                        if score < best_score:
                            best_score = score
                            best_overlap = overlap
                            best_x = cand_x
                            best_y = cand_y
                            best_yaw = cand_yaw

        accept_score = float(self.config.scan_match_accept_score_m)
        min_overlap = float(self.config.scan_match_min_overlap_fraction)
        if raw_reference_points >= max(int(self.config.scan_match_strict_reference_points), 1):
            accept_score = min(accept_score, float(self.config.scan_match_strict_accept_score_m))
            min_overlap = max(min_overlap, float(self.config.scan_match_strict_min_overlap_fraction))

        if best_score > accept_score or best_overlap < min_overlap:
            return None, best_score, best_overlap

        return (
            PoseStamped(
                ts=float(ts),
                frame_id=self.config.frame_id,
                position=Vector3(best_x, best_y, float(seed_pose.z)),
                    orientation=Quaternion.from_euler(Vector3(0.0, 0.0, best_yaw)),
            ),
            best_score,
            best_overlap,
        )

    def _bootstrap_pose(self, odom_seed: PoseStamped) -> PoseStamped:
        """Pose to commit a bootstrap / anchor-growth scan at.

        At a true cold start (no anchor yet) the odom seed defines the world
        origin. When an anchor already exists but is still being grown (undersized
        submap), reuse the established anchor pose so every growth scan lands at
        the same trusted location instead of following drifting raw odometry.
        """
        if self._anchor_pose is not None:
            anchor = self._anchor_pose
            return PoseStamped(
                ts=float(odom_seed.ts),
                frame_id=str(anchor.frame_id or self.config.frame_id),
                position=Vector3(float(anchor.x), float(anchor.y), float(anchor.z)),
                orientation=Quaternion.from_euler(Vector3(0.0, 0.0, float(anchor.yaw))),
            )
        return odom_seed

    def _scan_match_translation_window_for_seed(self, seed_pose: PoseStamped) -> float:
        translation_window = max(float(self.config.scan_match_translation_window_m), 0.02)
        anchor = self._anchor_pose
        if anchor is None:
            return translation_window
        low_motion_trigger_m = max(float(self.config.scan_match_low_motion_trigger_m), 0.0)
        low_motion_window_m = max(float(self.config.scan_match_low_motion_translation_window_m), 0.02)
        # Pure in-place turns are low-motion for the base center but not for an
        # off-center LiDAR head. During a turn the sensor sweeps an arc, so using
        # base translation here incorrectly collapsed the search window to the
        # "low motion" value and caused every post-turn snapshot to be rejected.
        seed_motion_m = _sensor_origin_displacement_m(
            anchor,
            seed_pose,
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        if seed_motion_m <= low_motion_trigger_m:
            return min(translation_window, low_motion_window_m)
        return translation_window

    def _should_try_turn_snapshot_relocalization(self, previous_pose: PoseStamped | None) -> bool:
        if previous_pose is None:
            return False
        return int(self._stationary_keyframe_motion_epoch) != int(self._last_committed_motion_epoch)

    def _refine_pose_with_turn_snapshot_search(
        self,
        *,
        ts: float,
        seed_pose: PoseStamped,
        previous_pose: PoseStamped,
        current_local_points: np.ndarray,
        reference_world: np.ndarray,
    ) -> tuple[PoseStamped | None, float, float]:
        if reference_world.size == 0 or current_local_points.size == 0:
            return None, float("inf"), 0.0

        source_local = _downsample_points(
            current_local_points,
            max(int(self.config.scan_match_max_points), 24),
        )
        raw_reference_points = int(len(reference_world))
        reference_world = _downsample_points(
            reference_world,
            max(int(self.config.scan_match_max_points) * 5, 200),
        )

        mount_x = float(self.config.lidar_mount_x_m)
        mount_y = float(self.config.lidar_mount_y_m)

        def _candidate_metrics(cand_x: float, cand_y: float, cand_yaw: float) -> tuple[float, float]:
            sensor_x = cand_x + (mount_x * math.cos(cand_yaw)) - (mount_y * math.sin(cand_yaw))
            sensor_y = cand_y + (mount_x * math.sin(cand_yaw)) + (mount_y * math.cos(cand_yaw))
            score, overlap = _scan_match_metrics(
                reference_world,
                _transform_local_points(source_local, x=sensor_x, y=sensor_y, yaw=cand_yaw),
                overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
            )
            return score, overlap

        seed_x = float(seed_pose.x)
        seed_y = float(seed_pose.y)
        yaw_center = float(previous_pose.yaw)
        best_x = seed_x
        best_y = seed_y
        best_yaw = yaw_center
        best_score, best_overlap = _candidate_metrics(best_x, best_y, best_yaw)

        coarse_half_width_deg = max(float(self.config.turn_snapshot_search_half_width_deg), 15.0)
        coarse_step_deg = max(float(self.config.turn_snapshot_search_step_deg), 1.0)
        refine_half_width_deg = max(float(self.config.turn_snapshot_search_refine_half_width_deg), coarse_step_deg)
        refine_step_deg = max(float(self.config.turn_snapshot_search_refine_step_deg), 0.5)
        translation_window = max(
            float(self.config.turn_snapshot_search_translation_window_m),
            self._scan_match_translation_window_for_seed(seed_pose),
            0.02,
        )
        search_levels = (
            (coarse_half_width_deg, coarse_step_deg, translation_window, 2),
            (refine_half_width_deg, refine_step_deg, max(translation_window * 0.35, 0.02), 1),
        )

        for level_index, (half_width_deg, step_deg, trans_step, n_steps) in enumerate(search_levels):
            if level_index == 0:
                center_x = seed_x
                center_y = seed_y
                center_yaw = yaw_center
            else:
                center_x = best_x
                center_y = best_y
                center_yaw = best_yaw
            yaw_offsets = np.arange(
                -half_width_deg,
                half_width_deg + (step_deg * 0.5),
                step_deg,
                dtype=np.float32,
            )
            offsets = range(-n_steps, n_steps + 1)
            for yaw_delta_deg in yaw_offsets:
                cand_yaw = _wrap_angle_rad(center_yaw + math.radians(float(yaw_delta_deg)))
                for dx_idx in offsets:
                    for dy_idx in offsets:
                        cand_x = center_x + (dx_idx * trans_step)
                        cand_y = center_y + (dy_idx * trans_step)
                        score, overlap = _candidate_metrics(cand_x, cand_y, cand_yaw)
                        if score < best_score:
                            best_score = score
                            best_overlap = overlap
                            best_x = cand_x
                            best_y = cand_y
                            best_yaw = cand_yaw

        accept_score = float(self.config.scan_match_accept_score_m)
        min_overlap = float(self.config.scan_match_min_overlap_fraction)
        if raw_reference_points >= max(int(self.config.scan_match_strict_reference_points), 1):
            accept_score = min(accept_score, float(self.config.scan_match_strict_accept_score_m))
            min_overlap = max(min_overlap, float(self.config.scan_match_strict_min_overlap_fraction))

        if best_score > accept_score or best_overlap < min_overlap:
            return None, best_score, best_overlap

        return (
            PoseStamped(
                ts=float(ts),
                frame_id=self.config.frame_id,
                position=Vector3(best_x, best_y, float(seed_pose.z)),
                orientation=Quaternion.from_euler(Vector3(0.0, 0.0, best_yaw)),
            ),
            best_score,
            best_overlap,
        )

    def _maybe_recover_turn_snapshot_pose(
        self,
        *,
        scan_ts: float,
        seed_pose: PoseStamped,
        previous_pose: PoseStamped | None,
        local_points_xy: np.ndarray,
        reference_world: np.ndarray,
        failed_reason: str,
        failed_score: float | None,
        failed_overlap: float | None,
    ) -> tuple[PoseStamped | None, str | None, float | None]:
        if not self._should_try_turn_snapshot_relocalization(previous_pose):
            return None, None, None

        recovered_pose, recovered_score, recovered_overlap = self._refine_pose_with_turn_snapshot_search(
            ts=float(scan_ts),
            seed_pose=seed_pose,
            previous_pose=previous_pose,
            current_local_points=local_points_xy,
            reference_world=reference_world,
        )
        recovered_seed_jump_m = None
        recovered_seed_yaw_jump_deg = None
        recovered_turn_from_previous_deg = None
        recovered_rejected = False
        recovered_reject_reason = None
        if recovered_pose is not None:
            self._last_turn_recovery_candidate_pose = recovered_pose
            self._last_turn_recovery_candidate_score = float(recovered_score)
            self._last_turn_recovery_candidate_overlap = float(recovered_overlap)
            recovered_seed_jump_m = math.hypot(
                float(recovered_pose.x) - float(seed_pose.x),
                float(recovered_pose.y) - float(seed_pose.y),
            )
            recovered_seed_yaw_jump_deg = abs(
                math.degrees(_wrap_angle_rad(float(recovered_pose.yaw) - float(seed_pose.yaw)))
            )
            recovered_turn_from_previous_deg = abs(
                math.degrees(_wrap_angle_rad(float(recovered_pose.yaw) - float(previous_pose.yaw)))
            )
            lidar_override_ok = (
                recovered_overlap >= float(self.config.turn_snapshot_lidar_override_overlap_floor)
                and recovered_score <= float(self.config.turn_snapshot_lidar_override_score_ceiling_m)
                and recovered_turn_from_previous_deg
                >= float(self.config.turn_snapshot_min_candidate_yaw_deg)
                and recovered_seed_jump_m
                <= float(self.config.turn_snapshot_lidar_override_max_translation_m)
            )
            if recovered_overlap < float(self.config.scan_match_min_overlap_fraction):
                recovered_rejected = True
                recovered_reject_reason = "below_min_overlap"
            elif (
                recovered_seed_jump_m > float(self.config.scan_match_seed_guard_translation_m)
                and not lidar_override_ok
            ):
                recovered_rejected = True
                recovered_reject_reason = "seed_translation_jump"
            elif (
                recovered_seed_yaw_jump_deg > float(self.config.scan_match_seed_guard_yaw_deg)
                and not lidar_override_ok
            ):
                recovered_rejected = True
                recovered_reject_reason = "seed_yaw_jump"
            elif (
                failed_overlap is not None
                and recovered_overlap + 1e-6 < float(failed_overlap)
                and failed_score is not None
                and recovered_score >= float(failed_score) - 1e-6
                and not lidar_override_ok
            ):
                recovered_rejected = True
                recovered_reject_reason = "worse_than_failed_candidate"
        self._last_turn_recovery_candidate_reject_reason = recovered_reject_reason
        trace_event(
            "lidar_pointcloud_adapter",
            "turn_snapshot_wide_search_attempt",
            failed_reason=failed_reason,
            failed_score=None if failed_score is None else round(float(failed_score), 4),
            failed_overlap=None if failed_overlap is None else round(float(failed_overlap), 3),
            reference_points=int(len(reference_world)),
            seed_pose_x=round(float(seed_pose.x), 4),
            seed_pose_y=round(float(seed_pose.y), 4),
            seed_pose_yaw_deg=round(math.degrees(float(seed_pose.yaw)), 2),
            previous_pose_x=round(float(previous_pose.x), 4),
            previous_pose_y=round(float(previous_pose.y), 4),
            previous_pose_yaw_deg=round(math.degrees(float(previous_pose.yaw)), 2),
            recovered=bool(recovered_pose is not None and not recovered_rejected),
            recovered_reject_reason=recovered_reject_reason,
            recovered_score=round(float(recovered_score), 4),
            recovered_overlap=round(float(recovered_overlap), 3),
            recovered_seed_jump_m=(
                None if recovered_seed_jump_m is None else round(float(recovered_seed_jump_m), 4)
            ),
            recovered_seed_yaw_jump_deg=(
                None
                if recovered_seed_yaw_jump_deg is None
                else round(float(recovered_seed_yaw_jump_deg), 2)
            ),
            recovered_turn_from_previous_deg=(
                None
                if recovered_turn_from_previous_deg is None
                else round(float(recovered_turn_from_previous_deg), 2)
            ),
            recovered_pose_x=None if recovered_pose is None else round(float(recovered_pose.x), 4),
            recovered_pose_y=None if recovered_pose is None else round(float(recovered_pose.y), 4),
            recovered_pose_yaw_deg=(
                None if recovered_pose is None else round(math.degrees(float(recovered_pose.yaw)), 2)
            ),
        )
        if recovered_pose is None or recovered_rejected:
            return None, None, None

        self._last_pose_resolve_reason = "turn_snapshot_wide_search_accepted"
        self._last_commit_kind = "matched"
        self._last_match_overlap_fraction = float(recovered_overlap)
        return recovered_pose, "matched", recovered_score

    def _should_allow_odom_turn_growth(
        self,
        *,
        seed_pose: PoseStamped,
        reference_count: int,
        score: float | None,
        overlap: float,
        previous_pose: PoseStamped | None = None,
        candidate_pose: PoseStamped | None = None,
    ) -> bool:
        if not bool(self.config.odom_turn_growth_enabled):
            return False
        anchor = self._anchor_pose
        if anchor is None:
            return False
        if reference_count <= 0:
            return False
        if reference_count > max(int(self.config.odom_turn_growth_reference_points_max), 1):
            return False

        dx = float(seed_pose.x) - float(anchor.x)
        dy = float(seed_pose.y) - float(anchor.y)
        translation_m = math.hypot(dx, dy)
        yaw_delta_deg = abs(
            math.degrees(_wrap_angle_rad(float(seed_pose.yaw) - float(anchor.yaw)))
        )
        if translation_m > float(self.config.odom_turn_growth_max_translation_m):
            return False
        if yaw_delta_deg < float(self.config.odom_turn_growth_min_turn_deg):
            return False
        if score is not None and float(score) > float(self.config.odom_turn_growth_max_score_m):
            return False
        if float(overlap) < float(self.config.odom_turn_growth_min_overlap_fraction):
            return False

        baseline_pose = previous_pose or anchor
        if candidate_pose is None:
            candidate_pose = self._last_turn_recovery_candidate_pose
        if candidate_pose is not None and baseline_pose is not None:
            seed_turn_signed_deg = math.degrees(
                _wrap_angle_rad(float(seed_pose.yaw) - float(baseline_pose.yaw))
            )
            candidate_turn_signed_deg = math.degrees(
                _wrap_angle_rad(float(candidate_pose.yaw) - float(baseline_pose.yaw))
            )
            seed_turn_deg = abs(float(seed_turn_signed_deg))
            candidate_turn_deg = abs(float(candidate_turn_signed_deg))
            candidate_seed_gap_deg = abs(
                math.degrees(_wrap_angle_rad(float(candidate_pose.yaw) - float(seed_pose.yaw)))
            )
            candidate_turn_ratio = candidate_turn_deg / max(seed_turn_deg, 1.0)
            opposite_turn_direction = (
                seed_turn_deg >= float(self.config.odom_turn_growth_candidate_sign_min_turn_deg)
                and candidate_turn_deg >= float(self.config.odom_turn_growth_candidate_sign_min_turn_deg)
                and (seed_turn_signed_deg * candidate_turn_signed_deg) < 0.0
            )
            candidate_turn_too_small = (
                candidate_seed_gap_deg
                > float(self.config.odom_turn_growth_candidate_yaw_margin_deg)
                and candidate_turn_ratio
                < float(self.config.odom_turn_growth_candidate_turn_ratio_floor)
            )
            if opposite_turn_direction or candidate_turn_too_small:
                trace_event(
                    "lidar_pointcloud_adapter",
                    "odom_turn_growth_denied_candidate_conflict",
                    seed_turn_deg=round(float(seed_turn_deg), 2),
                    candidate_turn_deg=round(float(candidate_turn_deg), 2),
                    candidate_seed_gap_deg=round(float(candidate_seed_gap_deg), 2),
                    candidate_turn_ratio=round(float(candidate_turn_ratio), 3),
                    opposite_turn_direction=bool(opposite_turn_direction),
                    candidate_reject_reason=self._last_turn_recovery_candidate_reject_reason,
                    candidate_overlap=(
                        None
                        if self._last_turn_recovery_candidate_overlap is None
                        else round(float(self._last_turn_recovery_candidate_overlap), 3)
                    ),
                    candidate_score=(
                        None
                        if self._last_turn_recovery_candidate_score is None
                        else round(float(self._last_turn_recovery_candidate_score), 4)
                    ),
                )
                return False
        return True

    def _should_allow_bootstrap_turn_seed_growth(
        self,
        *,
        seed_pose: PoseStamped,
        previous_pose: PoseStamped | None,
        reference_count: int,
        score: float | None,
        overlap: float,
    ) -> bool:
        if not bool(self.config.bootstrap_turn_seed_enabled):
            return False
        if previous_pose is None:
            return False
        if int(self._stationary_keyframe_motion_epoch) == int(self._last_committed_motion_epoch):
            return False
        if reference_count <= 0:
            return False
        if reference_count > max(int(self.config.bootstrap_turn_seed_reference_points_max), 1):
            return False
        if score is not None and float(score) > float(self.config.bootstrap_turn_seed_max_score_m):
            return False
        if float(overlap) > float(self.config.bootstrap_turn_seed_max_overlap_fraction):
            return False
        translation_m = math.hypot(
            float(seed_pose.x) - float(previous_pose.x),
            float(seed_pose.y) - float(previous_pose.y),
        )
        if translation_m > float(self.config.bootstrap_turn_seed_max_translation_m):
            return False
        seed_turn_deg = abs(
            math.degrees(_wrap_angle_rad(float(seed_pose.yaw) - float(previous_pose.yaw)))
        )
        if seed_turn_deg < float(self.config.bootstrap_turn_seed_min_turn_deg):
            return False
        return True

    def _resolve_scan_pose(
        self,
        *,
        scan_ts: float,
        local_points_xy: np.ndarray,
    ) -> tuple[PoseStamped | None, str | None, float | None]:
        """Resolve the pose to commit this scan at.

        Returns ``(pose, commit_kind, score)``. ``commit_kind`` is one of:
          * ``"bootstrap"`` – first/anchor-growth snapshot establishing the map
            origin (only allowed while fully stopped, with no usable fit yet);
          * ``"matched"``   – pose accepted by scan-matching against prior map;
          * ``"odom_only"`` – scan-matching disabled by config (raw odom mode);
          * ``None``        – NOT committed: caller must publish nothing
            pose-dependent (no blue map insert, no yellow scan, no green update,
            no localized pose), and keep the existing map memory as-is.
        """
        self._last_local_points_count = int(len(local_points_xy))
        self._last_commit_kind = None
        self._last_turn_recovery_candidate_pose = None
        self._last_turn_recovery_candidate_score = None
        self._last_turn_recovery_candidate_overlap = None
        self._last_turn_recovery_candidate_reject_reason = None
        raw_odom = self._latest_odom
        if raw_odom is None:
            self._last_pose_resolve_reason = "no_odom"
            logger.debug("Skipping Sourccey LiDAR scan because no odom has been received yet")
            return None, None, None

        odom = self._seed_pose_from_odom(raw_odom)
        self._last_seed_pose = odom

        now = time.time()
        if abs(now - float(raw_odom.ts)) > float(self.config.odom_stale_after_s):
            self._last_pose_resolve_reason = "stale_odom"
            logger.debug("Skipping Sourccey LiDAR scan because odom is stale")
            return None, None, None

        if self._should_reset_match_context(odom):
            logger.info(
                "Hard-resetting LiDAR scan-match context after large pose delta",
                odom_x=round(float(odom.x), 3),
                odom_y=round(float(odom.y), 3),
                odom_yaw_deg=round(math.degrees(float(odom.yaw)), 1),
            )
            trace_event(
                "lidar_pointcloud_adapter",
                "hard_reset_match_context",
                odom_x=round(float(odom.x), 3),
                odom_y=round(float(odom.y), 3),
                odom_yaw_deg=round(math.degrees(float(odom.yaw)), 1),
            )
            self._clear_match_context(clear_submap=True)
            self._obstacle_memory_points_xy = np.zeros((0, 2), dtype=np.float32)
            self._obstacle_memory_timestamps = np.zeros((0,), dtype=np.float64)

        if not bool(self.config.scan_match_enabled):
            self._last_reference_world_count = 0
            self._last_pose_resolve_reason = "odom_only"
            self._last_commit_kind = "odom_only"
            return odom, "odom_only", None

        reference_world = self._reference_world_points()
        self._last_reference_world_count = int(len(reference_world))

        # Cold start: no anchor at all yet. Bootstrap the map origin from the
        # current stationary snapshot (this DEFINES the world frame, so it is
        # trusted by construction — not a raw-odom "preview").
        if reference_world.size == 0:
            if local_points_xy.size == 0 or len(local_points_xy) < int(
                self.config.min_points_for_scan_match
            ):
                self._last_pose_resolve_reason = "insufficient_points"
                return None, None, None
            self._last_pose_resolve_reason = "bootstrap_anchor"
            self._last_commit_kind = "bootstrap"
            return self._bootstrap_pose(odom), "bootstrap", None

        min_reference_points = max(int(self.config.min_reference_points_for_scan_match), 1)
        if len(reference_world) < min_reference_points:
            # Anchor exists but is still too small to register against. Grow it by
            # committing more stationary points at the SAME anchor pose. We never
            # re-seed from raw odom here (that is what detached the map before):
            # while stationary the anchor pose is the trusted location.
            if len(local_points_xy) >= int(self.config.min_points_for_scan_match):
                self._last_pose_resolve_reason = "anchor_growth"
                self._last_commit_kind = "bootstrap"
                self._last_match_overlap_fraction = 0.0
                return self._bootstrap_pose(odom), "bootstrap", None

            self._last_pose_resolve_reason = "anchor_too_small"
            logger.warning(
                "Refusing LiDAR scan-match against undersized anchor",
                reference_points=int(len(reference_world)),
                min_reference_points=min_reference_points,
                local_points=int(len(local_points_xy)),
            )
            return None, None, None

        if local_points_xy.size == 0 or len(local_points_xy) < int(self.config.min_points_for_scan_match):
            self._last_pose_resolve_reason = "insufficient_points"
            return None, None, None

        translation_window_m = self._scan_match_translation_window_for_seed(odom)
        refined_pose, score, overlap = self._refine_pose_with_scan_match(
            ts=float(scan_ts),
            seed_pose=odom,
            current_local_points=local_points_xy,
            reference_world=reference_world,
            translation_window_m=translation_window_m,
        )
        if refined_pose is None:
            recovered_pose, recovered_kind, recovered_score = self._maybe_recover_turn_snapshot_pose(
                scan_ts=float(scan_ts),
                seed_pose=odom,
                previous_pose=self._last_matched_pose or self._anchor_pose,
                local_points_xy=local_points_xy,
                reference_world=reference_world,
                failed_reason="scan_match_rejected",
                failed_score=score,
                failed_overlap=overlap,
            )
            if recovered_pose is not None:
                return recovered_pose, recovered_kind, recovered_score
            if self._should_allow_bootstrap_turn_seed_growth(
                seed_pose=odom,
                previous_pose=self._last_matched_pose or self._anchor_pose,
                reference_count=int(len(reference_world)),
                score=score,
                overlap=overlap,
            ):
                self._last_pose_resolve_reason = "bootstrap_turn_seed"
                self._last_commit_kind = "bootstrap_turn_seed"
                self._last_match_overlap_fraction = float(overlap)
                trace_event(
                    "lidar_pointcloud_adapter",
                    "bootstrap_turn_seed_growth",
                    reference_points=int(len(reference_world)),
                    score=None if score is None else round(float(score), 4),
                    overlap_fraction=round(float(overlap), 3),
                    seed_pose_x=round(float(odom.x), 4),
                    seed_pose_y=round(float(odom.y), 4),
                    seed_pose_yaw_deg=round(math.degrees(float(odom.yaw)), 2),
                    previous_pose_x=round(float((self._last_matched_pose or self._anchor_pose).x), 4),
                    previous_pose_y=round(float((self._last_matched_pose or self._anchor_pose).y), 4),
                    previous_pose_yaw_deg=round(
                        math.degrees(float((self._last_matched_pose or self._anchor_pose).yaw)),
                        2,
                    ),
                )
                return odom, "bootstrap_turn_seed", score
            if self._should_allow_odom_turn_growth(
                seed_pose=odom,
                reference_count=int(len(reference_world)),
                score=score,
                overlap=overlap,
                previous_pose=self._last_matched_pose or self._anchor_pose,
            ):
                self._last_pose_resolve_reason = "odom_turn_growth_fallback"
                self._last_commit_kind = "odom_turn_growth"
                self._last_match_overlap_fraction = float(overlap)
                trace_event(
                    "lidar_pointcloud_adapter",
                    "odom_turn_growth_fallback",
                    reference_points=int(len(reference_world)),
                    score=None if score is None else round(float(score), 4),
                    overlap_fraction=round(float(overlap), 3),
                    seed_pose_x=round(float(odom.x), 4),
                    seed_pose_y=round(float(odom.y), 4),
                    seed_pose_yaw_deg=round(math.degrees(float(odom.yaw)), 2),
                    anchor_pose_x=round(float(self._anchor_pose.x), 4),
                    anchor_pose_y=round(float(self._anchor_pose.y), 4),
                    anchor_pose_yaw_deg=round(math.degrees(float(self._anchor_pose.yaw)), 2),
                )
                return odom, "odom_turn_growth", score
            # Reject: drop the short-lived local pose context, but PRESERVE the
            # accumulated submap so the next stationary scan can re-localize
            # against something real instead of smearing a detached second map.
            # Nothing pose-dependent is published this frame.
            self._clear_match_context(clear_submap=False)
            self._last_pose_resolve_reason = "scan_match_rejected"
            self._last_match_overlap_fraction = float(overlap)
            return None, None, score
        previous_pose = self._last_matched_pose or self._anchor_pose
        if previous_pose is not None:
            pose_jump_m = math.hypot(
                float(refined_pose.x) - float(previous_pose.x),
                float(refined_pose.y) - float(previous_pose.y),
            )
            yaw_jump_deg = abs(
                math.degrees(_wrap_angle_rad(float(refined_pose.yaw) - float(previous_pose.yaw)))
            )
            overlap_floor = float(self.config.scan_match_pose_jump_overlap_floor)
            jump_limit_m = float(self.config.scan_match_pose_jump_reject_m)
            yaw_jump_limit_deg = float(self.config.scan_match_yaw_jump_reject_deg)
            weak_overlap = float(overlap) < overlap_floor
            large_pose_jump = pose_jump_m > jump_limit_m
            large_yaw_jump = yaw_jump_deg > yaw_jump_limit_deg
            if weak_overlap and (large_pose_jump or large_yaw_jump):
                recovered_pose, recovered_kind, recovered_score = self._maybe_recover_turn_snapshot_pose(
                    scan_ts=float(scan_ts),
                    seed_pose=odom,
                    previous_pose=previous_pose,
                    local_points_xy=local_points_xy,
                    reference_world=reference_world,
                    failed_reason="scan_match_pose_jump_rejected",
                    failed_score=score,
                    failed_overlap=overlap,
                )
                if recovered_pose is not None:
                    return recovered_pose, recovered_kind, recovered_score
                if self._should_allow_bootstrap_turn_seed_growth(
                    seed_pose=odom,
                    previous_pose=previous_pose,
                    reference_count=int(len(reference_world)),
                    score=score,
                    overlap=overlap,
                ):
                    self._last_pose_resolve_reason = "bootstrap_turn_seed"
                    self._last_commit_kind = "bootstrap_turn_seed"
                    self._last_match_overlap_fraction = float(overlap)
                    trace_event(
                        "lidar_pointcloud_adapter",
                        "bootstrap_turn_seed_growth",
                        reference_points=int(len(reference_world)),
                        score=None if score is None else round(float(score), 4),
                        overlap_fraction=round(float(overlap), 3),
                        seed_pose_x=round(float(odom.x), 4),
                        seed_pose_y=round(float(odom.y), 4),
                        seed_pose_yaw_deg=round(math.degrees(float(odom.yaw)), 2),
                        previous_pose_x=round(float(previous_pose.x), 4),
                        previous_pose_y=round(float(previous_pose.y), 4),
                        previous_pose_yaw_deg=round(math.degrees(float(previous_pose.yaw)), 2),
                        rejected_candidate_x=round(float(refined_pose.x), 4),
                        rejected_candidate_y=round(float(refined_pose.y), 4),
                        rejected_candidate_yaw_deg=round(math.degrees(float(refined_pose.yaw)), 2),
                    )
                    return odom, "bootstrap_turn_seed", score
                if self._should_allow_odom_turn_growth(
                    seed_pose=odom,
                    reference_count=int(len(reference_world)),
                    score=score,
                    overlap=overlap,
                    previous_pose=previous_pose,
                    candidate_pose=refined_pose,
                ):
                    self._last_pose_resolve_reason = "odom_turn_growth_fallback"
                    self._last_commit_kind = "odom_turn_growth"
                    self._last_match_overlap_fraction = float(overlap)
                    trace_event(
                        "lidar_pointcloud_adapter",
                        "odom_turn_growth_fallback",
                        reference_points=int(len(reference_world)),
                        score=None if score is None else round(float(score), 4),
                        overlap_fraction=round(float(overlap), 3),
                        seed_pose_x=round(float(odom.x), 4),
                        seed_pose_y=round(float(odom.y), 4),
                        seed_pose_yaw_deg=round(math.degrees(float(odom.yaw)), 2),
                        anchor_pose_x=round(float(self._anchor_pose.x), 4),
                        anchor_pose_y=round(float(self._anchor_pose.y), 4),
                        anchor_pose_yaw_deg=round(math.degrees(float(self._anchor_pose.yaw)), 2),
                        rejected_candidate_x=round(float(refined_pose.x), 4),
                        rejected_candidate_y=round(float(refined_pose.y), 4),
                        rejected_candidate_yaw_deg=round(math.degrees(float(refined_pose.yaw)), 2),
                    )
                    return odom, "odom_turn_growth", score
                # Reject weak-overlap commits that would rotate or translate the
                # remembered obstacle layer a large distance relative to the last
                # trusted pose. These were the commits creating shifted green
                # corners/splatters in the trace after turn snapshots.
                self._clear_match_context(clear_submap=False)
                self._last_pose_resolve_reason = "scan_match_pose_jump_rejected"
                self._last_match_overlap_fraction = float(overlap)
                trace_event(
                    "lidar_pointcloud_adapter",
                    "scan_match_pose_jump_rejected",
                    pose_jump_m=round(float(pose_jump_m), 4),
                    yaw_jump_deg=round(float(yaw_jump_deg), 2),
                    translation_window_m=round(float(translation_window_m), 4),
                    overlap_fraction=round(float(overlap), 3),
                    score=None if score is None else round(float(score), 4),
                    previous_pose_x=round(float(previous_pose.x), 4),
                    previous_pose_y=round(float(previous_pose.y), 4),
                    previous_pose_yaw_deg=round(math.degrees(float(previous_pose.yaw)), 2),
                    candidate_pose_x=round(float(refined_pose.x), 4),
                    candidate_pose_y=round(float(refined_pose.y), 4),
                    candidate_pose_yaw_deg=round(math.degrees(float(refined_pose.yaw)), 2),
                )
                return None, None, score
        seed_pose_jump_m = math.hypot(
            float(refined_pose.x) - float(odom.x),
            float(refined_pose.y) - float(odom.y),
        )
        seed_yaw_jump_deg = abs(
            math.degrees(_wrap_angle_rad(float(refined_pose.yaw) - float(odom.yaw)))
        )
        if (
            float(overlap) < float(self.config.scan_match_seed_guard_overlap_floor)
            and (
                seed_pose_jump_m > float(self.config.scan_match_seed_guard_translation_m)
                or seed_yaw_jump_deg > float(self.config.scan_match_seed_guard_yaw_deg)
            )
        ):
            recovered_pose, recovered_kind, recovered_score = self._maybe_recover_turn_snapshot_pose(
                scan_ts=float(scan_ts),
                seed_pose=odom,
                previous_pose=previous_pose,
                local_points_xy=local_points_xy,
                reference_world=reference_world,
                failed_reason="scan_match_seed_guard_rejected",
                failed_score=score,
                failed_overlap=overlap,
            )
            if recovered_pose is not None:
                return recovered_pose, recovered_kind, recovered_score
            # Reject fits that "look" geometrically okay but still disagree too
            # much with the live IMU/odom seed. These are the bad stitches that
            # were slanting remembered front obstacles and twisting the room.
            self._clear_match_context(clear_submap=False)
            self._last_pose_resolve_reason = "scan_match_seed_guard_rejected"
            self._last_match_overlap_fraction = float(overlap)
            trace_event(
                "lidar_pointcloud_adapter",
                "scan_match_seed_guard_rejected",
                overlap_fraction=round(float(overlap), 3),
                score=None if score is None else round(float(score), 4),
                seed_pose_x=round(float(odom.x), 4),
                seed_pose_y=round(float(odom.y), 4),
                seed_pose_yaw_deg=round(math.degrees(float(odom.yaw)), 2),
                candidate_pose_x=round(float(refined_pose.x), 4),
                candidate_pose_y=round(float(refined_pose.y), 4),
                candidate_pose_yaw_deg=round(math.degrees(float(refined_pose.yaw)), 2),
                seed_pose_jump_m=round(float(seed_pose_jump_m), 4),
                seed_yaw_jump_deg=round(float(seed_yaw_jump_deg), 2),
            )
            return None, None, score
        if previous_pose is not None and int(self._stationary_keyframe_motion_epoch) != int(
            self._last_committed_motion_epoch
        ):
            seed_turn_from_last_commit_deg = abs(
                math.degrees(_wrap_angle_rad(float(odom.yaw) - float(previous_pose.yaw)))
            )
            candidate_turn_from_last_commit_deg = abs(
                math.degrees(_wrap_angle_rad(float(refined_pose.yaw) - float(previous_pose.yaw)))
            )
            if (
                seed_turn_from_last_commit_deg
                >= float(self.config.turn_snapshot_expected_seed_yaw_deg)
                and candidate_turn_from_last_commit_deg
                <= float(self.config.turn_snapshot_min_candidate_yaw_deg)
                and pose_jump_m >= float(self.config.turn_snapshot_stale_translation_floor_m)
                and float(overlap) >= float(self.config.turn_snapshot_stale_overlap_floor)
            ):
                recovered_pose, recovered_kind, recovered_score = self._maybe_recover_turn_snapshot_pose(
                    scan_ts=float(scan_ts),
                    seed_pose=odom,
                    previous_pose=previous_pose,
                    local_points_xy=local_points_xy,
                    reference_world=reference_world,
                    failed_reason="turn_snapshot_stale_pose_rejected",
                    failed_score=score,
                    failed_overlap=overlap,
                )
                if recovered_pose is not None:
                    return recovered_pose, recovered_kind, recovered_score
                # A real turn happened since the last committed snapshot, so a
                # candidate that "matches" by sliding a few centimeters while
                # barely changing yaw is not a trustworthy new map pose. Reject
                # it instead of letting the room freeze in place.
                self._clear_match_context(clear_submap=False)
                self._last_pose_resolve_reason = "turn_snapshot_stale_pose_rejected"
                self._last_match_overlap_fraction = float(overlap)
                trace_event(
                    "lidar_pointcloud_adapter",
                    "turn_snapshot_stale_pose_rejected",
                    overlap_fraction=round(float(overlap), 3),
                    score=None if score is None else round(float(score), 4),
                    pose_jump_m=round(float(pose_jump_m), 4),
                    seed_turn_from_last_commit_deg=round(float(seed_turn_from_last_commit_deg), 2),
                    candidate_turn_from_last_commit_deg=round(float(candidate_turn_from_last_commit_deg), 2),
                    previous_pose_x=round(float(previous_pose.x), 4),
                    previous_pose_y=round(float(previous_pose.y), 4),
                    previous_pose_yaw_deg=round(math.degrees(float(previous_pose.yaw)), 2),
                    seed_pose_x=round(float(odom.x), 4),
                    seed_pose_y=round(float(odom.y), 4),
                    seed_pose_yaw_deg=round(math.degrees(float(odom.yaw)), 2),
                    candidate_pose_x=round(float(refined_pose.x), 4),
                    candidate_pose_y=round(float(refined_pose.y), 4),
                    candidate_pose_yaw_deg=round(math.degrees(float(refined_pose.yaw)), 2),
                )
                return None, None, score
        self._last_pose_resolve_reason = "scan_match_accepted"
        self._last_commit_kind = "matched"
        self._last_match_overlap_fraction = float(overlap)
        return refined_pose, "matched", score

    def _update_submap(self, world_points_xy: np.ndarray, *, anchor_x: float, anchor_y: float) -> None:
        if world_points_xy.size == 0:
            return
        candidate = _downsample_points(world_points_xy, max(int(self.config.scan_match_max_points), 24))
        if self._submap_points_world.size == 0:
            merged = candidate.astype(np.float32, copy=True)
        else:
            merged = np.vstack((self._submap_points_world, candidate)).astype(np.float32, copy=False)

        # Keep the scan-matching reference local to the robot instead of letting
        # it grow into a global soup. The global map already lives downstream in
        # RayTracingVoxelMap; this submap should stay as a clean registration
        # anchor for the current neighborhood.
        local_radius_m = max(float(self.config.submap_local_radius_m), 0.25)
        deltas = merged - np.asarray((float(anchor_x), float(anchor_y)), dtype=np.float32)
        keep_mask = np.sum(deltas * deltas, axis=1) <= (local_radius_m * local_radius_m)
        local_points = merged[keep_mask]
        if local_points.size == 0:
            local_points = candidate.astype(np.float32, copy=True)

        self._submap_points_world = _downsample_points(
            local_points,
            max(int(self.config.submap_max_points), int(self.config.scan_match_max_points)),
        )

    def _publish_obstacle_memory(self, timestamp: float, *, visible_points_xy: np.ndarray | None = None) -> None:
        if not bool(self.config.obstacle_memory_enabled):
            return
        memory_points_xy = self._obstacle_memory_points_xy
        if visible_points_xy is not None and visible_points_xy.size != 0 and memory_points_xy.size != 0:
            clear_radius_m = max(float(self.config.obstacle_memory_visible_clear_radius_m), 1e-3)
            delta = memory_points_xy[:, None, :] - visible_points_xy[None, :, :]
            min_d2 = np.min(np.sum(delta * delta, axis=2), axis=1)
            keep_mask = min_d2 > (clear_radius_m * clear_radius_m)
            if len(keep_mask) == len(memory_points_xy):
                memory_points_xy = memory_points_xy[keep_mask]

        if memory_points_xy.size == 0:
            self.obstacle_memory.publish(
                PointCloud2.from_numpy(
                    np.zeros((0, 3), dtype=np.float32),
                    frame_id=self.config.frame_id,
                    timestamp=float(timestamp),
                )
            )
            return
        points_xyz = np.column_stack(
            (
                memory_points_xy[:, 0],
                memory_points_xy[:, 1],
                np.full(
                    (len(memory_points_xy),),
                    float(self.config.obstacle_memory_height_m),
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32, copy=False)
        self.obstacle_memory.publish(
            PointCloud2.from_numpy(
                points_xyz,
                frame_id=self.config.frame_id,
                timestamp=float(timestamp),
            )
        )

    def _publish_empty_obstacle_memory(self, timestamp: float) -> None:
        self.obstacle_memory.publish(
            PointCloud2.from_numpy(
                np.zeros((0, 3), dtype=np.float32),
                frame_id=self.config.frame_id,
                timestamp=float(timestamp),
            )
        )

    def _republish_obstacle_memory(self, timestamp: float) -> None:
        """Re-emit the retained obstacle memory unchanged (green persistence).

        Used on every NON-committed scan (settling, holdoff, rejected match, etc).
        Green points represent obstacles seen earlier that we are not re-observing
        with a trusted pose right now, so they must persist rather than blink to
        empty. Decay is still applied so genuinely stale memory ages out, but no
        new geometry is added and nothing is cleared by a current (untrusted) view.
        """
        if not bool(self.config.obstacle_memory_enabled):
            return
        now = float(timestamp)
        if self._obstacle_memory_timestamps.size != 0:
            keep_mask = (now - self._obstacle_memory_timestamps) <= float(
                self.config.obstacle_memory_decay_s
            )
            self._obstacle_memory_points_xy = self._obstacle_memory_points_xy[keep_mask]
            self._obstacle_memory_timestamps = self._obstacle_memory_timestamps[keep_mask]
        self._publish_obstacle_memory(timestamp, visible_points_xy=None)

    def _publish_empty_registered_scan(self, timestamp: float) -> None:
        self.registered_scan.publish(
            PointCloud2.from_numpy(
                np.zeros((0, 3), dtype=np.float32),
                frame_id=self.config.frame_id,
                timestamp=float(timestamp),
            )
        )

    def _update_obstacle_memory(self, world_points_xy: np.ndarray, *, timestamp: float) -> None:
        if not bool(self.config.obstacle_memory_enabled):
            return

        now = float(timestamp)
        if self._obstacle_memory_timestamps.size != 0:
            keep_mask = (now - self._obstacle_memory_timestamps) <= float(self.config.obstacle_memory_decay_s)
            self._obstacle_memory_points_xy = self._obstacle_memory_points_xy[keep_mask]
            self._obstacle_memory_timestamps = self._obstacle_memory_timestamps[keep_mask]

        if world_points_xy.size != 0:
            voxel = max(float(self.config.obstacle_memory_voxel_m), 1e-3)
            visible_cells, visible_points_xy = _voxelize_points(world_points_xy, voxel)

            if visible_cells.size != 0:
                replace_radius_m = max(
                    float(self.config.obstacle_memory_visible_clear_radius_m),
                    voxel * 1.5,
                )
                if self._obstacle_memory_points_xy.size != 0:
                    delta = self._obstacle_memory_points_xy[:, None, :] - visible_points_xy[None, :, :]
                    min_d2 = np.min(np.sum(delta * delta, axis=2), axis=1)
                    keep_mask = min_d2 > (replace_radius_m * replace_radius_m)
                    dropped = int(len(self._obstacle_memory_points_xy) - int(np.count_nonzero(keep_mask)))
                    if dropped > 0:
                        self._obstacle_memory_points_xy = self._obstacle_memory_points_xy[keep_mask]
                        self._obstacle_memory_timestamps = self._obstacle_memory_timestamps[keep_mask]
                        trace_event(
                            "lidar_pointcloud_adapter",
                            "obstacle_memory_replaced_visible_region",
                            dropped_points=dropped,
                            remaining_points=int(len(self._obstacle_memory_points_xy)),
                            replace_radius_m=round(float(replace_radius_m), 4),
                        )

                # Remembered obstacles are append-only world points from trusted
                # commits, but a newly trusted observation REPLACES nearby old
                # memory so small pose jitter does not build up a thick green wall
                # made of many copies of the same obstacle.
                if self._obstacle_memory_points_xy.size == 0:
                    merged_points = visible_points_xy.astype(np.float32, copy=True)
                    merged_timestamps = np.full((len(visible_points_xy),), now, dtype=np.float64)
                else:
                    merged_points = np.vstack((self._obstacle_memory_points_xy, visible_points_xy)).astype(
                        np.float32,
                        copy=False,
                    )
                    merged_timestamps = np.concatenate(
                        (
                            self._obstacle_memory_timestamps,
                            np.full((len(visible_points_xy),), now, dtype=np.float64),
                        )
                    )

                merged_cells = np.round(merged_points / voxel).astype(np.int64, copy=False)
                order_by_ts = np.argsort(merged_timestamps, kind="stable")
                cells_sorted = merged_cells[order_by_ts]
                _, last_in_reversed = np.unique(cells_sorted[::-1], axis=0, return_index=True)
                keep_indices = order_by_ts[len(cells_sorted) - 1 - last_in_reversed]
                if keep_indices.size != 0:
                    order = np.argsort(merged_timestamps[keep_indices])[::-1]
                    keep_indices = keep_indices[order]
                    max_points = max(int(self.config.obstacle_memory_max_points), 1)
                    keep_indices = keep_indices[:max_points]
                    self._obstacle_memory_points_xy = merged_points[keep_indices]
                    self._obstacle_memory_timestamps = merged_timestamps[keep_indices]

        self._publish_obstacle_memory(timestamp, visible_points_xy=world_points_xy)

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        local_points = _scan_to_local_points(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            invert_lateral_axis=bool(self.config.invert_lateral_axis),
            max_distance_m=float(self.config.max_distance_m),
            min_confidence=int(self.config.min_confidence),
            min_range_m=float(self.config.min_range_m),
        )
        local_points_xy = (
            np.asarray([(forward_m, lateral_m) for forward_m, lateral_m, _, _ in local_points], dtype=np.float32)
            if local_points
            else np.zeros((0, 2), dtype=np.float32)
        )
        local_hits = (
            np.asarray(local_points, dtype=np.float32)
            if local_points
            else np.zeros((0, 4), dtype=np.float32)
        )
        self._scan_count += 1
        self._last_local_points_count = int(len(local_points_xy))

        keyframe_hits, keyframe_points_xy, keyframe_reason = self._accumulate_stationary_keyframe(local_hits)
        if keyframe_points_xy is None or keyframe_hits is None:
            self._last_pose_resolve_reason = str(keyframe_reason)
            pose = None
            commit_kind = None
            score = None
        else:
            if self._hold_mapping_for_turn():
                self._reset_stationary_keyframe()
                self._last_pose_resolve_reason = "turn_holdoff_after_snapshot"
                pose = None
                commit_kind = None
                score = None
            else:
                self._last_local_points_count = int(len(keyframe_points_xy))
                pose, commit_kind, score = self._resolve_scan_pose(
                    scan_ts=float(scan.ts),
                    local_points_xy=keyframe_points_xy,
                )

        # ---------------------------------------------------------------
        # NOT COMMITTED: settling / holdoff / rejected / insufficient.
        # Nothing pose-dependent may be emitted. Yellow (current scan) is
        # cleared because we have no trusted current fit; green (memory) is
        # republished unchanged so earlier obstacles persist; the blue/global
        # map is NOT touched and no localized pose is published (downstream
        # keeps the last accepted pose). This is the core fix for the detached
        # preview overlay and the blinking/garbage map.
        # ---------------------------------------------------------------
        if pose is None or commit_kind is None:
            self._publish_empty_registered_scan(float(scan.ts))
            self._republish_obstacle_memory(float(scan.ts))
            self._debug_skip_count += 1
            skip_reason = str(self._last_pose_resolve_reason)
            if skip_reason != self._last_traced_skip_reason:
                trace_event(
                    "lidar_pointcloud_adapter",
                    "skip_map_insertion",
                    reason=skip_reason,
                    scan_count=int(self._scan_count),
                    local_points=int(self._last_local_points_count),
                    reference_points=int(self._last_reference_world_count),
                    submap_points=int(len(self._submap_points_world)),
                    memory_points=int(len(self._obstacle_memory_points_xy)),
                    keyframe_batches=int(len(self._stationary_keyframe_hits)),
                    holdoff_active=bool(self._hold_mapping_for_turn()),
                    stationary_streak=int(self._stationary_scan_streak),
                    score=None if score is None else round(float(score), 4),
                    overlap_fraction=round(float(self._last_match_overlap_fraction), 3),
                )
                self._last_traced_skip_reason = skip_reason
            if self._scan_count % max(int(self.config.log_every_scans), 1) == 0:
                logger.info(
                    "LiDAR snapshot not committed (no map insertion)",
                    reason=self._last_pose_resolve_reason,
                    holdoff_active=bool(self._hold_mapping_for_turn()),
                    stationary_streak=int(self._stationary_scan_streak),
                )
            if self._debug_skip_count % max(int(self.config.mapping_debug_log_every_skips), 1) == 0:
                self._debug_log(
                    "skip",
                    reason=self._last_pose_resolve_reason,
                    scan_count=self._scan_count,
                    local_points=self._last_local_points_count,
                    reference_points=self._last_reference_world_count,
                    submap_points=int(len(self._submap_points_world)),
                    memory_points=int(len(self._obstacle_memory_points_xy)),
                    keyframe_batches=int(len(self._stationary_keyframe_hits)),
                    holdoff_active=bool(self._hold_mapping_for_turn()),
                    insertion_allowed=False,
                    registered_scan_source="empty",
                    obstacle_memory_source="persisted",
                    score=None if score is None else round(float(score), 4),
                    overlap_fraction=round(float(self._last_match_overlap_fraction), 3),
                )
            return

        # ---------------------------------------------------------------
        # COMMITTED: the robot is fully stopped + settled and we have a trusted
        # pose (bootstrap anchor, scan-match fit, or odom-only mode). EVERY
        # pose-dependent output below is derived from this single ``pose`` so the
        # blue map, yellow current scan, green memory and localized pose can never
        # disagree about where the robot is.
        # ---------------------------------------------------------------
        cloud = build_pointcloud_from_scan(
            scan,
            pose,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            invert_lateral_axis=bool(self.config.invert_lateral_axis),
            lidar_mount_x_m=float(self.config.lidar_mount_x_m),
            lidar_mount_y_m=float(self.config.lidar_mount_y_m),
            max_distance_m=float(self.config.max_distance_m),
            min_confidence=int(self.config.min_confidence),
            free_ray_step_m=float(self.config.free_ray_step_m),
            free_ray_start_m=float(self.config.free_ray_start_m),
            free_height_m=float(self.config.free_height_m),
            obstacle_height_m=float(self.config.obstacle_height_m),
            frame_id=self.config.frame_id,
            min_range_m=float(self.config.min_range_m),
        )
        if keyframe_hits is not None:
            cloud = build_pointcloud_from_local_hits(
                keyframe_hits,
                pose,
                lidar_mount_x_m=float(self.config.lidar_mount_x_m),
                lidar_mount_y_m=float(self.config.lidar_mount_y_m),
                free_ray_step_m=float(self.config.free_ray_step_m),
                free_ray_start_m=float(self.config.free_ray_start_m),
                free_height_m=float(self.config.free_height_m),
                obstacle_height_m=float(self.config.obstacle_height_m),
                frame_id=self.config.frame_id,
                timestamp=float(scan.ts),
            )
        if len(cloud) == 0 and not bool(self.config.publish_empty_clouds):
            # Committed but produced no usable points: still keep green alive and
            # do not publish a bogus localized pose / empty insertion.
            self._publish_empty_registered_scan(float(scan.ts))
            self._republish_obstacle_memory(float(scan.ts))
            return

        # Blue / global map insertion (only ever reached while stopped + committed).
        self.lidar.publish(cloud)
        # The accepted pose that downstream (explorer, costmap exporter, TF) uses.
        self.localized_pose.publish(pose)
        # Yellow / currently-seen scan, from the SAME committed pose as the map.
        registered_scan = build_registered_scan_from_local_points(
            keyframe_points_xy,
            pose,
            z_height_m=float(self.config.registered_scan_height_m),
            frame_id=self.config.frame_id,
            timestamp=float(scan.ts),
            lidar_mount_x_m=float(self.config.lidar_mount_x_m),
            lidar_mount_y_m=float(self.config.lidar_mount_y_m),
        )
        self.registered_scan.publish(registered_scan)

        self._last_local_points = keyframe_points_xy
        self._last_matched_pose = pose
        # Persist the anchor location AND the raw odom reading at this commit so
        # undersized-anchor growth, post-reject recovery, and the next scan's odom
        # seed all stay pinned to the last trusted spot (see _bootstrap_pose and
        # _seed_pose_from_odom).
        self._anchor_pose = pose
        self._odom_at_anchor = self._latest_odom
        self._committed_scan_count += 1
        self._last_committed_motion_epoch = int(self._motion_epoch)
        self._stationary_keyframe_committed = True
        self._last_traced_skip_reason = None

        trusted_memory_commit = str(commit_kind) in {"bootstrap", "bootstrap_turn_seed", "odom_only"} or (
            str(commit_kind) == "matched"
            and float(self._last_match_overlap_fraction)
            >= float(self.config.obstacle_memory_match_overlap_floor)
            and (score is None or float(score) <= float(self.config.obstacle_memory_match_score_ceiling_m))
        )

        if keyframe_points_xy.size != 0:
            sensor_x, sensor_y = _sensor_origin_xy(
                pose,
                mount_x_m=float(self.config.lidar_mount_x_m),
                mount_y_m=float(self.config.lidar_mount_y_m),
            )
            world_points_xy = _transform_local_points(
                keyframe_points_xy,
                x=float(sensor_x),
                y=float(sensor_y),
                yaw=float(pose.yaw),
            )
            self._update_submap(world_points_xy, anchor_x=float(sensor_x), anchor_y=float(sensor_y))
            # Only trusted map commits may alter remembered obstacle geometry.
            # Low-trust bootstrap/growth assists must not drag the green memory
            # layer to a shifted/slanted position.
            if trusted_memory_commit:
                self._update_obstacle_memory(world_points_xy, timestamp=float(scan.ts))
            else:
                self._republish_obstacle_memory(float(scan.ts))
        else:
            self._republish_obstacle_memory(float(scan.ts))

        self._debug_skip_count = 0
        trace_event(
            "lidar_pointcloud_adapter",
            "commit_world_cloud",
            commit_kind=str(commit_kind),
            reason=str(self._last_pose_resolve_reason),
            scan_count=int(self._scan_count),
            committed_scans=int(self._committed_scan_count),
            local_points=int(self._last_local_points_count),
            reference_points=int(self._last_reference_world_count),
            submap_points=int(len(self._submap_points_world)),
            memory_points=int(len(self._obstacle_memory_points_xy)),
            pose_x=round(float(pose.x), 4),
            pose_y=round(float(pose.y), 4),
            pose_yaw_deg=round(math.degrees(float(pose.yaw)), 2),
            cmd_motion_age_s=None if self._last_motion_cmd_wall_ts <= 0.0 else round(time.time() - float(self._last_motion_cmd_wall_ts), 3),
            odom_motion_age_s=None if self._last_turn_motion_wall_ts is None else round(time.time() - float(self._last_turn_motion_wall_ts), 3),
            score=None if score is None else round(float(score), 4),
            overlap_fraction=round(float(self._last_match_overlap_fraction), 3),
        )
        if self._scan_count % max(int(self.config.log_every_scans), 1) == 0:
            logger.info(
                "SourcceyLidarPointCloudAdapter committed world cloud",
                scan_count=self._scan_count,
                committed_scans=self._committed_scan_count,
                points=len(cloud),
                commit_kind=commit_kind,
                pose_x=round(float(pose.x), 3),
                pose_y=round(float(pose.y), 3),
                pose_yaw_deg=round(math.degrees(float(pose.yaw)), 1),
                scan_match_score=None if score is None else round(float(score), 4),
            )
        if self._scan_count % max(int(self.config.mapping_debug_log_every_scans), 1) == 0:
            seed_pose = self._last_seed_pose
            pose_error_m = None
            pose_error_deg = None
            if seed_pose is not None:
                pose_error_m = round(
                    math.hypot(float(pose.x) - float(seed_pose.x), float(pose.y) - float(seed_pose.y)),
                    4,
                )
                pose_error_deg = round(
                    math.degrees(_wrap_angle_rad(float(pose.yaw) - float(seed_pose.yaw))),
                    2,
                )
            self._debug_log(
                "commit",
                reason=self._last_pose_resolve_reason,
                commit_kind=commit_kind,
                scan_count=self._scan_count,
                local_points=self._last_local_points_count,
                reference_points=self._last_reference_world_count,
                submap_points=int(len(self._submap_points_world)),
                memory_points=int(len(self._obstacle_memory_points_xy)),
                keyframe_batches=int(len(self._stationary_keyframe_hits)),
                insertion_allowed=True,
                registered_scan_source=f"committed:{commit_kind}",
                obstacle_memory_source=f"committed:{commit_kind}",
                pose_x=round(float(pose.x), 4),
                pose_y=round(float(pose.y), 4),
                pose_yaw_deg=round(math.degrees(float(pose.yaw)), 2),
                seed_pose_error_m=pose_error_m,
                seed_pose_error_deg=pose_error_deg,
                score=None if score is None else round(float(score), 4),
                overlap_fraction=round(float(self._last_match_overlap_fraction), 3),
            )
