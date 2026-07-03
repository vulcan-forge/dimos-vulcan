from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from reactivex.disposable import Disposable

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.module import logger
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

from .lidar_geometry import normalize_angle_deg
from .lidar_types import PlanarLidarScan
from .run_trace import trace_event

_DEFAULT_EXPORT_ROOT = DIMOS_PROJECT_ROOT / "assets" / "output" / "sourccey_maps"


def _blend_angle_rad(a: float, b: float, weight: float) -> float:
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return math.atan2(math.sin(a + delta * weight), math.cos(a + delta * weight))


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _ray_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps <= 0:
        return [(x0, y0)]
    xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
    ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)
    return list(zip(xs.tolist(), ys.tolist(), strict=False))


def _pose_xy_yaw(pose: PoseStamped) -> tuple[float, float, float]:
    return float(pose.x), float(pose.y), float(pose.yaw)


def _pose_from_xy_yaw(
    *,
    x: float,
    y: float,
    yaw: float,
    ts: float,
    frame_id: str,
) -> PoseStamped:
    return PoseStamped(
        ts=float(ts),
        frame_id=frame_id,
        position=Vector3(float(x), float(y), 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, float(yaw))),
    )


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indices]


def _voxelize_points(points_xy: np.ndarray, voxel_m: float) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    voxel = max(float(voxel_m), 1e-3)
    cells = np.round(points_xy / voxel).astype(np.int32, copy=False)
    _, unique_indices = np.unique(cells, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points_xy[unique_indices].astype(np.float32, copy=False)


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


def _sensor_origin_xy(
    *,
    x: float,
    y: float,
    yaw: float,
    mount_x_m: float,
    mount_y_m: float,
) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    sensor_x = float(x) + (float(mount_x_m) * cos_yaw) - (float(mount_y_m) * sin_yaw)
    sensor_y = float(y) + (float(mount_x_m) * sin_yaw) + (float(mount_y_m) * cos_yaw)
    return sensor_x, sensor_y


def _transform_local_points_with_mount(
    points_local_xy: np.ndarray,
    *,
    x: float,
    y: float,
    yaw: float,
    mount_x_m: float,
    mount_y_m: float,
) -> np.ndarray:
    if points_local_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    sensor_x, sensor_y = _sensor_origin_xy(
        x=float(x),
        y=float(y),
        yaw=float(yaw),
        mount_x_m=float(mount_x_m),
        mount_y_m=float(mount_y_m),
    )
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = points_local_xy @ rotation.T
    world[:, 0] += float(sensor_x)
    world[:, 1] += float(sensor_y)
    return world


def _transform_points_2d(
    points_xy: np.ndarray,
    *,
    x: float,
    y: float,
    yaw: float,
) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = points_xy @ rotation.T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    return world


def _relative_pose_2d(
    *,
    source_x: float,
    source_y: float,
    source_yaw: float,
    target_x: float,
    target_y: float,
    target_yaw: float,
) -> tuple[float, float, float]:
    dx_world = float(target_x) - float(source_x)
    dy_world = float(target_y) - float(source_y)
    cos_yaw = math.cos(float(source_yaw))
    sin_yaw = math.sin(float(source_yaw))
    rel_x = (dx_world * cos_yaw) + (dy_world * sin_yaw)
    rel_y = (-dx_world * sin_yaw) + (dy_world * cos_yaw)
    rel_yaw = _wrap_angle_rad(float(target_yaw) - float(source_yaw))
    return float(rel_x), float(rel_y), float(rel_yaw)


def _compose_pose_2d(
    *,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
    rel_x: float,
    rel_y: float,
    rel_yaw: float,
) -> tuple[float, float, float]:
    cos_yaw = math.cos(float(origin_yaw))
    sin_yaw = math.sin(float(origin_yaw))
    world_x = float(origin_x) + (float(rel_x) * cos_yaw) - (float(rel_y) * sin_yaw)
    world_y = float(origin_y) + (float(rel_x) * sin_yaw) + (float(rel_y) * cos_yaw)
    world_yaw = _wrap_angle_rad(float(origin_yaw) + float(rel_yaw))
    return float(world_x), float(world_y), float(world_yaw)


def _base_pose_from_sensor_xy_yaw(
    *,
    sensor_x: float,
    sensor_y: float,
    yaw: float,
    mount_x_m: float,
    mount_y_m: float,
) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    base_x = float(sensor_x) - (float(mount_x_m) * cos_yaw) + (float(mount_y_m) * sin_yaw)
    base_y = float(sensor_y) - (float(mount_x_m) * sin_yaw) - (float(mount_y_m) * cos_yaw)
    return float(base_x), float(base_y)


def _points_to_cloud(
    points_xy: np.ndarray,
    *,
    z_height_m: float,
    frame_id: str,
    timestamp: float,
    intensities: np.ndarray | None = None,
) -> PointCloud2:
    if points_xy.size == 0:
        return PointCloud2.from_numpy(
            np.zeros((0, 3), dtype=np.float32),
            frame_id=frame_id,
            timestamp=float(timestamp),
        )
    xyz = np.column_stack(
        (
            points_xy[:, 0],
            points_xy[:, 1],
            np.full((len(points_xy),), float(z_height_m), dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    return PointCloud2.from_numpy(
        xyz,
        frame_id=frame_id,
        timestamp=float(timestamp),
        intensities=intensities,
    )


def _scan_to_local_points(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_confidence: int,
    min_range_m: float,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
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
        if not math.isfinite(distance) or distance <= 0.0:
            continue
        if distance < float(min_range_m) or distance > float(max_distance_m):
            continue
        delta_deg = normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        if abs(delta_deg) > float(valid_angle_half_width_deg):
            continue
        theta = math.radians(delta_deg)
        forward_m = distance * math.cos(theta)
        lateral_m = distance * math.sin(theta)
        if invert_lateral_axis:
            lateral_m = -lateral_m
        points.append((forward_m, lateral_m))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


class SourcceyLidarOccupancyMapperConfig(ModuleConfig):
    map_size_m: float = 12.0
    resolution_m: float = 0.05
    frame_id: str = "world"
    forward_angle_deg: float = 270.0
    valid_angle_half_width_deg: float = 90.0
    invert_lateral_axis: bool = True
    lidar_mount_x_m: float = 0.0
    lidar_mount_y_m: float = 0.0
    min_range_m: float = 0.20
    max_distance_m: float = 8.0
    min_confidence: int = 0
    landmark_pose_blend: float = 0.0
    landmark_pose_max_age_s: float = 2.0
    scan_match_enabled: bool = True
    scan_match_max_points: int = 200
    scan_match_translation_window_m: float = 0.20
    scan_match_rotation_window_deg: float = 60.0
    scan_match_accept_score_m: float = 0.10
    scan_match_overlap_radius_m: float = 0.10
    scan_match_min_overlap_fraction: float = 0.14
    submap_max_points: int = 2200
    submap_local_radius_m: float = 5.0
    stationary_required_scans: int = 8
    stationary_keyframe_voxel_m: float = 0.03
    stationary_keyframe_max_points: int = 400
    cmd_vel_gate_enabled: bool = True
    cmd_vel_active_window_s: float = 1.0
    cmd_vel_motion_epsilon: float = 1e-3
    max_odom_linear_speed_for_commit_m_s: float = 0.04
    max_odom_angular_speed_for_commit_rad_s: float = 0.08
    free_ray_step_m: float = 0.05
    free_height_m: float = 0.00
    obstacle_height_m: float = 0.25
    registered_scan_height_m: float = 0.32
    obstacle_memory_height_m: float = 0.32
    global_map_max_free_points: int = 24000
    global_map_max_obstacle_points: int = 9000
    obstacle_memory_enabled: bool = True
    obstacle_memory_decay_s: float = 180.0
    obstacle_memory_max_points: int = 2500
    obstacle_memory_voxel_m: float = 0.05
    obstacle_memory_visible_clear_radius_m: float = 0.12
    export_dir: str = str(_DEFAULT_EXPORT_ROOT)
    export_png_name: str = "latest_map.png"
    export_metadata_name: str = "latest_map.json"
    export_interval_s: float = 0.75


class SourcceyLidarOccupancyMapper(Module):
    config: SourcceyLidarOccupancyMapperConfig

    scan: In[PlanarLidarScan]
    odom: In[PoseStamped]
    landmark_pose: In[PoseStamped]
    cmd_vel: In[Twist]

    global_costmap: Out[OccupancyGrid]
    localized_pose: Out[PoseStamped]
    global_map: Out[PointCloud2]
    registered_scan: Out[PointCloud2]
    obstacle_memory: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        map_size_m = max(2.0, float(self.config.map_size_m))
        resolution_m = max(0.01, float(self.config.resolution_m))
        self._width = int(math.ceil(map_size_m / resolution_m))
        self._height = int(math.ceil(map_size_m / resolution_m))
        self._resolution_m = resolution_m
        self._origin_x = -self._width * resolution_m / 2.0
        self._origin_y = -self._height * resolution_m / 2.0
        self._grid = np.full((self._height, self._width), int(CostValues.UNKNOWN), dtype=np.int8)

        self._latest_odom: PoseStamped | None = None
        self._latest_landmark_pose: PoseStamped | None = None
        self._latest_cmd_vel: Twist | None = None
        self._last_cmd_vel_wall_ts = 0.0
        self._last_odom_wall_ts = 0.0
        self._last_odom_linear_speed_m_s = 0.0
        self._last_odom_angular_speed_rad_s = 0.0

        self._last_matched_pose: PoseStamped | None = None
        self._last_integrated_pose: PoseStamped | None = None
        self._last_local_points = np.zeros((0, 2), dtype=np.float32)
        self._submap_points_world = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_memory_points_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_memory_timestamps = np.zeros((0,), dtype=np.float64)

        self._stationary_point_batches: list[np.ndarray] = []
        self._stationary_scan_streak = 0
        self._motion_epoch = 0
        self._last_committed_motion_epoch = -1
        self._motion_active = False

        self._last_export_ts = 0.0
        self._export_dir = Path(self.config.export_dir)
        self._export_png_path = self._export_dir / self.config.export_png_name
        self._export_metadata_path = self._export_dir / self.config.export_metadata_name

    @rpc
    def start(self) -> None:
        super().start()
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.landmark_pose.subscribe(self._on_landmark_pose)))
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))
        now = time.time()
        self.global_costmap.publish(self._make_grid_msg(now))
        self.global_map.publish(self._build_global_map_cloud(timestamp=now))
        self.registered_scan.publish(
            _points_to_cloud(
                np.zeros((0, 2), dtype=np.float32),
                z_height_m=float(self.config.registered_scan_height_m),
                frame_id=self.config.frame_id,
                timestamp=now,
            )
        )
        self.obstacle_memory.publish(
            _points_to_cloud(
                np.zeros((0, 2), dtype=np.float32),
                z_height_m=float(self.config.obstacle_memory_height_m),
                frame_id=self.config.frame_id,
                timestamp=now,
            )
        )
        self._write_map_export(ts=now, pose=None)
        trace_event("lidar_occupancy_mapper", "start")

    @rpc
    def stop(self) -> None:
        self._write_map_export(ts=time.time(), pose=self._last_matched_pose)
        trace_event("lidar_occupancy_mapper", "stop")
        super().stop()

    @rpc
    def reset_map(self) -> None:
        self._grid.fill(int(CostValues.UNKNOWN))
        self._last_matched_pose = None
        self._last_integrated_pose = None
        self._last_local_points = np.zeros((0, 2), dtype=np.float32)
        self._submap_points_world = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_memory_points_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_memory_timestamps = np.zeros((0,), dtype=np.float64)
        self._stationary_point_batches.clear()
        self._stationary_scan_streak = 0
        self._last_committed_motion_epoch = -1
        now = time.time()
        self.global_costmap.publish(self._make_grid_msg(now))
        self.global_map.publish(self._build_global_map_cloud(timestamp=now))
        self.registered_scan.publish(
            _points_to_cloud(
                np.zeros((0, 2), dtype=np.float32),
                z_height_m=float(self.config.registered_scan_height_m),
                frame_id=self.config.frame_id,
                timestamp=now,
            )
        )
        self.obstacle_memory.publish(
            _points_to_cloud(
                np.zeros((0, 2), dtype=np.float32),
                z_height_m=float(self.config.obstacle_memory_height_m),
                frame_id=self.config.frame_id,
                timestamp=now,
            )
        )
        self._write_map_export(ts=now, pose=None)
        trace_event("lidar_occupancy_mapper", "reset_map")

    def _on_odom(self, msg: PoseStamped) -> None:
        now = time.time()
        previous = self._latest_odom
        if previous is not None and self._last_odom_wall_ts > 0.0:
            dt = max(now - float(self._last_odom_wall_ts), 1e-3)
            dx = float(msg.x) - float(previous.x)
            dy = float(msg.y) - float(previous.y)
            dyaw = _wrap_angle_rad(float(msg.yaw) - float(previous.yaw))
            self._last_odom_linear_speed_m_s = math.hypot(dx, dy) / dt
            self._last_odom_angular_speed_rad_s = abs(float(dyaw)) / dt
        self._latest_odom = msg
        self._last_odom_wall_ts = now

    def _on_landmark_pose(self, msg: PoseStamped) -> None:
        self._latest_landmark_pose = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd_vel = msg
        self._last_cmd_vel_wall_ts = time.time()

    def _current_pose(self) -> PoseStamped | None:
        odom = self._latest_odom
        if odom is None:
            return self._last_matched_pose

        landmark = self._latest_landmark_pose
        if landmark is None:
            return odom
        if abs(float(odom.ts) - float(landmark.ts)) > float(self.config.landmark_pose_max_age_s):
            return odom

        weight = min(max(float(self.config.landmark_pose_blend), 0.0), 1.0)
        if weight <= 0.0:
            return odom

        yaw = _blend_angle_rad(float(odom.yaw), float(landmark.yaw), weight)
        return PoseStamped(
            ts=float(odom.ts),
            frame_id=self.config.frame_id,
            position=Vector3(
                odom.x * (1.0 - weight) + landmark.x * weight,
                odom.y * (1.0 - weight) + landmark.y * weight,
                odom.z * (1.0 - weight) + landmark.z * weight,
            ),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
        )

    def _clear_stationary_state(self) -> None:
        self._stationary_point_batches.clear()
        self._stationary_scan_streak = 0

    def _motion_is_active(self) -> bool:
        now = time.time()
        if bool(self.config.cmd_vel_gate_enabled) and self._latest_cmd_vel is not None:
            if (now - float(self._last_cmd_vel_wall_ts)) <= float(self.config.cmd_vel_active_window_s):
                linear = self._latest_cmd_vel.linear
                angular = self._latest_cmd_vel.angular
                magnitude = (
                    abs(float(linear.x))
                    + abs(float(linear.y))
                    + abs(float(linear.z))
                    + abs(float(angular.x))
                    + abs(float(angular.y))
                    + abs(float(angular.z))
                )
                if magnitude > float(self.config.cmd_vel_motion_epsilon):
                    return True
        return (
            float(self._last_odom_linear_speed_m_s) > float(self.config.max_odom_linear_speed_for_commit_m_s)
            or float(self._last_odom_angular_speed_rad_s)
            > float(self.config.max_odom_angular_speed_for_commit_rad_s)
        )

    def _set_motion_state(self, *, motion_active: bool, scan_ts: float) -> None:
        if motion_active == self._motion_active:
            return
        self._motion_active = bool(motion_active)
        if motion_active:
            self._motion_epoch += 1
            self._clear_stationary_state()
            self.registered_scan.publish(
                _points_to_cloud(
                    np.zeros((0, 2), dtype=np.float32),
                    z_height_m=float(self.config.registered_scan_height_m),
                    frame_id=self.config.frame_id,
                    timestamp=float(scan_ts),
                )
            )
            self.obstacle_memory.publish(
                _points_to_cloud(
                    self._hidden_memory_points(np.zeros((0, 2), dtype=np.float32), timestamp=float(scan_ts)),
                    z_height_m=float(self.config.obstacle_memory_height_m),
                    frame_id=self.config.frame_id,
                    timestamp=float(scan_ts),
                )
            )
            trace_event(
                "lidar_occupancy_mapper",
                "motion_active",
                motion_epoch=int(self._motion_epoch),
                odom_linear_speed_m_s=round(float(self._last_odom_linear_speed_m_s), 4),
                odom_angular_speed_rad_s=round(float(self._last_odom_angular_speed_rad_s), 4),
            )
        else:
            trace_event(
                "lidar_occupancy_mapper",
                "motion_stopped",
                motion_epoch=int(self._motion_epoch),
            )

    def _reference_world_points(self, *, pose: PoseStamped | None) -> np.ndarray:
        if self._submap_points_world.size != 0:
            return self._submap_points_world
        if pose is None or self._last_local_points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return _transform_local_points_with_mount(
            self._last_local_points,
            x=float(pose.x),
            y=float(pose.y),
            yaw=float(pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )

    def _refine_pose_with_scan_match(
        self,
        *,
        ts: float,
        seed_pose: PoseStamped,
        current_local_points: np.ndarray,
        reference_world: np.ndarray,
    ) -> tuple[PoseStamped | None, float, float]:
        if reference_world.size == 0:
            return None, float("inf"), 0.0

        source_local = _downsample_points(
            current_local_points,
            max(int(self.config.scan_match_max_points), 40),
        )
        best_x, best_y, best_yaw = _pose_xy_yaw(seed_pose)
        best_candidate = _transform_local_points_with_mount(
            source_local,
            x=best_x,
            y=best_y,
            yaw=best_yaw,
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        best_score, best_overlap = _scan_match_metrics(
            reference_world,
            best_candidate,
            overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
        )

        translation_window = max(float(self.config.scan_match_translation_window_m), 0.02)
        rotation_window = math.radians(max(float(self.config.scan_match_rotation_window_deg), 2.0))
        search_levels = (
            (translation_window, rotation_window, 4),
            (translation_window * 0.5, rotation_window * 0.4, 3),
            (translation_window * 0.2, rotation_window * 0.15, 2),
        )

        for trans_window, yaw_window, n_steps in search_levels:
            offsets = range(-n_steps, n_steps + 1)
            trans_step = float(trans_window) / max(int(n_steps), 1)
            yaw_step = float(yaw_window) / max(int(n_steps), 1)
            for dx_idx in offsets:
                for dy_idx in offsets:
                    for dyaw_idx in offsets:
                        cand_x = best_x + dx_idx * trans_step
                        cand_y = best_y + dy_idx * trans_step
                        cand_yaw = _wrap_angle_rad(best_yaw + dyaw_idx * yaw_step)
                        candidate_world = _transform_local_points_with_mount(
                            source_local,
                            x=cand_x,
                            y=cand_y,
                            yaw=cand_yaw,
                            mount_x_m=float(self.config.lidar_mount_x_m),
                            mount_y_m=float(self.config.lidar_mount_y_m),
                        )
                        score, overlap = _scan_match_metrics(
                            reference_world,
                            candidate_world,
                            overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
                        )
                        if (
                            score < best_score - 1e-6
                            or (abs(score - best_score) <= 1e-6 and overlap > best_overlap)
                        ):
                            best_score = score
                            best_overlap = overlap
                            best_x = cand_x
                            best_y = cand_y
                            best_yaw = cand_yaw

        if (
            best_score > float(self.config.scan_match_accept_score_m)
            or best_overlap < float(self.config.scan_match_min_overlap_fraction)
        ):
            return None, best_score, best_overlap

        return (
            _pose_from_xy_yaw(
                x=best_x,
                y=best_y,
                yaw=best_yaw,
                ts=float(ts),
                frame_id=self.config.frame_id,
            ),
            best_score,
            best_overlap,
        )

    def _refine_pose_from_previous_snapshot(
        self,
        *,
        ts: float,
        seed_pose: PoseStamped,
        current_local_points: np.ndarray,
    ) -> tuple[PoseStamped | None, float, float]:
        last_pose = self._last_matched_pose
        if last_pose is None or self._last_local_points.size == 0 or current_local_points.size == 0:
            return None, float("inf"), 0.0

        reference_local = _downsample_points(
            self._last_local_points,
            max(int(self.config.scan_match_max_points), 60),
        )
        source_local = _downsample_points(
            current_local_points,
            max(int(self.config.scan_match_max_points), 60),
        )

        last_sensor_x, last_sensor_y = _sensor_origin_xy(
            x=float(last_pose.x),
            y=float(last_pose.y),
            yaw=float(last_pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        seed_sensor_x, seed_sensor_y = _sensor_origin_xy(
            x=float(seed_pose.x),
            y=float(seed_pose.y),
            yaw=float(seed_pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        best_dx, best_dy, best_dyaw = _relative_pose_2d(
            source_x=float(last_sensor_x),
            source_y=float(last_sensor_y),
            source_yaw=float(last_pose.yaw),
            target_x=float(seed_sensor_x),
            target_y=float(seed_sensor_y),
            target_yaw=float(seed_pose.yaw),
        )
        best_candidate = _transform_points_2d(
            source_local,
            x=best_dx,
            y=best_dy,
            yaw=best_dyaw,
        )
        best_score, best_overlap = _scan_match_metrics(
            reference_local,
            best_candidate,
            overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
        )

        translation_window = max(float(self.config.scan_match_translation_window_m), 0.05)
        rotation_window = math.radians(max(float(self.config.scan_match_rotation_window_deg), 10.0))
        search_levels = (
            (translation_window * 1.25, rotation_window * 1.25, 5),
            (translation_window * 0.45, rotation_window * 0.40, 4),
            (translation_window * 0.15, rotation_window * 0.15, 3),
        )

        for trans_window, yaw_window, n_steps in search_levels:
            offsets = range(-n_steps, n_steps + 1)
            trans_step = float(trans_window) / max(int(n_steps), 1)
            yaw_step = float(yaw_window) / max(int(n_steps), 1)
            for dx_idx in offsets:
                for dy_idx in offsets:
                    for dyaw_idx in offsets:
                        cand_dx = best_dx + dx_idx * trans_step
                        cand_dy = best_dy + dy_idx * trans_step
                        cand_dyaw = _wrap_angle_rad(best_dyaw + dyaw_idx * yaw_step)
                        candidate_local = _transform_points_2d(
                            source_local,
                            x=cand_dx,
                            y=cand_dy,
                            yaw=cand_dyaw,
                        )
                        score, overlap = _scan_match_metrics(
                            reference_local,
                            candidate_local,
                            overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
                        )
                        if (
                            score < best_score - 1e-6
                            or (abs(score - best_score) <= 1e-6 and overlap > best_overlap)
                        ):
                            best_score = score
                            best_overlap = overlap
                            best_dx = cand_dx
                            best_dy = cand_dy
                            best_dyaw = cand_dyaw

        if (
            best_score > float(self.config.scan_match_accept_score_m)
            or best_overlap < float(self.config.scan_match_min_overlap_fraction)
        ):
            trace_event(
                "lidar_occupancy_mapper",
                "relative_snapshot_rejected",
                score=round(float(best_score), 5),
                overlap_fraction=round(float(best_overlap), 4),
                guess_dx=round(float(best_dx), 4),
                guess_dy=round(float(best_dy), 4),
                guess_dyaw_deg=round(math.degrees(float(best_dyaw)), 2),
            )
            return None, best_score, best_overlap

        sensor_x, sensor_y, sensor_yaw = _compose_pose_2d(
            origin_x=float(last_sensor_x),
            origin_y=float(last_sensor_y),
            origin_yaw=float(last_pose.yaw),
            rel_x=float(best_dx),
            rel_y=float(best_dy),
            rel_yaw=float(best_dyaw),
        )
        base_x, base_y = _base_pose_from_sensor_xy_yaw(
            sensor_x=float(sensor_x),
            sensor_y=float(sensor_y),
            yaw=float(sensor_yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        trace_event(
            "lidar_occupancy_mapper",
            "relative_snapshot_pose_resolved",
            dx=round(float(best_dx), 4),
            dy=round(float(best_dy), 4),
            dyaw_deg=round(math.degrees(float(best_dyaw)), 2),
            pose_x=round(float(base_x), 4),
            pose_y=round(float(base_y), 4),
            pose_yaw_deg=round(math.degrees(float(sensor_yaw)), 2),
            score=round(float(best_score), 5),
            overlap_fraction=round(float(best_overlap), 4),
        )
        return (
            _pose_from_xy_yaw(
                x=float(base_x),
                y=float(base_y),
                yaw=float(sensor_yaw),
                ts=float(ts),
                frame_id=self.config.frame_id,
            ),
            best_score,
            best_overlap,
        )

    def _resolve_snapshot_pose(
        self,
        *,
        ts: float,
        local_points: np.ndarray,
    ) -> tuple[PoseStamped | None, bool, float | None, float]:
        seed_pose = self._current_pose()
        if seed_pose is None:
            return None, False, None, 0.0
        if (
            not bool(self.config.scan_match_enabled)
            or self._last_matched_pose is None
            or self._last_local_points.size == 0
            or local_points.size == 0
        ):
            return seed_pose, True, None, 1.0

        reference_world = self._reference_world_points(pose=self._last_matched_pose)
        if reference_world.size == 0:
            return seed_pose, True, None, 1.0

        relative_pose, relative_score, relative_overlap = self._refine_pose_from_previous_snapshot(
            ts=float(ts),
            seed_pose=seed_pose,
            current_local_points=local_points,
        )
        if relative_pose is not None:
            refined_pose, score, overlap = self._refine_pose_with_scan_match(
                ts=float(ts),
                seed_pose=relative_pose,
                current_local_points=local_points,
                reference_world=reference_world,
            )
            if refined_pose is not None:
                shift_m = math.hypot(float(refined_pose.x) - float(relative_pose.x), float(refined_pose.y) - float(relative_pose.y))
                shift_yaw_deg = abs(
                    math.degrees(_wrap_angle_rad(float(refined_pose.yaw) - float(relative_pose.yaw)))
                )
                if shift_m <= 0.18 and shift_yaw_deg <= 18.0:
                    trace_event(
                        "lidar_occupancy_mapper",
                        "snapshot_pose_resolved",
                        method="relative_then_global_refine",
                        score=None if score is None else round(float(score), 5),
                        overlap_fraction=round(float(overlap), 4),
                        shift_m=round(float(shift_m), 4),
                        shift_yaw_deg=round(float(shift_yaw_deg), 2),
                    )
                    return refined_pose, True, score, overlap
                trace_event(
                    "lidar_occupancy_mapper",
                    "snapshot_global_refine_ignored",
                    reason="relative_guardrail",
                    relative_score=round(float(relative_score), 5),
                    relative_overlap_fraction=round(float(relative_overlap), 4),
                    score=None if score is None else round(float(score), 5),
                    overlap_fraction=round(float(overlap), 4),
                    shift_m=round(float(shift_m), 4),
                    shift_yaw_deg=round(float(shift_yaw_deg), 2),
                )
            trace_event(
                "lidar_occupancy_mapper",
                "snapshot_pose_resolved",
                method="relative_snapshot_match",
                score=round(float(relative_score), 5),
                overlap_fraction=round(float(relative_overlap), 4),
            )
            return relative_pose, True, relative_score, relative_overlap

        refined_pose, score, overlap = self._refine_pose_with_scan_match(
            ts=float(ts),
            seed_pose=seed_pose,
            current_local_points=local_points,
            reference_world=reference_world,
        )
        if refined_pose is None:
            return None, False, score, overlap
        trace_event(
            "lidar_occupancy_mapper",
            "snapshot_pose_resolved",
            method="global_only_fallback",
            score=None if score is None else round(float(score), 5),
            overlap_fraction=round(float(overlap), 4),
        )
        return refined_pose, True, score, overlap

    def _merge_stationary_snapshot(self) -> np.ndarray:
        if not self._stationary_point_batches:
            return np.zeros((0, 2), dtype=np.float32)
        merged = np.vstack(self._stationary_point_batches).astype(np.float32, copy=False)
        merged = _voxelize_points(merged, float(self.config.stationary_keyframe_voxel_m))
        merged = _downsample_points(merged, max(int(self.config.stationary_keyframe_max_points), 80))
        return merged.astype(np.float32, copy=False)

    def _update_submap(self, world_points: np.ndarray, *, pose: PoseStamped) -> None:
        if world_points.size == 0:
            return
        if self._submap_points_world.size == 0:
            merged = world_points.astype(np.float32, copy=True)
        else:
            merged = np.vstack((self._submap_points_world, world_points)).astype(np.float32, copy=False)
        merged = _voxelize_points(merged, float(self.config.stationary_keyframe_voxel_m))
        sensor_x, sensor_y = _sensor_origin_xy(
            x=float(pose.x),
            y=float(pose.y),
            yaw=float(pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        local_radius = max(float(self.config.submap_local_radius_m), 0.5)
        deltas = merged - np.asarray((sensor_x, sensor_y), dtype=np.float32)
        keep_mask = np.sum(deltas * deltas, axis=1) <= (local_radius * local_radius)
        cropped = merged[keep_mask]
        self._submap_points_world = _downsample_points(
            cropped,
            max(int(self.config.submap_max_points), int(self.config.stationary_keyframe_max_points)),
        ).astype(np.float32, copy=False)

    def _inside_grid(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def _world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        grid_x = int((x_m - self._origin_x) / self._resolution_m)
        grid_y = int((y_m - self._origin_y) / self._resolution_m)
        if not self._inside_grid(grid_x, grid_y):
            return None
        return grid_x, grid_y

    def _make_grid_msg(self, ts: float) -> OccupancyGrid:
        return OccupancyGrid(
            grid=self._grid.copy(),
            resolution=self._resolution_m,
            origin=Pose(
                position=Vector3(self._origin_x, self._origin_y, 0.0),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ),
            frame_id=self.config.frame_id,
            ts=float(ts),
        )

    def _grid_points(self, *, occupied: bool) -> np.ndarray:
        if occupied:
            cells = np.argwhere(self._grid >= int(CostValues.OCCUPIED))
        else:
            cells = np.argwhere(self._grid == int(CostValues.FREE))
        if len(cells) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        rows = cells[:, 0].astype(np.float32)
        cols = cells[:, 1].astype(np.float32)
        world_x = self._origin_x + (cols + 0.5) * self._resolution_m
        world_y = self._origin_y + (rows + 0.5) * self._resolution_m
        return np.column_stack((world_x, world_y)).astype(np.float32, copy=False)

    def _build_global_map_cloud(self, *, timestamp: float) -> PointCloud2:
        free_points = _downsample_points(
            self._grid_points(occupied=False),
            max(int(self.config.global_map_max_free_points), 1),
        )
        occupied_points = _downsample_points(
            self._grid_points(occupied=True),
            max(int(self.config.global_map_max_obstacle_points), 1),
        )
        if free_points.size == 0 and occupied_points.size == 0:
            return _points_to_cloud(
                np.zeros((0, 2), dtype=np.float32),
                z_height_m=float(self.config.free_height_m),
                frame_id=self.config.frame_id,
                timestamp=float(timestamp),
            )

        stacked_points: list[np.ndarray] = []
        stacked_intensities: list[np.ndarray] = []
        if free_points.size != 0:
            stacked_points.append(free_points)
            stacked_intensities.append(np.zeros((len(free_points),), dtype=np.float32))
        if occupied_points.size != 0:
            stacked_points.append(occupied_points)
            stacked_intensities.append(np.ones((len(occupied_points),), dtype=np.float32))
        all_points = np.vstack(stacked_points).astype(np.float32, copy=False)
        all_intensities = np.concatenate(stacked_intensities).astype(np.float32, copy=False)
        xyz = np.column_stack(
            (
                all_points[:, 0],
                all_points[:, 1],
                np.where(all_intensities > 0.0, float(self.config.obstacle_height_m), float(self.config.free_height_m)),
            )
        ).astype(np.float32, copy=False)
        return PointCloud2.from_numpy(
            xyz,
            frame_id=self.config.frame_id,
            timestamp=float(timestamp),
            intensities=all_intensities,
        )

    def _update_obstacle_memory(self, world_points_xy: np.ndarray, *, timestamp: float) -> None:
        if not bool(self.config.obstacle_memory_enabled):
            return
        now = float(timestamp)
        if self._obstacle_memory_timestamps.size != 0:
            keep_mask = (now - self._obstacle_memory_timestamps) <= float(self.config.obstacle_memory_decay_s)
            self._obstacle_memory_points_xy = self._obstacle_memory_points_xy[keep_mask]
            self._obstacle_memory_timestamps = self._obstacle_memory_timestamps[keep_mask]
        if world_points_xy.size == 0:
            return
        new_points = _voxelize_points(world_points_xy, float(self.config.obstacle_memory_voxel_m))
        if new_points.size == 0:
            return
        if self._obstacle_memory_points_xy.size == 0:
            merged_points = new_points
            merged_timestamps = np.full((len(new_points),), now, dtype=np.float64)
        else:
            merged_points = np.vstack((self._obstacle_memory_points_xy, new_points)).astype(
                np.float32,
                copy=False,
            )
            merged_timestamps = np.concatenate(
                (
                    self._obstacle_memory_timestamps,
                    np.full((len(new_points),), now, dtype=np.float64),
                )
            )
        voxel = max(float(self.config.obstacle_memory_voxel_m), 1e-3)
        merged_cells = np.round(merged_points / voxel).astype(np.int64, copy=False)
        order_by_ts = np.argsort(merged_timestamps, kind="stable")
        cells_sorted = merged_cells[order_by_ts]
        _, last_in_reversed = np.unique(cells_sorted[::-1], axis=0, return_index=True)
        keep_indices = order_by_ts[len(cells_sorted) - 1 - last_in_reversed]
        if keep_indices.size != 0:
            order = np.argsort(merged_timestamps[keep_indices])[::-1]
            keep_indices = keep_indices[order]
            keep_indices = keep_indices[: max(int(self.config.obstacle_memory_max_points), 1)]
            self._obstacle_memory_points_xy = merged_points[keep_indices].astype(np.float32, copy=False)
            self._obstacle_memory_timestamps = merged_timestamps[keep_indices]

    def _hidden_memory_points(self, visible_points_xy: np.ndarray, *, timestamp: float) -> np.ndarray:
        if not bool(self.config.obstacle_memory_enabled):
            return np.zeros((0, 2), dtype=np.float32)
        now = float(timestamp)
        if self._obstacle_memory_timestamps.size != 0:
            keep_mask = (now - self._obstacle_memory_timestamps) <= float(self.config.obstacle_memory_decay_s)
            self._obstacle_memory_points_xy = self._obstacle_memory_points_xy[keep_mask]
            self._obstacle_memory_timestamps = self._obstacle_memory_timestamps[keep_mask]
        if self._obstacle_memory_points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if visible_points_xy.size == 0:
            return self._obstacle_memory_points_xy.astype(np.float32, copy=False)
        visible_points_xy = _voxelize_points(visible_points_xy, float(self.config.obstacle_memory_voxel_m))
        if visible_points_xy.size == 0:
            return self._obstacle_memory_points_xy.astype(np.float32, copy=False)
        delta = self._obstacle_memory_points_xy[:, None, :] - visible_points_xy[None, :, :]
        min_d2 = np.min(np.sum(delta * delta, axis=2), axis=1)
        clear_radius = max(float(self.config.obstacle_memory_visible_clear_radius_m), 1e-3)
        keep_mask = min_d2 > (clear_radius * clear_radius)
        return self._obstacle_memory_points_xy[keep_mask].astype(np.float32, copy=False)

    def _publish_visual_layers(self, *, timestamp: float, visible_world_points_xy: np.ndarray) -> None:
        self.global_map.publish(self._build_global_map_cloud(timestamp=float(timestamp)))
        self.registered_scan.publish(
            _points_to_cloud(
                visible_world_points_xy,
                z_height_m=float(self.config.registered_scan_height_m),
                frame_id=self.config.frame_id,
                timestamp=float(timestamp),
            )
        )
        hidden_memory = self._hidden_memory_points(visible_world_points_xy, timestamp=float(timestamp))
        self.obstacle_memory.publish(
            _points_to_cloud(
                hidden_memory,
                z_height_m=float(self.config.obstacle_memory_height_m),
                frame_id=self.config.frame_id,
                timestamp=float(timestamp),
            )
        )

    def _maybe_export_map(self, *, ts: float, pose: PoseStamped | None) -> None:
        interval_s = max(0.1, float(self.config.export_interval_s))
        if ts - self._last_export_ts < interval_s:
            return
        self._write_map_export(ts=ts, pose=pose)
        self._last_export_ts = ts

    def _write_map_export(self, *, ts: float, pose: PoseStamped | None) -> None:
        image = np.full((self._height, self._width), 127, dtype=np.uint8)
        image[self._grid == int(CostValues.FREE)] = 255
        image[self._grid >= int(CostValues.OCCUPIED)] = 0
        image = np.flipud(image)
        cv2.imwrite(str(self._export_png_path), image)

        occupied_cells = int(np.sum(self._grid >= int(CostValues.OCCUPIED)))
        free_cells = int(np.sum(self._grid == int(CostValues.FREE)))
        unknown_cells = int(np.sum(self._grid == int(CostValues.UNKNOWN)))
        metadata = {
            "schema": "sourccey.native_costmap.v2",
            "ts": float(ts),
            "frame_id": self.config.frame_id,
            "resolution_m": self._resolution_m,
            "width": self._width,
            "height": self._height,
            "origin": {
                "x": self._origin_x,
                "y": self._origin_y,
            },
            "counts": {
                "occupied": occupied_cells,
                "free": free_cells,
                "unknown": unknown_cells,
            },
            "pose": None
            if pose is None
            else {
                "x": float(pose.x),
                "y": float(pose.y),
                "z": float(pose.z),
                "yaw": float(pose.yaw),
            },
            "png_path": str(self._export_png_path),
        }
        self._export_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(
            "Sourccey offboard occupancy map exported png=%s metadata=%s occupied=%s free=%s unknown=%s",
            self._export_png_path,
            self._export_metadata_path,
            occupied_cells,
            free_cells,
            unknown_cells,
        )

    def _integrate_snapshot(
        self,
        *,
        scan_ts: float,
        pose: PoseStamped,
        local_points_xy: np.ndarray,
        score: float | None,
        overlap: float,
    ) -> None:
        sensor_x, sensor_y = _sensor_origin_xy(
            x=float(pose.x),
            y=float(pose.y),
            yaw=float(pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        sensor_cell = self._world_to_grid(sensor_x, sensor_y)
        base_cell = self._world_to_grid(float(pose.x), float(pose.y))
        if sensor_cell is None or base_cell is None:
            trace_event(
                "lidar_occupancy_mapper",
                "snapshot_rejected_outside_grid",
                pose_x=round(float(pose.x), 4),
                pose_y=round(float(pose.y), 4),
                sensor_x=round(float(sensor_x), 4),
                sensor_y=round(float(sensor_y), 4),
            )
            return

        self._grid[sensor_cell[1], sensor_cell[0]] = int(CostValues.FREE)
        self._grid[base_cell[1], base_cell[0]] = int(CostValues.FREE)

        world_points = _transform_local_points_with_mount(
            local_points_xy,
            x=float(pose.x),
            y=float(pose.y),
            yaw=float(pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        for world_x, world_y in world_points:
            hit_cell = self._world_to_grid(float(world_x), float(world_y))
            if hit_cell is None:
                continue
            ray = _ray_cells(sensor_cell[0], sensor_cell[1], hit_cell[0], hit_cell[1])
            for free_x, free_y in ray[:-1]:
                if self._inside_grid(free_x, free_y) and self._grid[free_y, free_x] != int(
                    CostValues.OCCUPIED
                ):
                    self._grid[free_y, free_x] = int(CostValues.FREE)
            if self._inside_grid(hit_cell[0], hit_cell[1]):
                self._grid[hit_cell[1], hit_cell[0]] = int(CostValues.OCCUPIED)

        self._last_local_points = local_points_xy.astype(np.float32, copy=True)
        self._last_matched_pose = pose
        self._last_integrated_pose = pose
        self._update_submap(world_points, pose=pose)
        self._update_obstacle_memory(world_points, timestamp=float(scan_ts))
        self.localized_pose.publish(pose)
        self.global_costmap.publish(self._make_grid_msg(float(scan_ts)))
        self._publish_visual_layers(timestamp=float(scan_ts), visible_world_points_xy=world_points)
        self._maybe_export_map(ts=float(scan_ts), pose=pose)
        trace_event(
            "lidar_occupancy_mapper",
            "snapshot_committed",
            motion_epoch=int(self._motion_epoch),
            pose_x=round(float(pose.x), 4),
            pose_y=round(float(pose.y), 4),
            pose_yaw_deg=round(math.degrees(float(pose.yaw)), 2),
            local_points=int(len(local_points_xy)),
            visible_points=int(len(world_points)),
            score=None if score is None else round(float(score), 5),
            overlap_fraction=round(float(overlap), 4),
            grid_free=int(np.sum(self._grid == int(CostValues.FREE))),
            grid_occupied=int(np.sum(self._grid >= int(CostValues.OCCUPIED))),
            memory_points=int(len(self._obstacle_memory_points_xy)),
        )
        logger.info(
            "SourcceyLidarOccupancyMapper committed stationary snapshot pose=(%.3f, %.3f, %.1fdeg) points=%s score=%s overlap=%.3f",
            float(pose.x),
            float(pose.y),
            math.degrees(float(pose.yaw)),
            int(len(local_points_xy)),
            "bootstrap" if score is None else round(float(score), 4),
            float(overlap),
        )

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        local_points_xy = _scan_to_local_points(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            invert_lateral_axis=bool(self.config.invert_lateral_axis),
            max_distance_m=float(self.config.max_distance_m),
            min_confidence=int(self.config.min_confidence),
            min_range_m=float(self.config.min_range_m),
        )
        motion_active = self._motion_is_active()
        self._set_motion_state(motion_active=motion_active, scan_ts=float(scan.ts))
        if motion_active:
            return
        if local_points_xy.size == 0:
            trace_event(
                "lidar_occupancy_mapper",
                "snapshot_skipped_empty_scan",
                motion_epoch=int(self._motion_epoch),
            )
            return

        self._stationary_point_batches.append(local_points_xy)
        max_batches = max(int(self.config.stationary_required_scans) * 2, 4)
        if len(self._stationary_point_batches) > max_batches:
            self._stationary_point_batches = self._stationary_point_batches[-max_batches:]
        self._stationary_scan_streak += 1

        if self._stationary_scan_streak < max(int(self.config.stationary_required_scans), 1):
            return
        if self._last_committed_motion_epoch == self._motion_epoch:
            return

        snapshot_points = self._merge_stationary_snapshot()
        if snapshot_points.size == 0:
            trace_event(
                "lidar_occupancy_mapper",
                "snapshot_skipped_after_merge_empty",
                motion_epoch=int(self._motion_epoch),
                streak=int(self._stationary_scan_streak),
            )
            return

        pose, accepted, score, overlap = self._resolve_snapshot_pose(
            ts=float(scan.ts),
            local_points=snapshot_points,
        )
        if pose is None or not accepted:
            trace_event(
                "lidar_occupancy_mapper",
                "snapshot_rejected_scan_match",
                motion_epoch=int(self._motion_epoch),
                streak=int(self._stationary_scan_streak),
                local_points=int(len(snapshot_points)),
                score=None if score is None else round(float(score), 5),
                overlap_fraction=round(float(overlap), 4),
            )
            logger.info(
                "SourcceyLidarOccupancyMapper rejected stationary snapshot local_points=%s score=%s overlap=%.3f",
                int(len(snapshot_points)),
                "n/a" if score is None else round(float(score), 4),
                float(overlap),
            )
            return

        self._integrate_snapshot(
            scan_ts=float(scan.ts),
            pose=pose,
            local_points_xy=snapshot_points,
            score=score,
            overlap=overlap,
        )
        self._last_committed_motion_epoch = self._motion_epoch
        self._clear_stationary_state()
