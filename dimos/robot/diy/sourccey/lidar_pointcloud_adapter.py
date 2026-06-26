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
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

from .lidar_geometry import normalize_angle_deg
from .lidar_types import PlanarLidarScan

logger = setup_logger()


class SourcceyLidarPointCloudAdapterConfig(ModuleConfig):
    forward_angle_deg: float = 180.0
    valid_angle_half_width_deg: float = 90.0
    invert_lateral_axis: bool = False
    lidar_mount_x_m: float = 0.0
    lidar_mount_y_m: float = 0.0
    max_distance_m: float = 8.0
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
    scan_match_rotation_window_deg: float = 10.0
    scan_match_accept_score_m: float = 0.12
    submap_max_points: int = 1600
    registered_scan_height_m: float = 0.32
    reset_context_translation_m: float = 0.35
    reset_context_rotation_deg: float = 30.0
    min_points_for_scan_match: int = 40
    max_odom_angular_speed_for_mapping_rad_s: float = 0.30
    max_odom_linear_speed_for_mapping_m_s: float = 0.20
    mapping_holdoff_after_turn_s: float = 0.35
    obstacle_memory_enabled: bool = True
    obstacle_memory_decay_s: float = 3600.0
    obstacle_memory_max_points: int = 8000
    obstacle_memory_voxel_m: float = 0.04
    obstacle_memory_height_m: float = 0.32


def _scan_to_local_points(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_confidence: int,
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


def _scan_match_score(reference_world: np.ndarray, candidate_world: np.ndarray) -> float:
    if reference_world.size == 0 or candidate_world.size == 0:
        return float("inf")
    deltas = candidate_world[:, None, :] - reference_world[None, :, :]
    min_d2 = np.min(np.sum(deltas * deltas, axis=2), axis=1)
    return float(np.mean(np.clip(min_d2, 0.0, 0.25)))


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
) -> PointCloud2:
    local_points = _scan_to_local_points(
        scan,
        forward_angle_deg=forward_angle_deg,
        valid_angle_half_width_deg=valid_angle_half_width_deg,
        invert_lateral_axis=invert_lateral_axis,
        max_distance_m=max_distance_m,
        min_confidence=min_confidence,
    )
    if not local_points:
        return PointCloud2.from_numpy(
            np.zeros((0, 3), dtype=np.float32),
            frame_id=frame_id,
            timestamp=float(scan.ts),
        )

    free_points_local: list[np.ndarray] = []
    obstacle_points_local = np.zeros((len(local_points), 2), dtype=np.float32)
    obstacle_intensities = np.zeros((len(local_points),), dtype=np.float32)

    for index, (forward_m, lateral_m, distance_m, confidence) in enumerate(local_points):
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
        timestamp=float(scan.ts),
        intensities=intensities,
    )


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
    lidar: Out[PointCloud2]
    registered_scan: Out[PointCloud2]
    localized_pose: Out[PoseStamped]
    obstacle_memory: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._prev_odom: PoseStamped | None = None
        self._scan_count = 0
        self._last_local_points: np.ndarray | None = None
        self._last_matched_pose: PoseStamped | None = None
        self._submap_points_world = np.zeros((0, 2), dtype=np.float32)
        self._last_turn_motion_wall_ts: float | None = None
        self._obstacle_memory_points_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_memory_timestamps = np.zeros((0,), dtype=np.float64)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))

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
        self._prev_odom = self._latest_odom
        self._latest_odom = msg

    def _hold_mapping_for_turn(self) -> bool:
        if self._last_turn_motion_wall_ts is None:
            return False
        return (time.time() - float(self._last_turn_motion_wall_ts)) < float(
            self.config.mapping_holdoff_after_turn_s
        )

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

    def _clear_match_context(self) -> None:
        self._last_local_points = None
        self._last_matched_pose = None
        self._submap_points_world = np.zeros((0, 2), dtype=np.float32)

    def _should_reset_match_context(self, seed_pose: PoseStamped) -> bool:
        if self._last_matched_pose is None:
            return False
        dx = float(seed_pose.x) - float(self._last_matched_pose.x)
        dy = float(seed_pose.y) - float(self._last_matched_pose.y)
        distance_m = math.hypot(dx, dy)
        yaw_delta_deg = abs(
            math.degrees(_wrap_angle_rad(float(seed_pose.yaw) - float(self._last_matched_pose.yaw)))
        )
        return (
            distance_m > float(self.config.reset_context_translation_m)
            or yaw_delta_deg > float(self.config.reset_context_rotation_deg)
        )

    def _refine_pose_with_scan_match(
        self,
        *,
        ts: float,
        seed_pose: PoseStamped,
        current_local_points: np.ndarray,
        reference_world: np.ndarray,
    ) -> tuple[PoseStamped | None, float]:
        if reference_world.size == 0 or current_local_points.size == 0:
            return None, float("inf")

        source_local = _downsample_points(
            current_local_points,
            max(int(self.config.scan_match_max_points), 24),
        )
        best_x = float(seed_pose.x)
        best_y = float(seed_pose.y)
        best_yaw = float(seed_pose.yaw)
        seed_sensor_x, seed_sensor_y = _sensor_origin_xy(
            seed_pose,
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        best_score = _scan_match_score(
            reference_world,
            _transform_local_points(
                source_local,
                x=seed_sensor_x,
                y=seed_sensor_y,
                yaw=best_yaw,
            ),
        )

        translation_window = max(float(self.config.scan_match_translation_window_m), 0.02)
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
                        score = _scan_match_score(
                            reference_world,
                            _transform_local_points(source_local, x=sensor_x, y=sensor_y, yaw=cand_yaw),
                        )
                        if score < best_score:
                            best_score = score
                            best_x = cand_x
                            best_y = cand_y
                            best_yaw = cand_yaw

        if best_score > float(self.config.scan_match_accept_score_m):
            return None, best_score

        return (
            PoseStamped(
                ts=float(ts),
                frame_id=self.config.frame_id,
                position=Vector3(best_x, best_y, float(seed_pose.z)),
                orientation=Quaternion.from_euler(Vector3(0.0, 0.0, best_yaw)),
            ),
            best_score,
        )

    def _resolve_scan_pose(
        self,
        scan: PlanarLidarScan,
        local_points_xy: np.ndarray,
    ) -> tuple[PoseStamped | None, bool, float | None]:
        odom = self._latest_odom
        if odom is None:
            logger.debug("Skipping Sourccey LiDAR scan because no odom has been received yet")
            return None, False, None

        now = time.time()
        if abs(now - float(odom.ts)) > float(self.config.odom_stale_after_s):
            logger.debug("Skipping Sourccey LiDAR scan because odom is stale")
            return None, False, None

        if self._should_reset_match_context(odom):
            logger.info(
                "Resetting LiDAR scan-match context after large pose delta",
                odom_x=round(float(odom.x), 3),
                odom_y=round(float(odom.y), 3),
                odom_yaw_deg=round(math.degrees(float(odom.yaw)), 1),
            )
            self._clear_match_context()

        if self._hold_mapping_for_turn():
            return None, False, None

        if (
            not bool(self.config.scan_match_enabled)
            or self._last_local_points is None
            or self._last_matched_pose is None
            or local_points_xy.size == 0
            or len(local_points_xy) < int(self.config.min_points_for_scan_match)
        ):
            return odom, True, None

        reference_world = self._reference_world_points()
        if reference_world.size == 0:
            return odom, True, None

        refined_pose, score = self._refine_pose_with_scan_match(
            ts=float(scan.ts),
            seed_pose=odom,
            current_local_points=local_points_xy,
            reference_world=reference_world,
        )
        if refined_pose is None:
            self._clear_match_context()
            return odom, False, score
        return refined_pose, True, score

    def _update_submap(self, world_points_xy: np.ndarray) -> None:
        if world_points_xy.size == 0:
            return
        candidate = _downsample_points(world_points_xy, max(int(self.config.scan_match_max_points), 24))
        if self._submap_points_world.size == 0:
            self._submap_points_world = candidate.astype(np.float32, copy=True)
            return
        merged = np.vstack((self._submap_points_world, candidate)).astype(np.float32, copy=False)
        self._submap_points_world = _downsample_points(
            merged,
            max(int(self.config.submap_max_points), int(self.config.scan_match_max_points)),
        )

    def _publish_obstacle_memory(self, timestamp: float, *, visible_points_xy: np.ndarray | None = None) -> None:
        if not bool(self.config.obstacle_memory_enabled):
            return
        memory_points_xy = self._obstacle_memory_points_xy
        if visible_points_xy is not None and visible_points_xy.size != 0 and memory_points_xy.size != 0:
            voxel = max(float(self.config.obstacle_memory_voxel_m), 1e-3)
            visible_cells = {
                tuple(cell)
                for cell in np.round(visible_points_xy / voxel).astype(np.int32, copy=False).tolist()
            }
            if visible_cells:
                memory_cells = np.round(memory_points_xy / voxel).astype(np.int32, copy=False)
                keep_mask = np.asarray(
                    [tuple(cell.tolist()) not in visible_cells for cell in memory_cells],
                    dtype=bool,
                )
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

    def _update_obstacle_memory(self, world_points_xy: np.ndarray, *, timestamp: float) -> None:
        if not bool(self.config.obstacle_memory_enabled):
            return

        now = float(timestamp)
        if self._obstacle_memory_timestamps.size != 0:
            keep_mask = (now - self._obstacle_memory_timestamps) <= float(self.config.obstacle_memory_decay_s)
            self._obstacle_memory_points_xy = self._obstacle_memory_points_xy[keep_mask]
            self._obstacle_memory_timestamps = self._obstacle_memory_timestamps[keep_mask]

        if world_points_xy.size != 0:
            if self._obstacle_memory_points_xy.size == 0:
                merged_points = world_points_xy.astype(np.float32, copy=True)
                merged_timestamps = np.full((len(world_points_xy),), now, dtype=np.float64)
            else:
                merged_points = np.vstack((self._obstacle_memory_points_xy, world_points_xy)).astype(
                    np.float32,
                    copy=False,
                )
                merged_timestamps = np.concatenate(
                    (
                        self._obstacle_memory_timestamps,
                        np.full((len(world_points_xy),), now, dtype=np.float64),
                    )
                )

            voxel = max(float(self.config.obstacle_memory_voxel_m), 1e-3)
            quantized = np.round(merged_points / voxel).astype(np.int64, copy=False)
            # Collapse to one point per voxel cell, keeping the most recently
            # observed point in each cell. Sorting by timestamp ascending and then
            # taking the last occurrence per unique cell yields the newest sample.
            order_by_ts = np.argsort(merged_timestamps, kind="stable")
            cells_sorted = quantized[order_by_ts]
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
        )
        local_points_xy = (
            np.asarray([(forward_m, lateral_m) for forward_m, lateral_m, _, _ in local_points], dtype=np.float32)
            if local_points
            else np.zeros((0, 2), dtype=np.float32)
        )

        pose, accepted, score = self._resolve_scan_pose(scan, local_points_xy)
        if pose is None:
            if self._hold_mapping_for_turn():
                self._scan_count += 1
                if self._scan_count % max(int(self.config.log_every_scans), 1) == 0:
                    logger.info(
                        "Holding LiDAR map insertion during turn stabilization",
                        holdoff_s=round(float(self.config.mapping_holdoff_after_turn_s), 3),
                    )
                self._update_obstacle_memory(np.zeros((0, 2), dtype=np.float32), timestamp=float(scan.ts))
            return
        if not accepted and score is not None:
            logger.debug("Skipping LiDAR frame after failed scan match score=%.4f", float(score))
            return

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
        )
        if len(cloud) == 0 and not bool(self.config.publish_empty_clouds):
            return

        self.lidar.publish(cloud)
        self.localized_pose.publish(pose)
        registered_scan = build_registered_scan_from_local_points(
            local_points_xy,
            pose,
            z_height_m=float(self.config.registered_scan_height_m),
            frame_id=self.config.frame_id,
            timestamp=float(scan.ts),
            lidar_mount_x_m=float(self.config.lidar_mount_x_m),
            lidar_mount_y_m=float(self.config.lidar_mount_y_m),
        )
        self.registered_scan.publish(registered_scan)
        self._last_local_points = local_points_xy
        self._last_matched_pose = pose
        if local_points_xy.size != 0:
            sensor_x, sensor_y = _sensor_origin_xy(
                pose,
                mount_x_m=float(self.config.lidar_mount_x_m),
                mount_y_m=float(self.config.lidar_mount_y_m),
            )
            world_points_xy = _transform_local_points(
                local_points_xy,
                x=float(sensor_x),
                y=float(sensor_y),
                yaw=float(pose.yaw),
            )
            self._update_submap(world_points_xy)
            self._update_obstacle_memory(world_points_xy, timestamp=float(scan.ts))
        else:
            self._update_obstacle_memory(np.zeros((0, 2), dtype=np.float32), timestamp=float(scan.ts))
        self._scan_count += 1
        if self._scan_count % max(int(self.config.log_every_scans), 1) == 0:
            logger.info(
                "SourcceyLidarPointCloudAdapter published world cloud",
                scan_count=self._scan_count,
                points=len(cloud),
                pose_x=float(pose.x),
                pose_y=float(pose.y),
                pose_yaw=float(pose.yaw),
                scan_match_accepted=accepted,
                scan_match_score=None if score is None else round(float(score), 4),
            )
