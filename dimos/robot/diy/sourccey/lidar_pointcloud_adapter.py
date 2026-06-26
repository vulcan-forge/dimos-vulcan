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


def _scan_to_local_points(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
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
        _transform_xy(np.vstack(free_points_local).astype(np.float32, copy=False), pose)
        if free_points_local
        else np.zeros((0, 2), dtype=np.float32)
    )
    obstacle_world_xy = _transform_xy(obstacle_points_local, pose)

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
) -> PointCloud2:
    if local_points_xy.size == 0:
        return PointCloud2.from_numpy(np.zeros((0, 3), dtype=np.float32), frame_id=frame_id, timestamp=timestamp)

    world_xy = _transform_local_points(
        local_points_xy,
        x=float(pose.x),
        y=float(pose.y),
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._scan_count = 0
        self._last_local_points: np.ndarray | None = None
        self._last_matched_pose: PoseStamped | None = None
        self._submap_points_world = np.zeros((0, 2), dtype=np.float32)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_odom(self, msg: PoseStamped) -> None:
        self._latest_odom = msg

    def _reference_world_points(self) -> np.ndarray:
        if self._submap_points_world.size != 0:
            return self._submap_points_world
        if self._last_local_points is None or self._last_matched_pose is None:
            return np.zeros((0, 2), dtype=np.float32)
        return _transform_local_points(
            _downsample_points(self._last_local_points, max(int(self.config.scan_match_max_points), 24)),
            x=float(self._last_matched_pose.x),
            y=float(self._last_matched_pose.y),
            yaw=float(self._last_matched_pose.yaw),
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
        best_score = _scan_match_score(
            reference_world,
            _transform_local_points(source_local, x=best_x, y=best_y, yaw=best_yaw),
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
                        score = _scan_match_score(
                            reference_world,
                            _transform_local_points(source_local, x=cand_x, y=cand_y, yaw=cand_yaw),
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

        if (
            not bool(self.config.scan_match_enabled)
            or self._last_local_points is None
            or self._last_matched_pose is None
            or local_points_xy.size == 0
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

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        local_points = _scan_to_local_points(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
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
            return
        if not accepted and score is not None:
            logger.debug("Skipping LiDAR frame after failed scan match score=%.4f", float(score))
            return

        cloud = build_pointcloud_from_scan(
            scan,
            pose,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
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
        )
        self.registered_scan.publish(registered_scan)
        self._last_local_points = local_points_xy
        self._last_matched_pose = pose
        if local_points_xy.size != 0:
            world_points_xy = _transform_local_points(
                local_points_xy,
                x=float(pose.x),
                y=float(pose.y),
                yaw=float(pose.yaw),
            )
            self._update_submap(world_points_xy)
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
