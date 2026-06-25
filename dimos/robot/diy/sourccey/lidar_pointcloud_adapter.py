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
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

from .lidar_geometry import normalize_angle_deg
from .lidar_types import PlanarLidarScan

logger = setup_logger()


class SourcceyLidarPointCloudAdapterConfig(ModuleConfig):
    forward_angle_deg: float = 180.0
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


def _scan_to_local_points(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float,
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


def build_pointcloud_from_scan(
    scan: PlanarLidarScan,
    pose: PoseStamped,
    *,
    forward_angle_deg: float,
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


class SourcceyLidarPointCloudAdapter(Module):
    config: SourcceyLidarPointCloudAdapterConfig

    scan: In[PlanarLidarScan]
    odom: In[PoseStamped]
    lidar: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._scan_count = 0

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

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        odom = self._latest_odom
        if odom is None:
            logger.debug("Skipping Sourccey LiDAR scan because no odom has been received yet")
            return
        now = time.time()
        if abs(now - float(odom.ts)) > float(self.config.odom_stale_after_s):
            logger.debug("Skipping Sourccey LiDAR scan because odom is stale")
            return

        cloud = build_pointcloud_from_scan(
            scan,
            odom,
            forward_angle_deg=float(self.config.forward_angle_deg),
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
        self._scan_count += 1
        if self._scan_count % max(int(self.config.log_every_scans), 1) == 0:
            logger.info(
                "SourcceyLidarPointCloudAdapter published world cloud",
                scan_count=self._scan_count,
                points=len(cloud),
                pose_x=float(odom.x),
                pose_y=float(odom.y),
                pose_yaw=float(odom.yaw),
            )
