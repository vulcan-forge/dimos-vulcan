from __future__ import annotations

import math
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid

from .lidar_geometry import scan_to_local_xy
from .lidar_types import PlanarLidarScan


def _blend_angle_rad(a: float, b: float, weight: float) -> float:
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return math.atan2(math.sin(a + delta * weight), math.cos(a + delta * weight))


def _ray_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps <= 0:
        return [(x0, y0)]
    xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
    ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)
    return list(zip(xs.tolist(), ys.tolist(), strict=False))


class SourcceyLidarOccupancyMapperConfig(ModuleConfig):
    map_size_m: float = 12.0
    resolution_m: float = 0.05
    max_distance_m: float = 6.0
    min_confidence: int = 0
    forward_angle_deg: float = 180.0
    landmark_pose_blend: float = 0.35
    landmark_pose_max_age_s: float = 2.0
    frame_id: str = "map"


class SourcceyLidarOccupancyMapper(Module):
    config: SourcceyLidarOccupancyMapperConfig

    scan: In[PlanarLidarScan]
    odom: In[PoseStamped]
    landmark_pose: In[PoseStamped]

    global_costmap: Out[OccupancyGrid]
    localized_pose: Out[PoseStamped]

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

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.landmark_pose.subscribe(self._on_landmark_pose)))
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))

    @rpc
    def reset_map(self) -> None:
        self._grid.fill(int(CostValues.UNKNOWN))

    def _on_odom(self, msg: PoseStamped) -> None:
        self._latest_odom = msg

    def _on_landmark_pose(self, msg: PoseStamped) -> None:
        self._latest_landmark_pose = msg

    def _current_pose(self) -> PoseStamped | None:
        odom = self._latest_odom
        if odom is None:
            return None

        landmark = self._latest_landmark_pose
        if landmark is None:
            return odom
        if abs(float(odom.ts) - float(landmark.ts)) > float(self.config.landmark_pose_max_age_s):
            return odom

        weight = min(max(float(self.config.landmark_pose_blend), 0.0), 1.0)
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

    def _on_scan(self, scan: PlanarLidarScan) -> None:
        pose = self._current_pose()
        if pose is None:
            return

        self.localized_pose.publish(pose)
        local_points = scan_to_local_xy(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            max_distance_m=float(self.config.max_distance_m),
            min_confidence=int(self.config.min_confidence),
        )
        if local_points.size == 0:
            self.global_costmap.publish(self._make_grid_msg(scan.ts))
            return

        robot_cell = self._world_to_grid(float(pose.x), float(pose.y))
        if robot_cell is None:
            return
        self._grid[robot_cell[1], robot_cell[0]] = int(CostValues.FREE)

        cos_yaw = math.cos(float(pose.yaw))
        sin_yaw = math.sin(float(pose.yaw))
        for forward_m, lateral_m in local_points:
            world_x = float(pose.x) + float(forward_m) * cos_yaw - float(lateral_m) * sin_yaw
            world_y = float(pose.y) + float(forward_m) * sin_yaw + float(lateral_m) * cos_yaw
            hit_cell = self._world_to_grid(world_x, world_y)
            if hit_cell is None:
                continue
            ray = _ray_cells(robot_cell[0], robot_cell[1], hit_cell[0], hit_cell[1])
            for free_x, free_y in ray[:-1]:
                if self._inside_grid(free_x, free_y) and self._grid[free_y, free_x] != int(
                    CostValues.OCCUPIED
                ):
                    self._grid[free_y, free_x] = int(CostValues.FREE)
            if self._inside_grid(hit_cell[0], hit_cell[1]):
                self._grid[hit_cell[1], hit_cell[0]] = int(CostValues.OCCUPIED)

        self.global_costmap.publish(self._make_grid_msg(scan.ts))

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
