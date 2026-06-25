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
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid

from .lidar_geometry import scan_to_local_xy
from .lidar_types import PlanarLidarScan

_DEFAULT_EXPORT_ROOT = DIMOS_PROJECT_ROOT / "assets" / "output" / "sourccey_maps"


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
    export_dir: str = str(_DEFAULT_EXPORT_ROOT)
    export_png_name: str = "latest_map.png"
    export_metadata_name: str = "latest_map.json"
    export_interval_s: float = 1.0


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
        self._last_export_ts = 0.0
        self._export_dir = Path(self.config.export_dir)
        self._export_png_path = self._export_dir / self.config.export_png_name
        self._export_metadata_path = self._export_dir / self.config.export_metadata_name

    @rpc
    def start(self) -> None:
        super().start()
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._write_map_export(ts=time.time(), pose=self._current_pose())
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.landmark_pose.subscribe(self._on_landmark_pose)))
        self.register_disposable(Disposable(self.scan.subscribe(self._on_scan)))

    @rpc
    def stop(self) -> None:
        self._write_map_export(ts=time.time(), pose=self._current_pose())
        super().stop()

    @rpc
    def reset_map(self) -> None:
        self._grid.fill(int(CostValues.UNKNOWN))
        self._write_map_export(ts=time.time(), pose=self._current_pose())

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
        self._maybe_export_map(ts=float(scan.ts), pose=pose)

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

    def _maybe_export_map(self, ts: float, pose: PoseStamped | None) -> None:
        interval_s = max(0.1, float(self.config.export_interval_s))
        if ts - self._last_export_ts < interval_s:
            return
        self._write_map_export(ts=ts, pose=pose)
        self._last_export_ts = ts

    def _write_map_export(self, ts: float, pose: PoseStamped | None) -> None:
        image = np.full((self._height, self._width), 127, dtype=np.uint8)
        image[self._grid == int(CostValues.FREE)] = 255
        image[self._grid >= int(CostValues.OCCUPIED)] = 0
        image = np.flipud(image)
        cv2.imwrite(str(self._export_png_path), image)

        occupied_cells = int(np.sum(self._grid >= int(CostValues.OCCUPIED)))
        free_cells = int(np.sum(self._grid == int(CostValues.FREE)))
        unknown_cells = int(np.sum(self._grid == int(CostValues.UNKNOWN)))
        metadata = {
            "schema": "sourccey.occupancy_map.v1",
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
            "Sourccey occupancy map exported png=%s metadata=%s occupied=%s free=%s unknown=%s",
            self._export_png_path,
            self._export_metadata_path,
            occupied_cells,
            free_cells,
            unknown_cells,
        )
