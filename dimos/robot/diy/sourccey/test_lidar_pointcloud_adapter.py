from __future__ import annotations

import math

import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.robot.diy.sourccey.lidar_pointcloud_adapter import build_pointcloud_from_scan
from dimos.robot.diy.sourccey.lidar_types import PlanarLidarScan


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    return PoseStamped(
        ts=1.0,
        frame_id="world",
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def test_build_pointcloud_from_scan_generates_free_and_obstacle_points() -> None:
    scan = PlanarLidarScan(
        ts=1.0,
        frame_id="base_lidar",
        rpm=600.0,
        angles_deg=[180.0],
        distances_m=[1.0],
        confidences=[15],
    )

    cloud = build_pointcloud_from_scan(
        scan,
        _pose(0.0, 0.0, 0.0),
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        max_distance_m=8.0,
        min_confidence=0,
        free_ray_step_m=0.25,
        free_ray_start_m=0.25,
        free_height_m=0.0,
        obstacle_height_m=0.25,
        frame_id="world",
    )

    points, _ = cloud.as_numpy()
    intensities = cloud.intensities_f32()

    assert points.shape == (4, 3)
    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [0.25, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [1.00, 0.0, 0.25],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
    assert intensities is not None
    np.testing.assert_allclose(intensities, np.asarray([0.0, 0.0, 0.0, 15.0], dtype=np.float32))


def test_build_pointcloud_from_scan_applies_world_pose() -> None:
    scan = PlanarLidarScan(
        ts=1.0,
        frame_id="base_lidar",
        rpm=600.0,
        angles_deg=[180.0],
        distances_m=[1.0],
        confidences=[20],
    )

    cloud = build_pointcloud_from_scan(
        scan,
        _pose(1.0, 2.0, math.pi / 2.0),
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        max_distance_m=8.0,
        min_confidence=0,
        free_ray_step_m=0.5,
        free_ray_start_m=0.5,
        free_height_m=0.0,
        obstacle_height_m=0.25,
        frame_id="world",
    )

    points, _ = cloud.as_numpy()
    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [1.0, 2.5, 0.0],
                [1.0, 3.0, 0.25],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )


def test_build_pointcloud_from_scan_filters_back_half_plane() -> None:
    scan = PlanarLidarScan(
        ts=1.0,
        frame_id="base_lidar",
        rpm=600.0,
        angles_deg=[180.0, 0.0],
        distances_m=[1.0, 1.0],
        confidences=[20, 20],
    )

    cloud = build_pointcloud_from_scan(
        scan,
        _pose(0.0, 0.0, 0.0),
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        max_distance_m=8.0,
        min_confidence=0,
        free_ray_step_m=0.5,
        free_ray_start_m=0.5,
        free_height_m=0.0,
        obstacle_height_m=0.25,
        frame_id="world",
    )

    points, _ = cloud.as_numpy()
    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.25],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
