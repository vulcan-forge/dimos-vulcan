from __future__ import annotations

from dimos.robot.diy.sourccey.lidar_geometry import detect_stop_zone, scan_to_local_xy
from dimos.robot.diy.sourccey.lidar_types import PlanarLidarScan, StopZoneConfig


def test_scan_to_local_xy_respects_forward_angle() -> None:
    scan = PlanarLidarScan(
        ts=1.0,
        frame_id="base_lidar",
        rpm=600.0,
        angles_deg=[180.0, 270.0],
        distances_m=[1.0, 1.0],
        confidences=[100, 100],
    )

    points = scan_to_local_xy(scan, forward_angle_deg=180.0)

    assert points.shape == (2, 2)
    assert points[0][0] > 0.99
    assert abs(float(points[0][1])) < 1e-6
    assert abs(float(points[1][0])) < 1e-6
    assert points[1][1] < -0.99


def test_detect_stop_zone_counts_tripwire_points() -> None:
    cfg = StopZoneConfig(
        forward_angle_deg=180.0,
        min_distance_m=0.12,
        tripwire_distance_m=0.16,
        tripwire_half_width_m=0.10,
        tripwire_thickness_m=0.06,
        min_points_to_trigger=2,
        min_confidence=0,
    )
    scan = PlanarLidarScan(
        ts=2.0,
        frame_id="base_lidar",
        rpm=598.0,
        angles_deg=[180.0, 184.0, 90.0],
        distances_m=[0.14, 0.15, 0.15],
        confidences=[120, 110, 120],
    )

    state = detect_stop_zone(scan, cfg=cfg)

    assert state.blocked is True
    assert state.blocking_points == 2
    assert state.nearest_blocking_distance_m == 0.14
