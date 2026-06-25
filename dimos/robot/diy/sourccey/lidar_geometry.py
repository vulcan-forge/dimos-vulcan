from __future__ import annotations

import math

import numpy as np

from .lidar_types import PlanarLidarScan, StopZoneConfig, StopZoneState


def normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def scan_to_local_xy(
    scan: PlanarLidarScan,
    *,
    forward_angle_deg: float = 180.0,
    max_distance_m: float | None = None,
    min_confidence: int = 0,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for angle_deg, distance_m, confidence in zip(
        scan.angles_deg,
        scan.distances_m,
        scan.confidences,
        strict=False,
    ):
        if int(confidence) < int(min_confidence):
            continue
        distance = float(distance_m)
        if distance <= 0.0:
            continue
        if max_distance_m is not None and distance > max_distance_m:
            continue
        delta_deg = normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        theta = math.radians(delta_deg)
        forward_m = distance * math.cos(theta)
        lateral_m = distance * math.sin(theta)
        points.append((forward_m, lateral_m))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def point_in_stop_zone(
    angle_deg: float,
    distance_m: float,
    *,
    cfg: StopZoneConfig,
) -> bool:
    if distance_m <= 0.0:
        return False
    delta_deg = normalize_angle_deg(float(angle_deg) - float(cfg.forward_angle_deg))
    theta = math.radians(delta_deg)
    forward_m = float(distance_m) * math.cos(theta)
    lateral_m = float(distance_m) * math.sin(theta)
    return (
        int(cfg.min_confidence) >= 0
        and forward_m >= float(cfg.min_distance_m)
        and forward_m <= float(cfg.tripwire_distance_m) + float(cfg.tripwire_thickness_m) / 2.0
        and abs(lateral_m) <= float(cfg.tripwire_half_width_m)
    )


def detect_stop_zone(scan: PlanarLidarScan, *, cfg: StopZoneConfig) -> StopZoneState:
    blocking_points = 0
    nearest_distance: float | None = None
    for angle_deg, distance_m, confidence in zip(
        scan.angles_deg,
        scan.distances_m,
        scan.confidences,
        strict=False,
    ):
        if int(confidence) < int(cfg.min_confidence):
            continue
        distance = float(distance_m)
        if not point_in_stop_zone(float(angle_deg), distance, cfg=cfg):
            continue
        blocking_points += 1
        nearest_distance = distance if nearest_distance is None else min(nearest_distance, distance)
    threshold = max(int(cfg.min_points_to_trigger), 1)
    return StopZoneState(
        ts=float(scan.ts),
        blocked=blocking_points >= threshold,
        blocking_points=blocking_points,
        threshold_points=threshold,
        nearest_blocking_distance_m=nearest_distance,
    )

