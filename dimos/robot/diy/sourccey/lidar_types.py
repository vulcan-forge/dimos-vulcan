from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PlanarLidarScan:
    ts: float
    frame_id: str
    rpm: float
    angles_deg: list[float]
    distances_m: list[float]
    confidences: list[int]

    @classmethod
    def from_points(
        cls,
        *,
        ts: float,
        frame_id: str,
        rpm: float,
        points: list[list[float]] | list[tuple[float, float, int]],
    ) -> "PlanarLidarScan":
        angles_deg: list[float] = []
        distances_m: list[float] = []
        confidences: list[int] = []
        for angle_deg, distance_m, confidence in points:
            angles_deg.append(float(angle_deg))
            distances_m.append(float(distance_m))
            confidences.append(int(confidence))
        return cls(
            ts=float(ts),
            frame_id=frame_id,
            rpm=float(rpm),
            angles_deg=angles_deg,
            distances_m=distances_m,
            confidences=confidences,
        )

    @property
    def point_count(self) -> int:
        return len(self.angles_deg)


@dataclass(slots=True)
class StopZoneConfig:
    forward_angle_deg: float = 270.0
    min_distance_m: float = 0.03
    tripwire_distance_m: float = 0.14
    tripwire_half_width_m: float = 0.28
    tripwire_thickness_m: float = 0.12
    min_points_to_trigger: int = 8
    min_confidence: int = 0


@dataclass(slots=True)
class StopZoneState:
    ts: float
    blocked: bool
    blocking_points: int
    threshold_points: int
    nearest_blocking_distance_m: float | None
