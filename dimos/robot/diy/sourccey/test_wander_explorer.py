# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the pure decision helpers of the Sourccey wander explorer."""

from __future__ import annotations

import math

import numpy as np

from dimos.msgs.nav_msgs.OccupancyGrid import CostValues
from dimos.robot.diy.sourccey.lidar_types import PlanarLidarScan
from dimos.robot.diy.sourccey.wander_explorer import (
    angle_diff,
    costmap_raymarch,
    forward_clearance,
    frontier_cell_count,
    scan_to_local_xy,
    sector_clearance,
    select_world_heading,
)

FREE = int(CostValues.FREE)
UNKNOWN = int(CostValues.UNKNOWN)
OCCUPIED = int(CostValues.OCCUPIED)


def test_angle_diff_wraps_across_pi() -> None:
    assert math.isclose(
        angle_diff(math.radians(170.0), math.radians(-170.0)), math.radians(-20.0), abs_tol=1e-6
    )


def test_scan_to_local_xy_places_forward_point_ahead() -> None:
    # forward_angle_deg=180 means a 180-degree raw bearing is straight ahead.
    scan = PlanarLidarScan(
        ts=0.0,
        frame_id="lidar",
        rpm=600.0,
        angles_deg=[180.0, 90.0],
        distances_m=[2.0, 1.0],
        confidences=[10, 10],
    )
    points = scan_to_local_xy(
        scan,
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        invert_lateral_axis=True,
        max_distance_m=8.0,
        min_confidence=5,
    )
    # First point is straight ahead: forward~2, lateral~0.
    assert math.isclose(float(points[0, 0]), 2.0, abs_tol=1e-4)
    assert abs(float(points[0, 1])) < 1e-4


def test_scan_to_local_xy_filters_low_confidence_and_range() -> None:
    scan = PlanarLidarScan(
        ts=0.0,
        frame_id="lidar",
        rpm=600.0,
        angles_deg=[180.0, 180.0, 180.0],
        distances_m=[2.0, 2.0, 99.0],
        confidences=[1, 10, 10],
    )
    points = scan_to_local_xy(
        scan,
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        invert_lateral_axis=True,
        max_distance_m=8.0,
        min_confidence=5,
    )
    # Low-confidence point and out-of-range point are dropped.
    assert points.shape[0] == 1


def test_forward_clearance_detects_wall_ahead() -> None:
    points = np.array([[0.5, 0.0], [3.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    clearance = forward_clearance(
        points,
        corridor_half_width_m=0.28,
        forward_cone_half_deg=35.0,
        max_distance_m=8.0,
    )
    # Nearest in-corridor point is at 0.5 m; the side point at y=2 is ignored.
    assert math.isclose(clearance, 0.5, abs_tol=1e-4)


def test_forward_clearance_open_when_nothing_ahead() -> None:
    points = np.array([[1.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
    clearance = forward_clearance(
        points,
        corridor_half_width_m=0.28,
        forward_cone_half_deg=35.0,
        max_distance_m=8.0,
    )
    assert math.isclose(clearance, 8.0, abs_tol=1e-4)


def _grid(width: int, height: int, fill: int) -> np.ndarray:
    return np.full((height, width), fill, dtype=np.int8)


def test_costmap_raymarch_stops_at_occupied_cell() -> None:
    grid = _grid(20, 20, FREE)
    grid[:, 10] = OCCUPIED  # vertical wall at grid x=10 -> world x=1.0 (res 0.1)
    clearance, unknown_fraction = costmap_raymarch(
        grid,
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.1,
        start_x=0.0,
        start_y=0.5,
        heading_rad=0.0,  # +x
        lookahead_m=3.0,
        occupancy_threshold=60,
    )
    assert math.isclose(clearance, 1.0, abs_tol=0.1)
    assert unknown_fraction == 0.0


def test_costmap_raymarch_counts_unknown_fraction() -> None:
    grid = _grid(40, 10, UNKNOWN)
    clearance, unknown_fraction = costmap_raymarch(
        grid,
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.1,
        start_x=0.0,
        start_y=0.5,
        heading_rad=0.0,
        lookahead_m=3.0,
        occupancy_threshold=60,
    )
    # Nothing occupied -> full lookahead; all unknown -> fraction 1.0.
    assert math.isclose(clearance, 3.0, abs_tol=1e-6)
    assert unknown_fraction == 1.0


def test_select_world_heading_prefers_unexplored_direction() -> None:
    # Free to the right (+x), unknown to the left (-x). Both are clear, so the
    # explorer should steer toward the unexplored -x side.
    grid = _grid(60, 20, FREE)
    grid[:, :30] = UNKNOWN  # left half (world x < 1.5) is unexplored
    heading, unknown_fraction, clearance = select_world_heading(
        grid,
        origin_x=-3.0,
        origin_y=-1.0,
        resolution=0.1,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        n_candidates=36,
        lookahead_m=1.0,
        occupancy_threshold=60,
        min_clearance_m=0.5,
        clearance_weight=1.0,
        explore_weight=2.5,
        turn_penalty_weight=0.1,
    )
    # Chosen heading should point into the unexplored -x hemisphere
    # (it takes the smallest turn that still faces unknown space).
    assert math.cos(heading) < 0.0
    assert unknown_fraction > 0.5
    assert clearance > 0.5


def test_sector_clearance_finds_nearest_in_sector() -> None:
    # Point straight ahead at 1.5 m and one off to the side at 0.5 m.
    points = np.array([[1.5, 0.0], [0.1, 0.5]], dtype=np.float32)
    ahead = sector_clearance(
        points,
        center_bearing_rad=0.0,
        sector_half_width_rad=math.radians(12.0),
        max_distance_m=3.0,
    )
    assert math.isclose(ahead, 1.5, abs_tol=1e-4)


def test_select_world_heading_rejects_scan_blocked_forward() -> None:
    # Map says everything is clear and unexplored (so without the scan the
    # explorer would happily drive straight). The live scan reports a wall
    # 0.3 m straight ahead, so forward must be rejected in favor of a turn.
    grid = _grid(80, 80, UNKNOWN)
    scan_points = np.array([[0.3, 0.0], [0.32, 0.05], [0.31, -0.05]], dtype=np.float32)
    heading, _unknown, clearance = select_world_heading(
        grid,
        origin_x=-4.0,
        origin_y=-4.0,
        resolution=0.1,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        n_candidates=36,
        lookahead_m=2.0,
        occupancy_threshold=60,
        min_clearance_m=0.7,
        clearance_weight=1.0,
        explore_weight=2.5,
        turn_penalty_weight=0.1,
        scan_points_local=scan_points,
        scan_fov_half_deg=90.0,
        sector_half_deg=12.0,
    )
    # Should not pick straight ahead (blocked); a real turn is required.
    assert abs(heading) > math.radians(20.0)
    assert clearance >= 0.7


def test_frontier_cell_count_counts_free_bordering_unknown() -> None:
    grid = _grid(5, 5, UNKNOWN)
    grid[2, 2] = FREE  # single free cell surrounded by unknown -> 1 frontier cell
    assert frontier_cell_count(grid) == 1


def test_frontier_cell_count_zero_when_no_unknown_border() -> None:
    grid = _grid(5, 5, FREE)
    assert frontier_cell_count(grid) == 0
