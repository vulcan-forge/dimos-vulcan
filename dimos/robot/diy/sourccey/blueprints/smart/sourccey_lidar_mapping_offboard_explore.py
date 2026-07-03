from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.smart.sourccey_lidar_mapping_offboard_teleop import (
    sourccey_lidar_mapping_offboard_teleop,
)
from dimos.robot.diy.sourccey.wander_explorer import SourcceyWanderExplorer

# Autonomous "wander and map" stack:
# - reuses the offboard mapping + teleop blueprint (mapping, obstacle memory,
#   web teleop, and the LiDAR safety gate that hard-stops at walls)
# - adds SourcceyWanderExplorer, which drives forward, turns toward open and
#   unexplored space when blocked, and stops once the room is mapped.
#
# The explorer's cmd_vel is routed into the safety gate's input
# ("teleop_cmd_vel"), so every autonomous command still passes through the
# collision stop. Web teleop remains available and overrides the explorer when
# the joystick is used; the web "Start/Stop Explore" buttons toggle wandering.
sourccey_lidar_mapping_offboard_explore = (
    autoconnect(
        sourccey_lidar_mapping_offboard_teleop,
        SourcceyWanderExplorer.blueprint(
            auto_start=True,
            start_delay_s=1.0,
            startup_min_free_cells=0,
            startup_stabilization_s=0.6,
            startup_require_costmap=False,
            scan_min_range_m=0.3,
            turn_only_snapshot_mode=True,
            turn_only_step_deg=45.0,
            completion_enabled=False,
            # turn_trigger is the floor; the velocity-based lookahead raises the
            # effective trigger with speed (kept below resume_clearance for
            # drive<->turn hysteresis).
            turn_trigger_distance_m=0.24,
            resume_clearance_m=0.42,
            min_heading_clearance_m=0.35,
            redirect_when_explored=False,
            revisit_penalty_weight=10.0,
            revisit_memory_size=10,
            # Throttle units, not rad/s: the base needs ~0.8+ to actually rotate.
            turn_speed_rad_s=0.95,
            # Turn continuously and fast, then fully stop and let the mapper fit
            # stationary snapshots before any more motion.
            turn_burst_s=0.0,
            turn_settle_pause_s=0.0,
            turn_burst_mapping_enabled=False,
            turn_burst_mapping_min_wait_s=1.0,
            turn_burst_mapping_max_wait_s=2.0,
            drive_burst_s=0.65,
            drive_burst_mapping_enabled=True,
            drive_burst_mapping_min_wait_s=2.0,
            drive_burst_mapping_max_wait_s=4.5,
            cruise_speed_m_s=0.24,
            kick_speed_m_s=0.90,
            kick_duration_s=0.45,
            re_kick_speed_m_s=0.78,
            re_kick_duration_s=0.28,
            re_kick_period_s=1.4,
            # After each turn, pause and let the mapper commit a fresh stationary
            # snapshot before driving again. This is intentionally conservative.
            post_turn_mapping_enabled=True,
            post_turn_mapping_min_wait_s=5.0,
            post_turn_mapping_max_wait_s=18.0,
            # In precision snapshot mode we would rather stop and wait than let
            # the robot keep turning while the global map is frozen. Require a
            # real localized commit and downstream costmap refresh before the
            # next 45-degree step is allowed to begin.
            post_turn_require_pose_update=False,
            post_turn_require_costmap_update=True,
            post_turn_require_scan_update=True,
            post_turn_require_localized_commit=True,
            settle_min_pose_updates=1,
            settle_min_costmap_updates=1,
            settle_min_scan_updates=10,
            debug_enabled=True,
            debug_min_interval_s=0.25,
        ),
    )
    .remappings(
        [
            (SourcceyWanderExplorer, "odom", "odom"),
            (SourcceyWanderExplorer, "localized_pose", "localized_pose"),
            (SourcceyWanderExplorer, "cmd_vel", "teleop_cmd_vel"),
        ]
    )
    .global_config(n_workers=10)
)
