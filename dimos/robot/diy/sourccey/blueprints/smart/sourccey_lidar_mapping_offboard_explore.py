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
            start_delay_s=6.0,
            startup_min_free_cells=180,
            startup_stabilization_s=2.0,
            scan_min_range_m=0.3,
            # turn_trigger is the floor; the velocity-based lookahead raises the
            # effective trigger with speed (kept below resume_clearance for
            # drive<->turn hysteresis).
            turn_trigger_distance_m=0.32,
            resume_clearance_m=0.6,
            # Throttle units, not rad/s: the base needs ~0.8+ to actually rotate.
            turn_speed_rad_s=0.9,
            # Turn in longer bursts with a short settle so it doesn't dwell.
            turn_burst_s=0.5,
            turn_settle_pause_s=0.35,
        ),
    )
    .remappings(
        [
            (SourcceyWanderExplorer, "odom", "localized_pose"),
            (SourcceyWanderExplorer, "cmd_vel", "teleop_cmd_vel"),
        ]
    )
    .global_config(n_workers=10)
)
