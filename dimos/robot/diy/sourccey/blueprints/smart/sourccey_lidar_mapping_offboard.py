from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.connection import SourcceyConnection
from dimos.robot.diy.sourccey.lidar_occupancy_mapper import SourcceyLidarOccupancyMapper
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher
from dimos.robot.diy.sourccey.occupancy_grid_exporter import SourcceyOccupancyGridExporter
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


# Offboard/client-side mapping:
# - the Pi keeps the robot host, cameras, IMU, and raw LiDAR TCP streamer alive
# - the client Dimos process connects to those streams and does the heavy mapping work
# - the dedicated occupancy mapper commits only settled stationary snapshots
#   instead of continuously smearing scans through the old ray-tracing chain
sourccey_lidar_mapping_offboard = autoconnect(
    sourccey_basic,
    SourcceyLidarScanPublisher.blueprint(
        source="tcp",
        port=8765,
    ),
    SourcceyLidarOccupancyMapper.blueprint(
        forward_angle_deg=270.0,
        valid_angle_half_width_deg=90.0,
        invert_lateral_axis=True,
        lidar_mount_x_m=0.2286,
        lidar_mount_y_m=0.0,
        min_range_m=0.20,
        max_distance_m=8.0,
        min_confidence=5,
        scan_match_translation_window_m=0.35,
        scan_match_rotation_window_deg=120.0,
        scan_match_accept_score_m=0.11,
        scan_match_overlap_radius_m=0.10,
        scan_match_min_overlap_fraction=0.12,
        submap_max_points=2400,
        submap_local_radius_m=6.0,
        stationary_required_scans=8,
        stationary_keyframe_voxel_m=0.03,
        stationary_keyframe_max_points=420,
        cmd_vel_gate_enabled=True,
        cmd_vel_active_window_s=1.0,
        max_odom_linear_speed_for_commit_m_s=0.04,
        max_odom_angular_speed_for_commit_rad_s=0.08,
        obstacle_memory_enabled=True,
        obstacle_memory_decay_s=180.0,
        obstacle_memory_max_points=2500,
        obstacle_memory_voxel_m=0.05,
        obstacle_memory_height_m=0.32,
        obstacle_memory_visible_clear_radius_m=0.12,
    ),
    SourcceyOccupancyGridExporter.blueprint(),
).remappings(
    [
        (SourcceyConnection, "localized_pose", "localized_pose"),
        (WebsocketVisModule, "odom", "odom"),
        (SourcceyOccupancyGridExporter, "odom", "localized_pose"),
        (SourcceyLidarOccupancyMapper, "cmd_vel", "cmd_vel"),
    ]
).global_config(n_workers=6)
