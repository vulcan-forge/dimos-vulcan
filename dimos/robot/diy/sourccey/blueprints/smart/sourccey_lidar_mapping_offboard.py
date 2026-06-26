from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import SimpleOccupancyConfig
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.connection import SourcceyConnection
from dimos.robot.diy.sourccey.lidar_pointcloud_adapter import SourcceyLidarPointCloudAdapter
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher
from dimos.robot.diy.sourccey.occupancy_grid_exporter import SourcceyOccupancyGridExporter
from dimos.robot.diy.sourccey.pose_to_odometry import SourcceyPoseToOdometry
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


# Offboard/client-side mapping:
# - the Pi keeps the robot host, cameras, IMU, and raw LiDAR TCP streamer alive
# - the client Dimos process connects to those streams and does the heavy mapping work
sourccey_lidar_mapping_offboard = autoconnect(
    sourccey_basic,
    SourcceyLidarScanPublisher.blueprint(
        source="tcp",
        port=8765,
    ),
    SourcceyLidarPointCloudAdapter.blueprint(
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        invert_lateral_axis=True,
        lidar_mount_x_m=0.2286,
        lidar_mount_y_m=0.0,
        max_distance_m=8.0,
        min_confidence=5,
        free_ray_step_m=0.05,
        free_ray_start_m=0.08,
        free_height_m=0.0,
        obstacle_height_m=0.25,
        odom_stale_after_s=0.75,
        scan_match_translation_window_m=0.12,
        scan_match_rotation_window_deg=6.0,
        scan_match_accept_score_m=0.06,
        reset_context_translation_m=0.20,
        reset_context_rotation_deg=16.0,
        min_points_for_scan_match=60,
        max_odom_angular_speed_for_mapping_rad_s=0.22,
        max_odom_linear_speed_for_mapping_m_s=0.14,
        mapping_holdoff_after_turn_s=0.45,
        obstacle_memory_decay_s=3600.0,
        obstacle_memory_max_points=8000,
        obstacle_memory_voxel_m=0.04,
        obstacle_memory_height_m=0.32,
    ),
    SourcceyPoseToOdometry.blueprint(
        child_frame_id="base_lidar",
        lidar_mount_x_m=0.2286,
        lidar_mount_y_m=0.0,
    ),
    RayTracingVoxelMap.blueprint(
        voxel_size=0.05,
        max_range=8.0,
        ray_subsample=1,
        shadow_depth=0.05,
        grace_depth=0.05,
        min_health=-1,
        max_health=1,
        recency_window=4,
    ),
    CostMapper.blueprint(
        algo="simple",
        config=SimpleOccupancyConfig(
            resolution=0.05,
            frame_id="world",
            min_height=0.08,
            max_height=0.40,
        ),
        initial_safe_radius_meters=0.30,
    ),
    SourcceyOccupancyGridExporter.blueprint(),
).remappings(
    [
        (SourcceyConnection, "localized_pose", "localized_pose"),
        (WebsocketVisModule, "odom", "localized_pose"),
        (SourcceyPoseToOdometry, "pose", "localized_pose"),
        (SourcceyPoseToOdometry, "odometry", "localized_odometry"),
        (RayTracingVoxelMap, "odometry", "localized_odometry"),
        (SourcceyOccupancyGridExporter, "odom", "localized_pose"),
    ]
).global_config(n_workers=8)
