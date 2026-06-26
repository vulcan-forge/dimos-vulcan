from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import SimpleOccupancyConfig
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.lidar_pointcloud_adapter import SourcceyLidarPointCloudAdapter
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher
from dimos.robot.diy.sourccey.occupancy_grid_exporter import SourcceyOccupancyGridExporter
from dimos.robot.diy.sourccey.pose_to_odometry import SourcceyPoseToOdometry

sourccey_lidar_mapping = autoconnect(
    sourccey_basic,
    SourcceyLidarScanPublisher.blueprint(),
    SourcceyLidarPointCloudAdapter.blueprint(
        forward_angle_deg=180.0,
        valid_angle_half_width_deg=90.0,
        invert_lateral_axis=True,
        max_distance_m=8.0,
        min_confidence=5,
        free_ray_step_m=0.05,
        free_ray_start_m=0.08,
        free_height_m=0.0,
        obstacle_height_m=0.25,
        odom_stale_after_s=0.75,
    ),
    SourcceyPoseToOdometry.blueprint(),
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
        (SourcceyPoseToOdometry, "pose", "localized_pose"),
        (SourcceyPoseToOdometry, "odometry", "localized_odometry"),
        (RayTracingVoxelMap, "odometry", "localized_odometry"),
        (SourcceyOccupancyGridExporter, "odom", "localized_pose"),
    ]
).global_config(n_workers=8)
