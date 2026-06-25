from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import SimpleOccupancyConfig
from dimos.mapping.voxels import VoxelGridMapper
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.lidar_pointcloud_adapter import SourcceyLidarPointCloudAdapter
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher
from dimos.robot.diy.sourccey.occupancy_grid_exporter import SourcceyOccupancyGridExporter

sourccey_lidar_mapping = autoconnect(
    sourccey_basic,
    SourcceyLidarScanPublisher.blueprint(),
    SourcceyLidarPointCloudAdapter.blueprint(
        forward_angle_deg=180.0,
        max_distance_m=8.0,
        min_confidence=5,
        free_ray_step_m=0.05,
        free_ray_start_m=0.08,
        free_height_m=0.0,
        obstacle_height_m=0.25,
        odom_stale_after_s=0.75,
    ),
    VoxelGridMapper.blueprint(
        voxel_size=0.05,
        carve_columns=True,
        frame_id="world",
        device="CPU:0",
    ),
    CostMapper.blueprint(
        algo="simple",
        config=SimpleOccupancyConfig(
            resolution=0.05,
            frame_id="world",
            min_height=0.10,
            max_height=0.60,
        ),
    ),
    SourcceyOccupancyGridExporter.blueprint(),
).global_config(n_workers=8)
