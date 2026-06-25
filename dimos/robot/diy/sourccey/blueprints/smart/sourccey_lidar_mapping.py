from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.lidar_occupancy_mapper import SourcceyLidarOccupancyMapper
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher
from dimos.robot.diy.sourccey.semantic_navigation import SourcceySemanticNavigator
from dimos.robot.diy.sourccey.visual_landmark_memory import SourcceyVisualLandmarkMemory

sourccey_lidar_mapping = autoconnect(
    sourccey_basic,
    SourcceyLidarScanPublisher.blueprint(),
    SourcceyLidarOccupancyMapper.blueprint(),
    SourcceyVisualLandmarkMemory.blueprint(),
    SourcceySemanticNavigator.blueprint(),
).global_config(n_workers=7)
