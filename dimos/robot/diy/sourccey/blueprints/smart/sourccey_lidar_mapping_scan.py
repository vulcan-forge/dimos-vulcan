from __future__ import annotations

"""Sourccey native LiDAR mapping plus automated survey rotation."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.smart.sourccey_lidar_mapping import sourccey_lidar_mapping
from dimos.robot.diy.sourccey.survey_rotation import SourcceySurveyRotation

sourccey_lidar_mapping_scan = autoconnect(
    sourccey_lidar_mapping,
    SourcceySurveyRotation.blueprint(),
).global_config(n_workers=8)
