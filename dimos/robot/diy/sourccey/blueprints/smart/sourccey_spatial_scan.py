from __future__ import annotations

"""Sourccey spatial mapping scan blueprint.

Runs spatial memory and the automated segmented rotation in the same DimOS
coordinator so preview updates and stored frames continue while the robot
turns in place.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.smart.sourccey_spatial import sourccey_spatial
from dimos.robot.diy.sourccey.survey_rotation import SourcceySurveyRotation

sourccey_spatial_scan = autoconnect(
    sourccey_spatial,
    SourcceySurveyRotation.blueprint(),
).global_config(n_workers=6)
