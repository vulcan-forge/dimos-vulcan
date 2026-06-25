from __future__ import annotations

"""Sourccey survey capture blueprint.

Captures primary/companion/bottom frames plus odom/imu metadata into a session
folder for later mapping or reconstruction passes.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.survey_recorder import SourcceySurveyRecorder

sourccey_survey_recorder = autoconnect(
    sourccey_basic,
    SourcceySurveyRecorder.blueprint(),
).global_config(n_workers=5)
