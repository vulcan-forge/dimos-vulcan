from __future__ import annotations

"""Sourccey automated survey scan blueprint.

Runs the survey recorder while commanding a segmented base rotation so capture
quality checks can be repeated consistently without manual teleop.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_survey_recorder import sourccey_survey_recorder
from dimos.robot.diy.sourccey.survey_rotation import SourcceySurveyRotation

sourccey_survey_scan = autoconnect(
    sourccey_survey_recorder,
    SourcceySurveyRotation.blueprint(),
).global_config(n_workers=6)
