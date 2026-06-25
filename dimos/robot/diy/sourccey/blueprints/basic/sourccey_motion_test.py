from __future__ import annotations

"""Small scripted motion test for Sourccey.

Useful for validating the Dimos -> cmd_vel -> robot path without needing the
pygame keyboard teleop window.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.motion_test import SourcceyMotionTest

sourccey_motion_test = autoconnect(
    sourccey_basic,
    SourcceyMotionTest.blueprint(),
)
