from __future__ import annotations

"""Sourccey keyboard teleop over the native Dimos Sourccey connection."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop

# Sourccey's base seems to need materially stronger linear/strafe commands than
# the generic Dimos keyboard teleop defaults before it will actually translate.
# Turning already behaves well, so we keep angular near the stock value while
# lifting linear speed to match the robot's normal manual-drive range.
sourccey_keyboard_teleop = autoconnect(
    sourccey_basic,
    KeyboardTeleop.blueprint(
        linear_speed=0.95,
        angular_speed=0.8,
        boost_multiplier=1.2,
        slow_multiplier=0.75,
        publish_only_when_active=True,
    ),
)
