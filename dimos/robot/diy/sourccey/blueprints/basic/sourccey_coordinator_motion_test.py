from __future__ import annotations

"""Scripted coordinator-path motion test for Sourccey.

This validates the full Dimos control stack by publishing a short Twist into the
coordinator command topic after Sourccey has had time to receive its first real
state packet.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMSharedMemoryTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_coordinator import sourccey_coordinator
from dimos.robot.diy.sourccey.motion_test import SourcceyMotionTest

sourccey_coordinator_motion_test = (
    autoconnect(
        sourccey_coordinator,
        SourcceyMotionTest.blueprint(
            startup_delay_s=4.0,
            move_duration_s=0.9,
            command_rate_hz=14.0,
            linear_x=0.95,
        ),
    )
    .transports(
        {
            ("cmd_vel", Twist): LCMSharedMemoryTransport("/cmd_vel", Twist),
        }
    )
)
