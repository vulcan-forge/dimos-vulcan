from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.lidar_safety_gate import SourcceyLidarSafetyGate
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop

sourccey_lidar_keyboard_teleop = (
    autoconnect(
        sourccey_basic,
        KeyboardTeleop.blueprint(
            linear_speed=0.95,
            angular_speed=0.8,
            boost_multiplier=1.2,
            slow_multiplier=0.75,
            publish_only_when_active=True,
        ),
        SourcceyLidarScanPublisher.blueprint(),
        SourcceyLidarSafetyGate.blueprint(),
    )
    .remappings(
        [
            (KeyboardTeleop, "cmd_vel", "teleop_cmd_vel"),
            (SourcceyLidarSafetyGate, "cmd_vel_in", "teleop_cmd_vel"),
            (SourcceyLidarSafetyGate, "cmd_vel", "cmd_vel"),
        ]
    )
    .global_config(n_workers=6)
)
