from __future__ import annotations

"""Sourccey ControlCoordinator blueprint.

Wires Sourccey's native Dimos connection into the generic control coordinator so
higher-level Dimos motion stacks can command the base through the standard
`twist_command` -> `cmd_vel` path, without requiring LCM on the local machine.
"""

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMSharedMemoryTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.diy.sourccey.connection import SourcceyConnection

_sourccey_joints = make_twist_base_joints("sourccey")

sourccey_coordinator = (
    autoconnect(
        SourcceyConnection.blueprint(),
        ControlCoordinator.blueprint(
            hardware=[
                HardwareComponent(
                    hardware_id="sourccey",
                    hardware_type=HardwareType.BASE,
                    joints=_sourccey_joints,
                    adapter_type="transport_lcm_shm",
                ),
            ],
            tasks=[
                TaskConfig(
                    name="vel_sourccey",
                    type="velocity",
                    joint_names=_sourccey_joints,
                    priority=10,
                ),
            ],
        ),
    )
    .remappings(
        [
            (ControlCoordinator, "twist_command", "cmd_vel"),
            (SourcceyConnection, "cmd_vel", "sourccey_cmd_vel"),
            (SourcceyConnection, "odom", "sourccey_odom"),
        ]
    )
    .transports(
        {
            ("cmd_vel", Twist): LCMSharedMemoryTransport("/cmd_vel", Twist),
            ("twist_command", Twist): LCMSharedMemoryTransport("/cmd_vel", Twist),
            ("sourccey_cmd_vel", Twist): LCMSharedMemoryTransport("/sourccey/cmd_vel", Twist),
            ("sourccey_odom", PoseStamped): LCMSharedMemoryTransport("/sourccey/odom", PoseStamped),
        }
    )
)
