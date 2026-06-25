from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.smart.sourccey_lidar_mapping import sourccey_lidar_mapping
from dimos.robot.diy.sourccey.lidar_safety_gate import SourcceyLidarSafetyGate
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


sourccey_lidar_mapping_teleop = autoconnect(
    sourccey_lidar_mapping,
    SourcceyLidarSafetyGate.blueprint(),
).remappings(
    [
        (WebsocketVisModule, "tele_cmd_vel", "teleop_cmd_vel"),
        (SourcceyLidarSafetyGate, "cmd_vel_in", "teleop_cmd_vel"),
        (SourcceyLidarSafetyGate, "cmd_vel", "cmd_vel"),
    ]
)
