from __future__ import annotations

"""Sourccey sensor/debug blueprint.

Starts the native Sourccey connection and Rerun visualization from sourccey_basic,
then adds a live console monitor for camera freshness, odom, imu, and joint-state.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.sensor_debug_monitor import SourcceySensorDebugMonitor

sourccey_sensor_debug = autoconnect(
    sourccey_basic,
    SourcceySensorDebugMonitor.blueprint(),
).global_config(n_workers=4)
