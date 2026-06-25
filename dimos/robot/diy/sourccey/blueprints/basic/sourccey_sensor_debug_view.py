from __future__ import annotations

"""Sourccey sensor/debug blueprint with local OpenCV feed windows."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_sensor_debug import sourccey_sensor_debug
from dimos.robot.diy.sourccey.opencv_debug_viewer import SourcceyOpenCVDebugViewer

sourccey_sensor_debug_view = autoconnect(
    sourccey_sensor_debug,
    SourcceyOpenCVDebugViewer.blueprint(),
).global_config(n_workers=5)
