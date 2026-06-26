from __future__ import annotations

from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry


class SourcceyPoseToOdometryConfig(ModuleConfig):
    child_frame_id: str = "base_link"


class SourcceyPoseToOdometry(Module):
    config: SourcceyPoseToOdometryConfig

    pose: In[PoseStamped]
    odometry: Out[Odometry]

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.pose.subscribe(self._on_pose)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_pose(self, msg: PoseStamped) -> None:
        self.odometry.publish(
            Odometry(
                ts=float(msg.ts),
                frame_id=str(msg.frame_id or "world"),
                child_frame_id=str(self.config.child_frame_id),
                pose=Pose(
                    position=[float(msg.x), float(msg.y), float(msg.z)],
                    orientation=[
                        float(msg.orientation.x),
                        float(msg.orientation.y),
                        float(msg.orientation.z),
                        float(msg.orientation.w),
                    ],
                ),
                twist=Twist(
                    linear=Vector3(0.0, 0.0, 0.0),
                    angular=Vector3(0.0, 0.0, 0.0),
                ),
            )
        )
