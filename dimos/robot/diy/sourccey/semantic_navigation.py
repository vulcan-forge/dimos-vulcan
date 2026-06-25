from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from reactivex.disposable import Disposable

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.utils.logging_config import setup_logger

from .visual_landmark_memory import LandmarkRecord, load_landmark_records

logger = setup_logger()

_DEFAULT_STORE_PATH = DIMOS_PROJECT_ROOT / "assets" / "output" / "sourccey_landmarks" / "landmarks.pkl"


class SourcceySemanticNavigatorConfig(ModuleConfig):
    store_path: str = str(_DEFAULT_STORE_PATH)
    goal_frame_id: str = "map"


class SourcceySemanticNavigator(Module):
    config: SourcceySemanticNavigatorConfig

    localized_pose: In[PoseStamped]
    global_costmap: In[OccupancyGrid]

    goal_request: Out[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_pose: PoseStamped | None = None
        self._latest_costmap: OccupancyGrid | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.localized_pose.subscribe(self._on_pose)))
        self.register_disposable(Disposable(self.global_costmap.subscribe(self._on_costmap)))

    @rpc
    def list_places(self) -> list[str]:
        return sorted(
            {
                str(record.get("name"))
                for record in self._records()
                if record.get("name")
            }
        )

    @rpc
    def resolve_goal(self, query: str) -> dict[str, Any] | None:
        records = self._records()
        if not records:
            return None
        query_lower = query.strip().lower()
        query_tokens = [token for token in query_lower.split() if token]
        current_pose = self._latest_pose

        best_record: LandmarkRecord | None = None
        best_score = -1.0
        best_distance = float("inf")
        for record in records:
            fields = " ".join(
                str(value).lower()
                for value in (record.get("name"), record.get("room"), record.get("note"))
                if value
            )
            score = 0.0
            if str(record.get("name", "")).lower() == query_lower:
                score += 100.0
            if query_lower and query_lower in fields:
                score += 30.0
            score += sum(8.0 for token in query_tokens if token in fields)
            if score <= 0.0:
                continue

            pose = tuple(record.get("pose", (0.0, 0.0, 0.0)))
            distance = float("inf")
            if current_pose is not None:
                distance = math.hypot(float(current_pose.x) - float(pose[0]), float(current_pose.y) - float(pose[1]))
                score -= 0.05 * distance
            if score > best_score or (math.isclose(score, best_score) and distance < best_distance):
                best_record = record
                best_score = score
                best_distance = distance

        if best_record is None:
            return None

        return {
            "name": best_record.get("name"),
            "room": best_record.get("room"),
            "note": best_record.get("note"),
            "pose": tuple(best_record.get("pose", (0.0, 0.0, 0.0))),
            "yaw_rad": float(best_record.get("yaw_rad", 0.0)),
            "score": best_score,
        }

    @rpc
    def navigate_to(self, query: str) -> bool:
        goal = self.resolve_goal(query)
        if goal is None:
            return False
        self.goal_request.publish(self._pose_from_goal(goal))
        logger.info("Semantic goal resolved: %s -> %s", query, goal)
        return True

    @rpc
    def navigate_to_room(self, room_name: str) -> bool:
        room_lower = room_name.strip().lower()
        room_records = [
            record
            for record in self._records()
            if str(record.get("room", "")).strip().lower() == room_lower
        ]
        if not room_records:
            return False

        xs = [float(record.get("pose", (0.0, 0.0, 0.0))[0]) for record in room_records]
        ys = [float(record.get("pose", (0.0, 0.0, 0.0))[1]) for record in room_records]
        zs = [float(record.get("pose", (0.0, 0.0, 0.0))[2]) for record in room_records]
        mean_goal = {
            "name": room_name,
            "room": room_name,
            "note": f"centroid of room '{room_name}'",
            "pose": (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)),
            "yaw_rad": float(room_records[0].get("yaw_rad", 0.0)),
            "score": float(len(room_records)),
        }
        self.goal_request.publish(self._pose_from_goal(mean_goal))
        return True

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest_pose = msg

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._latest_costmap = msg

    def _records(self) -> list[LandmarkRecord]:
        return load_landmark_records(self.config.store_path)

    def _pose_from_goal(self, goal: dict[str, Any]) -> PoseStamped:
        x, y, z = goal["pose"]
        yaw = float(goal.get("yaw_rad", 0.0))
        return PoseStamped(
            ts=0.0,
            frame_id=self.config.goal_frame_id,
            position=Vector3(float(x), float(y), float(z)),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
        )
