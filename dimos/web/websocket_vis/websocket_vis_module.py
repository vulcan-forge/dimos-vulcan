#!/usr/bin/env python3

# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WebSocket Visualization Module for Dimos navigation and mapping.

This module provides a WebSocket data server for real-time visualization.
The frontend is served from a separate HTML file.
"""

import asyncio
import json
from pathlib import Path as FilePath
import threading
import time
from typing import Any
import webbrowser

from dimos_lcm.std_msgs import Bool
from reactivex.disposable import Disposable
import socketio  # type: ignore[import-untyped]
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route
import uvicorn

from dimos.utils.data import get_data

# Path to the frontend HTML templates and command-center build
_TEMPLATES_DIR = FilePath(__file__).parent.parent / "templates"
_DASHBOARD_HTML = _TEMPLATES_DIR / "rerun_dashboard.html"
_COMMAND_CENTER_HTML = _TEMPLATES_DIR / "sourccey_command_center.html"
_COMMAND_CENTER_DIR = (
    FilePath(__file__).parent.parent / "command-center-extension" / "dist-standalone"
)
_DEFAULT_SOURCCEY_MAP_DIR = FilePath(__file__).parent.parent.parent.parent / "assets" / "output" / "sourccey_maps"

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.models import LatLon
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.logging_config import setup_logger

from .optimized_costmap import OptimizedCostmapEncoder

logger = setup_logger()

_browser_open_lock = threading.Lock()
_browser_opened = False


class WebsocketConfig(ModuleConfig):
    port: int = 7779


class WebsocketVisModule(Module):
    """
    WebSocket-based visualization module for real-time navigation data.

    This module provides a web interface for visualizing:
    - Robot position and orientation
    - Navigation paths
    - Costmaps
    - Interactive goal setting via mouse clicks

    Inputs:
        - robot_pose: Current robot position
        - path: Navigation path
        - global_costmap: Global costmap for visualization

    Outputs:
        - click_goal: Goal position from user clicks
    """

    config: WebsocketConfig

    # LCM inputs
    odom: In[PoseStamped]
    gps_location: In[LatLon]
    path: In[Path]
    global_costmap: In[OccupancyGrid]

    # LCM outputs
    goal_request: Out[PoseStamped]
    gps_goal: Out[LatLon]
    explore_cmd: Out[Bool]
    stop_explore_cmd: Out[Bool]
    tele_cmd_vel: Out[Twist]
    movecmd_stamped: Out[TwistStamped]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the WebSocket visualization module.

        Args:
            port: Port to run the web server on
            cfg: Optional global config for viewer settings
        """
        super().__init__(**kwargs)
        self._uvicorn_server_thread: threading.Thread | None = None
        self.sio: socketio.AsyncServer | None = None
        self.app = None
        self._broadcast_loop = None
        self._broadcast_thread = None
        self._uvicorn_server: uvicorn.Server | None = None

        self.vis_state = {}  # type: ignore[var-annotated]
        self.state_lock = threading.Lock()
        self.costmap_encoder = OptimizedCostmapEncoder(chunk_size=64)

        # Track GPS goal points for visualization
        self.gps_goal_points: list[dict[str, float]] = []
        logger.info(
            f"WebSocket visualization module initialized on port {self.config.port}, GPS goal tracking enabled"
        )

    def _start_broadcast_loop(self) -> None:
        def websocket_vis_loop() -> None:
            self._broadcast_loop = asyncio.new_event_loop()  # type: ignore[assignment]
            asyncio.set_event_loop(self._broadcast_loop)
            try:
                self._broadcast_loop.run_forever()  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
            finally:
                self._broadcast_loop.close()  # type: ignore[attr-defined]

        self._broadcast_thread = threading.Thread(target=websocket_vis_loop, daemon=True)  # type: ignore[assignment]
        self._broadcast_thread.start()  # type: ignore[attr-defined]

    @rpc
    def start(self) -> None:
        super().start()

        self._create_server()

        self._start_broadcast_loop()

        self._uvicorn_server_thread = threading.Thread(target=self._run_uvicorn_server, daemon=True)
        self._uvicorn_server_thread.start()

        # Only auto-open when the user chose web-based viewing.
        if self.config.g.viewer == "rerun" and self.config.g.rerun_open in ("web", "both"):
            url = f"http://localhost:{self.config.port}/"
            logger.info(f"Dimensional Command Center: {url}")

            global _browser_opened
            with _browser_open_lock:
                if not _browser_opened:
                    try:
                        webbrowser.open_new_tab(url)
                        _browser_opened = True
                    except Exception as e:
                        logger.debug(f"Failed to open browser: {e}")

        try:
            unsub = self.odom.subscribe(self._on_robot_pose)
            self.register_disposable(Disposable(unsub))
        except Exception:
            ...

        try:
            unsub = self.gps_location.subscribe(self._on_gps_location)
            self.register_disposable(Disposable(unsub))
        except Exception:
            ...

        try:
            unsub = self.path.subscribe(self._on_path)
            self.register_disposable(Disposable(unsub))
        except Exception:
            ...

        try:
            unsub = self.global_costmap.subscribe(self._on_global_costmap)
            self.register_disposable(Disposable(unsub))
        except Exception:
            ...

    @rpc
    def stop(self) -> None:
        if getattr(self, "_ws_stopped", False):
            return
        self._ws_stopped = True

        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True

        if self.sio and self._broadcast_loop and not self._broadcast_loop.is_closed():

            async def _disconnect_all() -> None:
                await self.sio.disconnect()

            asyncio.run_coroutine_threadsafe(_disconnect_all(), self._broadcast_loop)

        if self._broadcast_loop and not self._broadcast_loop.is_closed():
            self._broadcast_loop.call_soon_threadsafe(self._broadcast_loop.stop)

        if self._broadcast_thread and self._broadcast_thread.is_alive():
            self._broadcast_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

        if self._uvicorn_server_thread and self._uvicorn_server_thread.is_alive():
            self._uvicorn_server_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

        super().stop()

    @rpc
    def set_gps_travel_goal_points(self, points: list[LatLon]) -> None:
        json_points = [{"lat": x.lat, "lon": x.lon} for x in points]
        self.vis_state["gps_travel_goal_points"] = json_points
        self._emit("gps_travel_goal_points", json_points)

    def _create_server(self) -> None:
        # Create SocketIO server
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

        async def serve_index(request):  # type: ignore[no-untyped-def]
            """Serve the dashboard HTML at root."""
            return FileResponse(_DASHBOARD_HTML, media_type="text/html")

        async def serve_command_center(request):  # type: ignore[no-untyped-def]
            """Serve the lightweight Sourccey command center."""
            if _COMMAND_CENTER_HTML.exists():
                return FileResponse(_COMMAND_CENTER_HTML, media_type="text/html")
            index_file = get_data("command_center.html")
            if index_file.exists():
                return FileResponse(index_file, media_type="text/html")
            return Response(
                content="Command center not built. Run: cd dimos/web/command-center-extension && npm install && npm run build:standalone",
                status_code=503,
                media_type="text/plain",
            )

        async def api_move(request):  # type: ignore[no-untyped-def]
            data = await request.json()
            if not isinstance(data, dict):
                return JSONResponse({"ok": False, "error": "Expected JSON object"}, status_code=400)
            self._publish_move_command("http", data)
            return JSONResponse({"ok": True})

        async def api_start_explore(request):  # type: ignore[no-untyped-def]
            logger.info("Starting exploration via HTTP command-center")
            self.explore_cmd.publish(Bool(data=True))
            return JSONResponse({"ok": True})

        async def api_stop_explore(request):  # type: ignore[no-untyped-def]
            logger.info("Stopping exploration via HTTP command-center")
            self.stop_explore_cmd.publish(Bool(data=True))
            return JSONResponse({"ok": True})

        async def api_latest_map(request):  # type: ignore[no-untyped-def]
            map_png = _DEFAULT_SOURCCEY_MAP_DIR / "latest_map.png"
            if not map_png.exists():
                return Response(content="Map image not available yet", status_code=404, media_type="text/plain")
            return FileResponse(map_png, media_type="image/png")

        async def api_latest_map_metadata(request):  # type: ignore[no-untyped-def]
            map_json = _DEFAULT_SOURCCEY_MAP_DIR / "latest_map.json"
            if not map_json.exists():
                return JSONResponse({"ok": False, "error": "metadata not available"}, status_code=404)
            try:
                payload = json.loads(map_json.read_text(encoding="utf-8"))
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
            return JSONResponse(payload)

        routes = [
            Route("/", serve_index),
            Route("/command-center", serve_command_center),
            Route("/api/move", api_move, methods=["POST"]),
            Route("/api/start-explore", api_start_explore, methods=["POST"]),
            Route("/api/stop-explore", api_stop_explore, methods=["POST"]),
            Route("/api/latest-map.png", api_latest_map),
            Route("/api/latest-map.json", api_latest_map_metadata),
        ]

        starlette_app = Starlette(routes=routes)

        self.app = socketio.ASGIApp(self.sio, starlette_app)

        # Register SocketIO event handlers
        @self.sio.event  # type: ignore[untyped-decorator]
        async def connect(sid, environ) -> None:  # type: ignore[no-untyped-def]
            with self.state_lock:
                current_state = dict(self.vis_state)

            # Include GPS goal points in the initial state
            if self.gps_goal_points:
                current_state["gps_travel_goal_points"] = self.gps_goal_points

            # Force full costmap update on new connection
            self.costmap_encoder.last_full_grid = None

            await self.sio.emit("full_state", current_state, room=sid)  # type: ignore[union-attr]
            logger.info(
                f"Client {sid} connected, sent state with {len(self.gps_goal_points)} GPS goal points"
            )

        @self.sio.event  # type: ignore[untyped-decorator]
        async def click(sid, position) -> None:  # type: ignore[no-untyped-def]
            goal = PoseStamped(
                position=(position[0], position[1], 0),
                orientation=(0, 0, 0, 1),  # Default orientation
                frame_id="world",
            )
            self.goal_request.publish(goal)
            logger.info(
                "Click goal published", x=round(goal.position.x, 3), y=round(goal.position.y, 3)
            )

        @self.sio.event  # type: ignore[untyped-decorator]
        async def gps_goal(sid: str, goal: dict[str, float]) -> None:
            logger.info(f"Received GPS goal: {goal}")

            # Publish the goal to LCM
            self.gps_goal.publish(LatLon(lat=goal["lat"], lon=goal["lon"]))

            # Add to goal points list for visualization
            self.gps_goal_points.append(goal)
            logger.info(f"Added GPS goal to list. Total goals: {len(self.gps_goal_points)}")

            # Emit updated goal points back to all connected clients
            if self.sio is not None:
                await self.sio.emit("gps_travel_goal_points", self.gps_goal_points)
            logger.debug(
                f"Emitted gps_travel_goal_points with {len(self.gps_goal_points)} points: {self.gps_goal_points}"
            )

        @self.sio.event  # type: ignore[untyped-decorator]
        async def start_explore(sid: str) -> None:
            logger.info("Starting exploration")
            self.explore_cmd.publish(Bool(data=True))

        @self.sio.event  # type: ignore[untyped-decorator]
        async def stop_explore(sid) -> None:  # type: ignore[no-untyped-def]
            logger.info("Stopping exploration")
            self.stop_explore_cmd.publish(Bool(data=True))

        @self.sio.event  # type: ignore[untyped-decorator]
        async def clear_gps_goals(sid: str) -> None:
            logger.info("Clearing all GPS goal points")
            self.gps_goal_points.clear()
            if self.sio is not None:
                await self.sio.emit("gps_travel_goal_points", self.gps_goal_points)
            logger.info("GPS goal points cleared and updated clients")

        @self.sio.event  # type: ignore[untyped-decorator]
        async def move_command(sid: str, data: dict[str, Any]) -> None:
            self._publish_move_command(sid, data)

    def _run_uvicorn_server(self) -> None:
        config = uvicorn.Config(
            self.app,  # type: ignore[arg-type]
            host=self.config.g.listen_host,
            port=self.config.port,
            log_level="error",  # Reduce verbosity
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()

    def _on_robot_pose(self, msg: PoseStamped) -> None:
        pose_data = {"type": "vector", "c": [msg.position.x, msg.position.y, msg.position.z]}
        self.vis_state["robot_pose"] = pose_data
        self._emit("robot_pose", pose_data)

    def _on_gps_location(self, msg: LatLon) -> None:
        pose_data = {"lat": msg.lat, "lon": msg.lon}
        self.vis_state["gps_location"] = pose_data
        self._emit("gps_location", pose_data)

    def _on_path(self, msg: Path) -> None:
        points = [[pose.position.x, pose.position.y] for pose in msg.poses]
        path_data = {"type": "path", "points": points}
        self.vis_state["path"] = path_data
        self._emit("path", path_data)

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        costmap_data = self._process_costmap(msg)
        self.vis_state["costmap"] = costmap_data
        self._emit("costmap", costmap_data)

    def _process_costmap(self, costmap: OccupancyGrid) -> dict[str, Any]:
        """Convert OccupancyGrid to visualization format."""
        grid_data = self.costmap_encoder.encode_costmap(costmap.grid)

        return {
            "type": "costmap",
            "grid": grid_data,
            "origin": {
                "type": "vector",
                "c": [costmap.origin.position.x, costmap.origin.position.y, 0],
            },
            "resolution": costmap.resolution,
            "origin_theta": 0,  # Assuming no rotation for now
        }

    def _emit(self, event: str, data: Any) -> None:
        if self._broadcast_loop and not self._broadcast_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self._broadcast_loop)

    def _publish_move_command(self, sid: str, data: dict[str, Any]) -> None:
        logger.info("Received web move_command", sid=sid, data=data)

        linear = data.get("linear", {})
        angular = data.get("angular", {})
        twist = Twist(
            linear=Vector3(
                float(linear.get("x", 0.0)),
                float(linear.get("y", 0.0)),
                float(linear.get("z", 0.0)),
            ),
            angular=Vector3(
                float(angular.get("x", 0.0)),
                float(angular.get("y", 0.0)),
                float(angular.get("z", 0.0)),
            ),
        )

        if self.tele_cmd_vel and self.tele_cmd_vel.transport:
            logger.info(
                "Publishing tele_cmd_vel",
                linear_x=round(float(twist.linear.x), 4),
                linear_y=round(float(twist.linear.y), 4),
                angular_z=round(float(twist.angular.z), 4),
            )
            self.tele_cmd_vel.publish(twist)
        else:
            logger.warning("tele_cmd_vel transport is unavailable for web move_command")

        if self.movecmd_stamped and self.movecmd_stamped.transport:
            self.movecmd_stamped.publish(
                TwistStamped(
                    ts=time.time(),
                    frame_id="base_link",
                    linear=twist.linear,
                    angular=twist.angular,
                )
            )
        else:
            logger.debug("movecmd_stamped transport is unavailable for web move_command")
