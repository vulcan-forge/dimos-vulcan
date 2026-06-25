from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DEFAULT_EXPORT_ROOT = DIMOS_PROJECT_ROOT / "assets" / "output" / "sourccey_maps"


class SourcceyOccupancyGridExporterConfig(ModuleConfig):
    export_dir: str = Field(default_factory=lambda: str(_DEFAULT_EXPORT_ROOT))
    export_png_name: str = "latest_map.png"
    export_metadata_name: str = "latest_map.json"
    export_interval_s: float = 0.75


class SourcceyOccupancyGridExporter(Module):
    config: SourcceyOccupancyGridExporterConfig

    global_costmap: In[OccupancyGrid]
    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._last_export_ts = 0.0
        self._export_dir = Path(self.config.export_dir)
        self._export_png_path = self._export_dir / self.config.export_png_name
        self._export_metadata_path = self._export_dir / self.config.export_metadata_name

    @rpc
    def start(self) -> None:
        super().start()
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.global_costmap.subscribe(self._on_costmap)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_odom(self, msg: PoseStamped) -> None:
        self._latest_odom = msg

    def _on_costmap(self, grid: OccupancyGrid) -> None:
        if float(grid.ts) - self._last_export_ts < max(float(self.config.export_interval_s), 0.1):
            return
        self._write_export(grid)
        self._last_export_ts = float(grid.ts)

    def _write_export(self, grid: OccupancyGrid) -> None:
        image = np.full((grid.height, grid.width), 127, dtype=np.uint8)
        image[grid.grid == int(CostValues.FREE)] = 255
        image[grid.grid >= int(CostValues.OCCUPIED)] = 0
        image = np.flipud(image)
        cv2.imwrite(str(self._export_png_path), image)

        pose = self._latest_odom
        counts = {
            "occupied": int(np.sum(grid.grid >= int(CostValues.OCCUPIED))),
            "free": int(np.sum(grid.grid == int(CostValues.FREE))),
            "unknown": int(np.sum(grid.grid == int(CostValues.UNKNOWN))),
        }
        metadata = {
            "schema": "sourccey.native_costmap.v1",
            "ts": float(grid.ts or time.time()),
            "frame_id": grid.frame_id,
            "resolution_m": float(grid.resolution),
            "width": int(grid.width),
            "height": int(grid.height),
            "origin": {
                "x": float(grid.origin.position.x),
                "y": float(grid.origin.position.y),
            },
            "counts": counts,
            "pose": None
            if pose is None
            else {
                "x": float(pose.x),
                "y": float(pose.y),
                "z": float(pose.z),
                "yaw": float(pose.yaw),
            },
            "png_path": str(self._export_png_path),
        }
        self._export_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(
            "Sourccey native costmap exported",
            png=str(self._export_png_path),
            metadata=str(self._export_metadata_path),
            occupied=counts["occupied"],
            free=counts["free"],
            unknown=counts["unknown"],
        )
