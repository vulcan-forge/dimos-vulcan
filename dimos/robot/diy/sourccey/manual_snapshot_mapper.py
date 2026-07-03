from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Bool import Bool
from dimos.utils.logging_config import setup_logger

from .lidar_occupancy_mapper import (
    SourcceyLidarOccupancyMapper,
    SourcceyLidarOccupancyMapperConfig,
    _base_pose_from_sensor_xy_yaw,
    _compose_pose_2d,
    _downsample_points,
    _pose_from_xy_yaw,
    _scan_to_local_points,
    _scan_match_metrics,
    _sensor_origin_xy,
    _transform_points_2d,
    _transform_local_points_with_mount,
    _voxelize_points,
)
from .run_trace import trace_event

logger = setup_logger()

_DEFAULT_MANUAL_SNAPSHOT_EXPORT_ROOT = DIMOS_PROJECT_ROOT / "assets" / "output" / "sourccey_manual_snapshots"


def _point_signature(points: np.ndarray) -> dict[str, Any]:
    if points.size == 0:
        return {
            "digest": "empty",
            "point_count": 0,
            "min_xy": [0.0, 0.0],
            "max_xy": [0.0, 0.0],
            "mean_xy": [0.0, 0.0],
            "span_xy": [0.0, 0.0],
            "radial_mean": 0.0,
            "radial_std": 0.0,
            "radial_max": 0.0,
        }

    pts = points.astype(np.float32, copy=False)
    rounded = np.round(pts, 3).astype(np.float32, copy=False)
    min_xy = np.min(rounded, axis=0)
    max_xy = np.max(rounded, axis=0)
    mean_xy = np.mean(rounded, axis=0)
    span_xy = max_xy - min_xy
    radii = np.linalg.norm(rounded, axis=1)
    digest = hashlib.sha1(rounded.tobytes()).hexdigest()[:16]
    return {
        "digest": digest,
        "point_count": int(len(rounded)),
        "min_xy": [round(float(min_xy[0]), 4), round(float(min_xy[1]), 4)],
        "max_xy": [round(float(max_xy[0]), 4), round(float(max_xy[1]), 4)],
        "mean_xy": [round(float(mean_xy[0]), 4), round(float(mean_xy[1]), 4)],
        "span_xy": [round(float(span_xy[0]), 4), round(float(span_xy[1]), 4)],
        "radial_mean": round(float(np.mean(radii)), 4),
        "radial_std": round(float(np.std(radii)), 4),
        "radial_max": round(float(np.max(radii)), 4),
    }


def _snapshot_similarity_summary(
    *,
    reference_local: np.ndarray,
    source_local: np.ndarray,
    overlap_radius_m: float,
) -> dict[str, Any]:
    if reference_local.size == 0 or source_local.size == 0:
        return {
            "reference_digest": _point_signature(reference_local)["digest"],
            "source_digest": _point_signature(source_local)["digest"],
            "best_seed_yaw_deg": None,
            "best_score": None,
            "best_overlap_fraction": 0.0,
        }

    best_seed_yaw_deg: float | None = None
    best_score = float("inf")
    best_overlap = 0.0
    for yaw_seed_deg in (0.0, 90.0, -90.0, 180.0):
        candidate_local = _transform_points_2d(
            source_local,
            x=0.0,
            y=0.0,
            yaw=math.radians(float(yaw_seed_deg)),
        )
        score, overlap = _scan_match_metrics(
            reference_local,
            candidate_local,
            overlap_radius_m=float(overlap_radius_m),
        )
        if score < best_score - 1e-6 or (abs(score - best_score) <= 1e-6 and float(overlap) > best_overlap):
            best_seed_yaw_deg = float(yaw_seed_deg)
            best_score = float(score)
            best_overlap = float(overlap)

    return {
        "reference_digest": _point_signature(reference_local)["digest"],
        "source_digest": _point_signature(source_local)["digest"],
        "best_seed_yaw_deg": None if best_seed_yaw_deg is None else round(float(best_seed_yaw_deg), 2),
        "best_score": None if not math.isfinite(best_score) else round(float(best_score), 5),
        "best_overlap_fraction": round(float(best_overlap), 4),
    }


class SourcceyManualSnapshotMapperConfig(SourcceyLidarOccupancyMapperConfig):
    recent_scans_for_snapshot: int = Field(default=4, ge=1, le=16)
    snapshot_request_debounce_s: float = Field(default=0.35, ge=0.05, le=3.0)
    snapshot_capture_timeout_s: float = Field(default=2.0, ge=0.25, le=10.0)
    save_manual_snapshots: bool = True
    manual_snapshot_export_dir: str = str(_DEFAULT_MANUAL_SNAPSHOT_EXPORT_ROOT)
    capture_only_mode: bool = True
    manual_relative_translation_window_m: float = Field(default=0.60, ge=0.05, le=2.5)
    manual_relative_rotation_window_deg: float = Field(default=135.0, ge=10.0, le=180.0)
    manual_relative_accept_score_m: float = Field(default=0.09, ge=0.005, le=0.5)
    manual_relative_min_overlap_fraction: float = Field(default=0.22, ge=0.01, le=1.0)


class SourcceyManualSnapshotMapper(SourcceyLidarOccupancyMapper):
    config: SourcceyManualSnapshotMapperConfig

    snapshot_request: In[Bool]
    reset_request: In[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._recent_scan_batches: list[np.ndarray] = []
        self._pending_snapshot_scan_batches: list[np.ndarray] = []
        self._pending_snapshot_scan_meta: list[dict[str, Any]] = []
        self._latest_local_points = np.zeros((0, 2), dtype=np.float32)
        self._latest_capture_points = np.zeros((0, 2), dtype=np.float32)
        self._last_scan_ts = 0.0
        self._last_snapshot_request_wall_ts = 0.0
        self._scan_count = 0
        self._snapshot_request_count = 0
        self._last_nonempty_scan_ts = 0.0
        self._last_snapshot_local_points = np.zeros((0, 2), dtype=np.float32)
        self._last_snapshot_pose = None
        self._snapshot_capture_active = False
        self._snapshot_capture_request_index = 0
        self._snapshot_capture_started_wall_ts = 0.0
        self._snapshot_capture_start_scan_count = 0
        self._manual_snapshot_export_dir = Path(self.config.manual_snapshot_export_dir)
        self._manual_snapshot_export_dir.mkdir(parents=True, exist_ok=True)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.snapshot_request.subscribe(self._on_snapshot_request)))
        self.register_disposable(Disposable(self.reset_request.subscribe(self._on_reset_request)))
        self.reset_map()
        trace_event(
            "manual_snapshot_mapper",
            "start",
            capture_only_mode=bool(self.config.capture_only_mode),
            recent_scans_for_snapshot=int(self.config.recent_scans_for_snapshot),
            scan_match_translation_window_m=float(self.config.scan_match_translation_window_m),
            scan_match_rotation_window_deg=float(self.config.scan_match_rotation_window_deg),
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            min_range_m=float(self.config.min_range_m),
            max_distance_m=float(self.config.max_distance_m),
        )
        logger.info(
            "Manual snapshot mapper started capture_only=%s recent_scans=%s forward=%.1f half_width=%.1f",
            bool(self.config.capture_only_mode),
            int(self.config.recent_scans_for_snapshot),
            float(self.config.forward_angle_deg),
            float(self.config.valid_angle_half_width_deg),
        )

    @rpc
    def stop(self) -> None:
        trace_event(
            "manual_snapshot_mapper",
            "stop",
            scan_count=int(self._scan_count),
            snapshot_request_count=int(self._snapshot_request_count),
            buffered_scans=int(len(self._recent_scan_batches)),
        )
        super().stop()

    @rpc
    def reset_map(self) -> None:
        self._recent_scan_batches.clear()
        self._pending_snapshot_scan_batches.clear()
        self._pending_snapshot_scan_meta.clear()
        self._latest_local_points = np.zeros((0, 2), dtype=np.float32)
        self._latest_capture_points = np.zeros((0, 2), dtype=np.float32)
        self._last_scan_ts = 0.0
        self._last_snapshot_local_points = np.zeros((0, 2), dtype=np.float32)
        self._last_snapshot_pose = None
        self._snapshot_capture_active = False
        self._snapshot_capture_request_index = 0
        self._snapshot_capture_started_wall_ts = 0.0
        self._snapshot_capture_start_scan_count = 0
        super().reset_map()
        trace_event(
            "manual_snapshot_mapper",
            "reset_map",
            scan_count=int(self._scan_count),
            snapshot_request_count=int(self._snapshot_request_count),
        )
        logger.info("Manual snapshot mapper reset map and cleared buffered scans")

    def _on_reset_request(self, msg: Bool) -> None:
        trace_event("manual_snapshot_mapper", "reset_request_received", value=bool(msg.data))
        if not bool(msg.data):
            return
        self.reset_map()

    def _on_scan(self, scan) -> None:  # type: ignore[override]
        self._scan_count += 1
        local_points_xy = _scan_to_local_points(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=float(self.config.valid_angle_half_width_deg),
            invert_lateral_axis=bool(self.config.invert_lateral_axis),
            max_distance_m=float(self.config.max_distance_m),
            min_confidence=int(self.config.min_confidence),
            min_range_m=float(self.config.min_range_m),
        )
        capture_points_xy = _scan_to_local_points(
            scan,
            forward_angle_deg=float(self.config.forward_angle_deg),
            valid_angle_half_width_deg=180.0,
            invert_lateral_axis=bool(self.config.invert_lateral_axis),
            max_distance_m=float(self.config.max_distance_m),
            min_confidence=int(self.config.min_confidence),
            min_range_m=float(self.config.min_range_m),
        )
        self._last_scan_ts = float(scan.ts)
        self._latest_local_points = local_points_xy.astype(np.float32, copy=True)
        self._latest_capture_points = capture_points_xy.astype(np.float32, copy=True)
        scan_signature = _point_signature(local_points_xy)
        capture_signature = _point_signature(capture_points_xy)
        trace_event(
            "manual_snapshot_mapper",
            "scan_received",
            scan_count=int(self._scan_count),
            scan_ts=round(float(scan.ts), 5),
            raw_points=int(len(scan.angles_deg)),
            filtered_points=int(len(local_points_xy)),
            capture_points=int(len(capture_points_xy)),
            buffered_scans=int(len(self._recent_scan_batches)),
            scan_digest=scan_signature["digest"],
            scan_span_xy=scan_signature["span_xy"],
            scan_radial_mean=scan_signature["radial_mean"],
        )
        if bool(self.config.capture_only_mode) and capture_points_xy.size == 0:
            trace_event(
                "manual_snapshot_mapper",
                "scan_capture_empty",
                scan_count=int(self._scan_count),
                scan_ts=round(float(scan.ts), 5),
            )
            return

        if local_points_xy.size == 0 and not bool(self.config.capture_only_mode):
            trace_event(
                "manual_snapshot_mapper",
                "scan_filtered_empty",
                scan_count=int(self._scan_count),
                scan_ts=round(float(scan.ts), 5),
            )
            return

        self._last_nonempty_scan_ts = float(scan.ts)
        if not bool(self.config.capture_only_mode):
            self._append_recent_scan_batch(local_points_xy)
        trace_event(
            "manual_snapshot_mapper",
            "scan_buffered",
            scan_count=int(self._scan_count),
            scan_ts=round(float(scan.ts), 5),
            filtered_points=int(len(local_points_xy)),
            buffered_scans=int(len(self._recent_scan_batches)),
            latest_nonempty_scan_age_s=round(max(time.time() - float(self._last_nonempty_scan_ts), 0.0), 4),
        )

        if not self._snapshot_capture_active:
            return

        if (time.time() - float(self._snapshot_capture_started_wall_ts)) > float(self.config.snapshot_capture_timeout_s):
            trace_event(
                "manual_snapshot_mapper",
                "snapshot_capture_timed_out",
                request_index=int(self._snapshot_capture_request_index),
                collected_scans=int(len(self._pending_snapshot_scan_batches)),
                timeout_s=float(self.config.snapshot_capture_timeout_s),
            )
            logger.warning(
                "Manual snapshot request #%s timed out waiting for fresh scans",
                int(self._snapshot_capture_request_index),
            )
            self._clear_snapshot_capture()
            return

        if int(self._scan_count) <= int(self._snapshot_capture_start_scan_count):
            return

        pending_points = capture_points_xy if bool(self.config.capture_only_mode) else local_points_xy
        self._pending_snapshot_scan_batches.append(pending_points.astype(np.float32, copy=True))
        batch_signature = _point_signature(pending_points)
        self._pending_snapshot_scan_meta.append(
            {
                "scan_count": int(self._scan_count),
                "scan_ts": round(float(scan.ts), 5),
                **batch_signature,
            }
        )
        max_batches = max(int(self.config.recent_scans_for_snapshot), 1)
        if len(self._pending_snapshot_scan_batches) > max_batches:
            self._pending_snapshot_scan_batches = self._pending_snapshot_scan_batches[-max_batches:]
            self._pending_snapshot_scan_meta = self._pending_snapshot_scan_meta[-max_batches:]
        trace_event(
            "manual_snapshot_mapper",
            "snapshot_capture_progress",
            request_index=int(self._snapshot_capture_request_index),
            collected_scans=int(len(self._pending_snapshot_scan_batches)),
            required_scans=max_batches,
            scan_count=int(self._scan_count),
            scan_ts=round(float(scan.ts), 5),
            scan_digest=batch_signature["digest"],
            distinct_pending_digests=int(len({meta["digest"] for meta in self._pending_snapshot_scan_meta})),
        )
        if len(self._pending_snapshot_scan_batches) < max_batches:
            return

        request_index = int(self._snapshot_capture_request_index)
        captured_scans = int(len(self._pending_snapshot_scan_batches))
        captured_scan_meta = [dict(meta) for meta in self._pending_snapshot_scan_meta]
        snapshot_points = self._merge_recent_snapshot(self._pending_snapshot_scan_batches)
        self._clear_snapshot_capture()
        self._process_snapshot_points(
            request_index=request_index,
            snapshot_points=snapshot_points,
            scan_ts=float(self._last_scan_ts or time.time()),
            captured_scans=captured_scans,
            captured_scan_meta=captured_scan_meta,
        )

    def _append_recent_scan_batch(self, local_points_xy: np.ndarray) -> None:
        self._recent_scan_batches.append(local_points_xy.astype(np.float32, copy=True))
        max_batches = max(int(self.config.recent_scans_for_snapshot), 1)
        if len(self._recent_scan_batches) > max_batches:
            self._recent_scan_batches = self._recent_scan_batches[-max_batches:]

    def _merge_recent_snapshot(self, batches: list[np.ndarray] | None = None) -> np.ndarray:
        if batches is None:
            batches = self._recent_scan_batches
        trace_event(
            "manual_snapshot_mapper",
            "merge_recent_snapshot_begin",
            buffered_scans=int(len(batches)),
        )
        if not batches:
            trace_event("manual_snapshot_mapper", "merge_recent_snapshot_empty_buffer")
            return np.zeros((0, 2), dtype=np.float32)
        raw_points = int(sum(len(batch) for batch in batches))
        merged = np.vstack(batches).astype(np.float32, copy=False)
        merged = _voxelize_points(merged, float(self.config.stationary_keyframe_voxel_m))
        merged = _downsample_points(merged, max(int(self.config.stationary_keyframe_max_points), 80))
        trace_event(
            "manual_snapshot_mapper",
            "merge_recent_snapshot_end",
            buffered_scans=int(len(batches)),
            raw_points=raw_points,
            merged_points=int(len(merged)),
        )
        return merged.astype(np.float32, copy=False)

    def _arm_snapshot_capture(self, request_index: int) -> None:
        self._snapshot_capture_active = True
        self._snapshot_capture_request_index = int(request_index)
        self._snapshot_capture_started_wall_ts = time.time()
        self._snapshot_capture_start_scan_count = int(self._scan_count)
        self._pending_snapshot_scan_batches.clear()
        self._pending_snapshot_scan_meta.clear()
        trace_event(
            "manual_snapshot_mapper",
            "snapshot_capture_armed",
            request_index=int(request_index),
            start_scan_count=int(self._snapshot_capture_start_scan_count),
            required_scans=int(self.config.recent_scans_for_snapshot),
            timeout_s=float(self.config.snapshot_capture_timeout_s),
        )
        logger.info(
            "Manual snapshot request #%s armed; waiting for %s fresh scans after scan_count=%s",
            int(request_index),
            int(self.config.recent_scans_for_snapshot),
            int(self._snapshot_capture_start_scan_count),
        )

    def _clear_snapshot_capture(self) -> None:
        self._snapshot_capture_active = False
        self._snapshot_capture_request_index = 0
        self._snapshot_capture_started_wall_ts = 0.0
        self._snapshot_capture_start_scan_count = 0
        self._pending_snapshot_scan_batches.clear()
        self._pending_snapshot_scan_meta.clear()

    def _snapshot_artifact_paths(self, request_index: int) -> dict[str, Path]:
        stem = f"snapshot_{int(request_index):03d}"
        return {
            "local_npy": self._manual_snapshot_export_dir / f"{stem}_local.npy",
            "world_npy": self._manual_snapshot_export_dir / f"{stem}_world.npy",
            "metadata_json": self._manual_snapshot_export_dir / f"{stem}.json",
            "preview_png": self._manual_snapshot_export_dir / f"{stem}_preview.png",
        }

    def _render_snapshot_preview(
        self,
        *,
        local_points: np.ndarray,
        world_points: np.ndarray,
        request_index: int,
    ) -> np.ndarray:
        canvas_size = 900
        image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        image[:] = (10, 10, 10)
        pad = 0.35
        all_points: list[np.ndarray] = []
        if local_points.size != 0:
            all_points.append(local_points.astype(np.float32, copy=False))
        if world_points.size != 0:
            all_points.append(world_points.astype(np.float32, copy=False))
        all_points.append(np.asarray([[0.0, 0.0]], dtype=np.float32))
        merged = np.vstack(all_points)
        min_xy = np.min(merged, axis=0)
        max_xy = np.max(merged, axis=0)
        center = (min_xy + max_xy) * 0.5
        span = np.max(max_xy - min_xy)
        span = max(float(span), 1.0)
        half = (span * 0.5) + pad

        def world_to_px(points_xy: np.ndarray) -> np.ndarray:
            if points_xy.size == 0:
                return np.zeros((0, 2), dtype=np.int32)
            normalized = (points_xy - center) / (2.0 * half)
            px = ((normalized[:, 0] + 0.5) * (canvas_size - 1)).astype(np.int32)
            py = ((0.5 - normalized[:, 1]) * (canvas_size - 1)).astype(np.int32)
            return np.column_stack((px, py)).astype(np.int32, copy=False)

        local_px = world_to_px(local_points)
        world_px = world_to_px(world_points)
        origin_px = world_to_px(np.asarray([[0.0, 0.0]], dtype=np.float32))

        if len(local_px):
            for x_px, y_px in local_px.tolist():
                cv2.circle(image, (int(x_px), int(y_px)), 2, (255, 210, 110), -1)
        if len(world_px):
            for x_px, y_px in world_px.tolist():
                cv2.circle(image, (int(x_px), int(y_px)), 2, (110, 190, 255), -1)
        if len(origin_px):
            x0, y0 = origin_px[0].tolist()
            cv2.circle(image, (int(x0), int(y0)), 6, (255, 255, 255), 2)
            cv2.line(image, (int(x0) - 16, int(y0)), (int(x0) + 16, int(y0)), (80, 255, 255), 1)
            cv2.line(image, (int(x0), int(y0) - 16), (int(x0), int(y0) + 16), (80, 255, 255), 1)

        cv2.putText(
            image,
            f"snapshot {int(request_index)}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "orange=local  blue=integrated",
            (24, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return image

    def _save_manual_snapshot_artifacts(
        self,
        *,
        request_index: int,
        scan_ts: float,
        local_points: np.ndarray,
        pose: Any,
        score: float | None,
        overlap: float,
        captured_scan_meta: list[dict[str, Any]] | None = None,
    ) -> None:
        if not bool(self.config.save_manual_snapshots):
            return
        try:
            self._manual_snapshot_export_dir.mkdir(parents=True, exist_ok=True)
            world_points = _transform_local_points_with_mount(
                local_points,
                x=float(pose.x),
                y=float(pose.y),
                yaw=float(pose.yaw),
                mount_x_m=float(self.config.lidar_mount_x_m),
                mount_y_m=float(self.config.lidar_mount_y_m),
            ).astype(np.float32, copy=False)
            paths = self._snapshot_artifact_paths(int(request_index))
            np.save(paths["local_npy"], local_points.astype(np.float32, copy=False))
            np.save(paths["world_npy"], world_points)
            preview = self._render_snapshot_preview(
                local_points=local_points.astype(np.float32, copy=False),
                world_points=world_points,
                request_index=int(request_index),
            )
            cv2.imwrite(str(paths["preview_png"]), preview)
            metadata = {
                "schema": "sourccey.manual_snapshot.v1",
                "request_index": int(request_index),
                "scan_ts": float(scan_ts),
                "local_point_count": int(len(local_points)),
                "world_point_count": int(len(world_points)),
                "local_signature": _point_signature(local_points),
                "capture_batches": list(captured_scan_meta or []),
                "pose": {
                    "x": float(pose.x),
                    "y": float(pose.y),
                    "z": float(pose.z),
                    "yaw": float(pose.yaw),
                },
                "score": None if score is None else float(score),
                "overlap_fraction": float(overlap),
                "local_npy_path": str(paths["local_npy"]),
                "world_npy_path": str(paths["world_npy"]),
                "preview_png_path": str(paths["preview_png"]),
            }
            paths["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            trace_event(
                "manual_snapshot_mapper",
                "manual_snapshot_artifacts_saved",
                request_index=int(request_index),
                local_points=int(len(local_points)),
                world_points=int(len(world_points)),
                metadata_path=str(paths["metadata_json"]),
                preview_path=str(paths["preview_png"]),
            )
            logger.info(
                "Saved manual snapshot #%s artifacts metadata=%s preview=%s",
                int(request_index),
                paths["metadata_json"],
                paths["preview_png"],
            )
        except Exception as exc:
            trace_event(
                "manual_snapshot_mapper",
                "manual_snapshot_artifacts_save_failed",
                request_index=int(request_index),
                error=str(exc),
            )
            logger.warning("Failed to save manual snapshot #%s artifacts: %s", int(request_index), exc)

    def _manual_reference_world(self) -> np.ndarray:
        occupied = self._grid_points(occupied=True)
        if occupied.size != 0:
            trace_event(
                "manual_snapshot_mapper",
                "reference_world_selected",
                source="occupied_grid",
                points=int(len(occupied)),
            )
            return _voxelize_points(occupied, float(self.config.stationary_keyframe_voxel_m))
        if self._submap_points_world.size != 0:
            trace_event(
                "manual_snapshot_mapper",
                "reference_world_selected",
                source="submap_points",
                points=int(len(self._submap_points_world)),
            )
            return self._submap_points_world.astype(np.float32, copy=False)
        if self._last_matched_pose is not None and self._last_local_points.size != 0:
            trace_event(
                "manual_snapshot_mapper",
                "reference_world_selected",
                source="last_matched_pose",
                points=int(len(self._last_local_points)),
            )
            return _transform_local_points_with_mount(
                self._last_local_points,
                x=float(self._last_matched_pose.x),
                y=float(self._last_matched_pose.y),
                yaw=float(self._last_matched_pose.yaw),
                mount_x_m=float(self.config.lidar_mount_x_m),
                mount_y_m=float(self.config.lidar_mount_y_m),
            )
        trace_event("manual_snapshot_mapper", "reference_world_selected", source="empty", points=0)
        return np.zeros((0, 2), dtype=np.float32)

    def _manual_seed_poses(self, *, ts: float) -> list[tuple[str, Any]]:
        seed_specs: list[tuple[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()
        base_candidates = [
            ("current", self._current_pose()),
            ("last", self._last_matched_pose),
        ]
        yaw_offsets_deg = (0.0, 90.0, -90.0, 180.0)
        for base_label, base_pose in base_candidates:
            if base_pose is None:
                continue
            for yaw_offset_deg in yaw_offsets_deg:
                yaw = float(base_pose.yaw) + math.radians(float(yaw_offset_deg))
                pose = _pose_from_xy_yaw(
                    x=float(base_pose.x),
                    y=float(base_pose.y),
                    yaw=yaw,
                    ts=float(ts),
                    frame_id=self.config.frame_id,
                )
                key = (
                    int(round(float(pose.x) * 100)),
                    int(round(float(pose.y) * 100)),
                    int(round(math.degrees(float(pose.yaw)))),
                )
                if key in seen:
                    continue
                seen.add(key)
                seed_specs.append((f"{base_label}_{int(yaw_offset_deg)}deg", pose))
        if not seed_specs:
            seed_specs.append(
                (
                    "origin_0deg",
                    _pose_from_xy_yaw(
                        x=0.0,
                        y=0.0,
                        yaw=0.0,
                        ts=float(ts),
                        frame_id=self.config.frame_id,
                    ),
                )
            )
        trace_event(
            "manual_snapshot_mapper",
            "manual_seed_poses_built",
            seed_count=int(len(seed_specs)),
            seeds=[label for label, _ in seed_specs],
        )
        return seed_specs

    def _resolve_relative_snapshot_pose(
        self,
        *,
        ts: float,
        local_points: np.ndarray,
    ) -> tuple[Any | None, float | None, float]:
        if self._last_snapshot_pose is None or self._last_snapshot_local_points.size == 0:
            trace_event("manual_snapshot_mapper", "relative_snapshot_pose_skipped", reason="no_previous_snapshot")
            return None, None, 0.0

        reference_local = _downsample_points(
            self._last_snapshot_local_points.astype(np.float32, copy=False),
            max(int(self.config.scan_match_max_points), 40),
        )
        source_local = _downsample_points(
            local_points.astype(np.float32, copy=False),
            max(int(self.config.scan_match_max_points), 40),
        )
        if reference_local.size == 0 or source_local.size == 0:
            trace_event("manual_snapshot_mapper", "relative_snapshot_pose_skipped", reason="empty_local_points")
            return None, None, 0.0

        translation_window = max(float(self.config.manual_relative_translation_window_m), 0.05)
        rotation_window = math.radians(max(float(self.config.manual_relative_rotation_window_deg), 2.0))
        search_levels = (
            (translation_window, rotation_window, 5),
            (translation_window * 0.4, rotation_window * 0.35, 4),
            (translation_window * 0.15, rotation_window * 0.15, 3),
        )
        yaw_seeds_deg = (0.0, 90.0, -90.0, 180.0)
        best_rel_x = 0.0
        best_rel_y = 0.0
        best_rel_yaw = 0.0
        best_score = float("inf")
        best_overlap = 0.0
        best_seed = 0.0

        trace_event(
            "manual_snapshot_mapper",
            "relative_snapshot_pose_begin",
            local_points=int(len(source_local)),
            reference_points=int(len(reference_local)),
            yaw_seeds_deg=list(yaw_seeds_deg),
        )

        for yaw_seed_deg in yaw_seeds_deg:
            seed_yaw = math.radians(float(yaw_seed_deg))
            seed_candidate = _transform_points_2d(source_local, x=0.0, y=0.0, yaw=seed_yaw)
            seed_score, seed_overlap = _scan_match_metrics(
                reference_local,
                seed_candidate,
                overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
            )
            cand_best_x = 0.0
            cand_best_y = 0.0
            cand_best_yaw = seed_yaw
            cand_best_score = float(seed_score)
            cand_best_overlap = float(seed_overlap)

            for trans_window, yaw_window, n_steps in search_levels:
                offsets = range(-n_steps, n_steps + 1)
                trans_step = float(trans_window) / max(int(n_steps), 1)
                yaw_step = float(yaw_window) / max(int(n_steps), 1)
                for dx_idx in offsets:
                    for dy_idx in offsets:
                        for dyaw_idx in offsets:
                            cand_x = cand_best_x + dx_idx * trans_step
                            cand_y = cand_best_y + dy_idx * trans_step
                            cand_yaw = cand_best_yaw + dyaw_idx * yaw_step
                            candidate_local = _transform_points_2d(source_local, x=cand_x, y=cand_y, yaw=cand_yaw)
                            score, overlap = _scan_match_metrics(
                                reference_local,
                                candidate_local,
                                overlap_radius_m=float(self.config.scan_match_overlap_radius_m),
                            )
                            if score < cand_best_score - 1e-6 or (
                                abs(score - cand_best_score) <= 1e-6 and float(overlap) > cand_best_overlap
                            ):
                                cand_best_x = float(cand_x)
                                cand_best_y = float(cand_y)
                                cand_best_yaw = float(cand_yaw)
                                cand_best_score = float(score)
                                cand_best_overlap = float(overlap)

            trace_event(
                "manual_snapshot_mapper",
                "relative_snapshot_seed_attempt",
                seed_yaw_deg=round(float(yaw_seed_deg), 2),
                rel_x=round(float(cand_best_x), 4),
                rel_y=round(float(cand_best_y), 4),
                rel_yaw_deg=round(math.degrees(float(cand_best_yaw)), 2),
                score=round(float(cand_best_score), 5),
                overlap_fraction=round(float(cand_best_overlap), 4),
            )

            if cand_best_score < best_score - 1e-6 or (
                abs(cand_best_score - best_score) <= 1e-6 and cand_best_overlap > best_overlap
            ):
                best_rel_x = float(cand_best_x)
                best_rel_y = float(cand_best_y)
                best_rel_yaw = float(cand_best_yaw)
                best_score = float(cand_best_score)
                best_overlap = float(cand_best_overlap)
                best_seed = float(yaw_seed_deg)

        trace_event(
            "manual_snapshot_mapper",
            "relative_snapshot_pose_best_candidate",
            seed_yaw_deg=round(float(best_seed), 2),
            rel_x=round(float(best_rel_x), 4),
            rel_y=round(float(best_rel_y), 4),
            rel_yaw_deg=round(math.degrees(float(best_rel_yaw)), 2),
            score=round(float(best_score), 5),
            overlap_fraction=round(float(best_overlap), 4),
        )

        if best_score > float(self.config.manual_relative_accept_score_m) or best_overlap < float(
            self.config.manual_relative_min_overlap_fraction
        ):
            trace_event(
                "manual_snapshot_mapper",
                "relative_snapshot_pose_rejected",
                score=round(float(best_score), 5),
                overlap_fraction=round(float(best_overlap), 4),
                accept_score=float(self.config.manual_relative_accept_score_m),
                min_overlap_fraction=float(self.config.manual_relative_min_overlap_fraction),
            )
            return None, best_score, best_overlap

        prev_sensor_x, prev_sensor_y = _sensor_origin_xy(
            x=float(self._last_snapshot_pose.x),
            y=float(self._last_snapshot_pose.y),
            yaw=float(self._last_snapshot_pose.yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        sensor_x, sensor_y, sensor_yaw = _compose_pose_2d(
            origin_x=float(prev_sensor_x),
            origin_y=float(prev_sensor_y),
            origin_yaw=float(self._last_snapshot_pose.yaw),
            rel_x=float(best_rel_x),
            rel_y=float(best_rel_y),
            rel_yaw=float(best_rel_yaw),
        )
        base_x, base_y = _base_pose_from_sensor_xy_yaw(
            sensor_x=float(sensor_x),
            sensor_y=float(sensor_y),
            yaw=float(sensor_yaw),
            mount_x_m=float(self.config.lidar_mount_x_m),
            mount_y_m=float(self.config.lidar_mount_y_m),
        )
        pose = _pose_from_xy_yaw(
            x=float(base_x),
            y=float(base_y),
            yaw=float(sensor_yaw),
            ts=float(ts),
            frame_id=self.config.frame_id,
        )
        trace_event(
            "manual_snapshot_mapper",
            "relative_snapshot_pose_resolved",
            pose_x=round(float(pose.x), 4),
            pose_y=round(float(pose.y), 4),
            pose_yaw_deg=round(math.degrees(float(pose.yaw)), 2),
            score=round(float(best_score), 5),
            overlap_fraction=round(float(best_overlap), 4),
        )
        return pose, best_score, best_overlap

    def _resolve_manual_snapshot_pose(
        self,
        *,
        ts: float,
        local_points: np.ndarray,
    ) -> tuple[Any | None, float | None, float]:
        trace_event(
            "manual_snapshot_mapper",
            "resolve_snapshot_pose_begin",
            local_points=int(len(local_points)),
            has_last_pose=bool(self._last_matched_pose is not None),
        )
        if self._last_matched_pose is None:
            pose = _pose_from_xy_yaw(
                x=0.0,
                y=0.0,
                yaw=0.0,
                ts=float(ts),
                frame_id=self.config.frame_id,
            )
            trace_event("manual_snapshot_mapper", "bootstrap_snapshot_pose", pose_x=0.0, pose_y=0.0, pose_yaw_deg=0.0)
            return pose, None, 1.0

        reference_world = self._manual_reference_world()
        if reference_world.size == 0:
            trace_event("manual_snapshot_mapper", "manual_snapshot_no_reference")
            relative_pose, relative_score, relative_overlap = self._resolve_relative_snapshot_pose(
                ts=float(ts),
                local_points=local_points,
            )
            if relative_pose is not None:
                trace_event(
                    "manual_snapshot_mapper",
                    "manual_snapshot_pose_resolved_via_relative",
                    pose_x=round(float(relative_pose.x), 4),
                    pose_y=round(float(relative_pose.y), 4),
                    pose_yaw_deg=round(math.degrees(float(relative_pose.yaw)), 2),
                    score=None if relative_score is None else round(float(relative_score), 5),
                    overlap_fraction=round(float(relative_overlap), 4),
                )
                return relative_pose, relative_score, relative_overlap
            return None, None, 0.0

        best_pose = None
        best_score = float("inf")
        best_overlap = 0.0
        best_label = ""
        rejected_best_score = float("inf")
        rejected_best_overlap = 0.0
        rejected_best_label = ""

        for seed_label, seed_pose in self._manual_seed_poses(ts=float(ts)):
            candidate_pose, score, overlap = self._refine_pose_with_scan_match(
                ts=float(ts),
                seed_pose=seed_pose,
                current_local_points=local_points,
                reference_world=reference_world,
            )
            trace_event(
                "manual_snapshot_mapper",
                "manual_snapshot_seed_attempt",
                seed=seed_label,
                seed_x=round(float(seed_pose.x), 4),
                seed_y=round(float(seed_pose.y), 4),
                seed_yaw_deg=round(math.degrees(float(seed_pose.yaw)), 2),
                accepted=candidate_pose is not None,
                score=None if score is None else round(float(score), 5),
                overlap_fraction=round(float(overlap), 4),
            )
            if candidate_pose is not None:
                score_value = float(score if score is not None else 0.0)
                if score_value < best_score - 1e-6 or (
                    abs(score_value - best_score) <= 1e-6 and float(overlap) > best_overlap
                ):
                    best_pose = candidate_pose
                    best_score = score_value
                    best_overlap = float(overlap)
                    best_label = seed_label
                continue
            score_value = float(score if score is not None else float("inf"))
            if score_value < rejected_best_score - 1e-6 or (
                abs(score_value - rejected_best_score) <= 1e-6 and float(overlap) > rejected_best_overlap
            ):
                rejected_best_score = score_value
                rejected_best_overlap = float(overlap)
                rejected_best_label = seed_label

        if best_pose is None:
            relative_pose, relative_score, relative_overlap = self._resolve_relative_snapshot_pose(
                ts=float(ts),
                local_points=local_points,
            )
            trace_event(
                "manual_snapshot_mapper",
                "manual_snapshot_pose_rejected",
                best_seed=rejected_best_label,
                best_score=None if not math.isfinite(rejected_best_score) else round(float(rejected_best_score), 5),
                best_overlap_fraction=round(float(rejected_best_overlap), 4),
                relative_fallback_pose_found=bool(relative_pose is not None),
                relative_fallback_score=None if relative_score is None else round(float(relative_score), 5),
                relative_fallback_overlap_fraction=round(float(relative_overlap), 4),
                relative_fallback_applied=False,
            )
            return None, None if not math.isfinite(rejected_best_score) else rejected_best_score, rejected_best_overlap

        trace_event(
            "manual_snapshot_mapper",
            "manual_snapshot_pose_resolved",
            seed=best_label,
            pose_x=round(float(best_pose.x), 4),
            pose_y=round(float(best_pose.y), 4),
            pose_yaw_deg=round(math.degrees(float(best_pose.yaw)), 2),
            score=round(float(best_score), 5),
            overlap_fraction=round(float(best_overlap), 4),
        )
        return best_pose, best_score, best_overlap

    def _process_snapshot_points(
        self,
        *,
        request_index: int,
        snapshot_points: np.ndarray,
        scan_ts: float,
        captured_scans: int,
        captured_scan_meta: list[dict[str, Any]],
    ) -> None:
        snapshot_signature = _point_signature(snapshot_points)
        trace_event(
            "manual_snapshot_mapper",
            "snapshot_request_received",
            request_index=int(request_index),
            buffered_scans=int(captured_scans),
            local_points=int(len(snapshot_points)),
            last_scan_ts=round(float(scan_ts), 4),
            scan_count=int(self._scan_count),
            snapshot_digest=snapshot_signature["digest"],
            snapshot_span_xy=snapshot_signature["span_xy"],
            capture_batch_digests=[meta.get("digest") for meta in captured_scan_meta],
        )
        logger.info(
            "Manual snapshot request #%s received fresh_scans=%s merged_points=%s scan_ts=%.4f",
            int(request_index),
            int(captured_scans),
            int(len(snapshot_points)),
            float(scan_ts),
        )
        if snapshot_points.size == 0:
            trace_event("manual_snapshot_mapper", "snapshot_request_empty_buffer")
            logger.warning("Manual snapshot request ignored: no fresh LiDAR points captured yet.")
            return

        if bool(self.config.capture_only_mode):
            pose = _pose_from_xy_yaw(
                x=0.0,
                y=0.0,
                yaw=0.0,
                ts=float(scan_ts or time.time()),
                frame_id=self.config.frame_id,
            )
            trace_event(
                "manual_snapshot_mapper",
                "snapshot_request_capture_only",
                request_index=int(request_index),
                local_points=int(len(snapshot_points)),
                snapshot_digest=snapshot_signature["digest"],
                captured_scans=int(captured_scans),
            )
            self._save_manual_snapshot_artifacts(
                request_index=int(request_index),
                scan_ts=float(scan_ts or time.time()),
                local_points=snapshot_points.astype(np.float32, copy=False),
                pose=pose,
                score=None,
                overlap=0.0,
                captured_scan_meta=captured_scan_meta,
            )
            self._last_snapshot_local_points = snapshot_points.astype(np.float32, copy=True)
            self._last_snapshot_pose = pose
            self._recent_scan_batches.clear()
            trace_event(
                "manual_snapshot_mapper",
                "snapshot_request_complete",
                request_index=int(request_index),
                buffered_scans=int(len(self._recent_scan_batches)),
                mode="capture_only",
            )
            logger.info(
                "SNAPSHOT #%s CAPTURE COMPLETE: saved single-scan artifact points=%s digest=%s. Safe to move the robot now.",
                int(request_index),
                int(len(snapshot_points)),
                snapshot_signature["digest"],
            )
            return

    def _on_snapshot_request(self, msg: Bool) -> None:
        self._snapshot_request_count += 1
        trace_event(
            "manual_snapshot_mapper",
            "snapshot_request_signal_received",
            request_index=int(self._snapshot_request_count),
            value=bool(msg.data),
            buffered_scans=int(len(self._recent_scan_batches)),
            last_scan_ts=round(float(self._last_scan_ts), 5),
            last_nonempty_scan_ts=round(float(self._last_nonempty_scan_ts), 5),
        )
        if not bool(msg.data):
            return
        now = time.time()
        if (now - float(self._last_snapshot_request_wall_ts)) < float(self.config.snapshot_request_debounce_s):
            trace_event(
                "manual_snapshot_mapper",
                "snapshot_request_debounced",
                request_index=int(self._snapshot_request_count),
                debounce_s=round(now - float(self._last_snapshot_request_wall_ts), 5),
            )
            return
        self._last_snapshot_request_wall_ts = now
        if self._snapshot_capture_active:
            trace_event(
                "manual_snapshot_mapper",
                "snapshot_request_replaced_pending_capture",
                old_request_index=int(self._snapshot_capture_request_index),
                new_request_index=int(self._snapshot_request_count),
                collected_scans=int(len(self._pending_snapshot_scan_batches)),
            )
            logger.info(
                "Manual snapshot request #%s replaced pending request #%s",
                int(self._snapshot_request_count),
                int(self._snapshot_capture_request_index),
            )
            self._clear_snapshot_capture()

        if bool(self.config.capture_only_mode):
            if self._latest_capture_points.size == 0:
                trace_event("manual_snapshot_mapper", "snapshot_request_empty_buffer")
                logger.warning("Manual snapshot request ignored: no raw LiDAR points buffered yet.")
                return
        else:
            if self._latest_local_points.size == 0:
                trace_event("manual_snapshot_mapper", "snapshot_request_empty_buffer")
                logger.warning("Manual snapshot request ignored: no LiDAR points buffered yet.")
                return

        if bool(self.config.capture_only_mode):
            trace_event(
                "manual_snapshot_mapper",
                "snapshot_capture_only_begin",
                request_index=int(self._snapshot_request_count),
                scan_count=int(self._scan_count),
                scan_ts=round(float(self._last_scan_ts), 5),
                mode="single_fresh_raw_scan_after_request",
            )
            logger.info(
                "SNAPSHOT #%s CAPTURE STARTED: waiting for the next fresh RAW LiDAR frame after this request. Do not move the robot until completion is logged.",
                int(self._snapshot_request_count),
            )
            self._arm_snapshot_capture(int(self._snapshot_request_count))
            return



