from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_OUTPUT_DIR = DIMOS_PROJECT_ROOT / "assets" / "output"
_MEMORY_DIR = _OUTPUT_DIR / "memory"
_SPATIAL_MEMORY_DIR = _MEMORY_DIR / "spatial_memory"
_DEFAULT_SESSION_ROOT = _SPATIAL_MEMORY_DIR / "sessions"
_DEFAULT_RECON_ROOT = _SPATIAL_MEMORY_DIR / "reconstruction"
_LATEST_SESSION_PATH = _SPATIAL_MEMORY_DIR / "latest_session.json"
_DEFAULT_PRIMARY_FOV_DEG = 78.0
_DEFAULT_BOTTOM_FOV_DEG = 110.0


def _ensure_uint8_bgr(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pose_to_dict(pose: Any | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    orientation = getattr(pose, "orientation", None)
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(orientation.x),
            "y": float(orientation.y),
            "z": float(orientation.z),
            "w": float(orientation.w),
        },
        "frame_id": str(getattr(pose, "frame_id", "")),
        "ts": float(getattr(pose, "ts", 0.0)),
    }


def _camera_info_to_dict(camera_info: CameraInfo | None) -> dict[str, Any] | None:
    if camera_info is None:
        return None
    return {
        "frame_id": camera_info.frame_id,
        "width": int(camera_info.width),
        "height": int(camera_info.height),
        "K": [float(x) for x in camera_info.K],
        "D": [float(x) for x in camera_info.D],
        "R": [float(x) for x in camera_info.R],
        "P": [float(x) for x in camera_info.P],
        "distortion_model": camera_info.distortion_model,
        "ts": float(camera_info.ts),
    }


def _quaternion_to_matrix(quaternion_dict: dict[str, float]) -> np.ndarray:
    quat = Quaternion(
        quaternion_dict["x"],
        quaternion_dict["y"],
        quaternion_dict["z"],
        quaternion_dict["w"],
    )
    return quat.to_rotation_matrix()


def _pose_dict_to_matrix(pose_dict: dict[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_to_matrix(pose_dict["orientation"])
    matrix[:3, 3] = np.array(
        [
            pose_dict["position"]["x"],
            pose_dict["position"]["y"],
            pose_dict["position"]["z"],
        ],
        dtype=np.float64,
    )
    return matrix


def _intrinsics_from_fov(width: float, height: float, fov_deg: float, axis: str = "horizontal") -> np.ndarray:
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    fov_rad = math.radians(max(1.0, float(fov_deg)))
    if axis == "vertical":
        fy = height / (2.0 * math.tan(fov_rad / 2.0))
        fx = fy
    else:
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _projection_from_pose_and_intrinsics(pose_dict: dict[str, Any], intrinsics: np.ndarray) -> np.ndarray:
    transform_wc = _pose_dict_to_matrix(pose_dict)
    transform_cw = np.linalg.inv(transform_wc)
    return intrinsics @ transform_cw[:3, :]


def _camera_depths(points_world: np.ndarray, pose_dict: dict[str, Any]) -> np.ndarray:
    transform_wc = _pose_dict_to_matrix(pose_dict)
    transform_cw = np.linalg.inv(transform_wc)
    points_h = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_camera = (transform_cw @ points_h.T).T
    return points_camera[:, 2]


def _rotation_delta_deg(pose_a: dict[str, Any], pose_b: dict[str, Any]) -> float:
    qa = Quaternion(
        pose_a["orientation"]["x"],
        pose_a["orientation"]["y"],
        pose_a["orientation"]["z"],
        pose_a["orientation"]["w"],
    )
    qb = Quaternion(
        pose_b["orientation"]["x"],
        pose_b["orientation"]["y"],
        pose_b["orientation"]["z"],
        pose_b["orientation"]["w"],
    )
    return math.degrees(qa.angle_to(qb))


def _translation_delta_m(pose_a: dict[str, Any], pose_b: dict[str, Any]) -> float:
    a = np.array([pose_a["position"][axis] for axis in ("x", "y", "z")], dtype=np.float64)
    b = np.array([pose_b["position"][axis] for axis in ("x", "y", "z")], dtype=np.float64)
    return float(np.linalg.norm(b - a))


def _load_manifest(session: str | Path) -> tuple[Path, dict[str, Any]]:
    session_path = Path(session)
    if str(session) == "latest":
        if _LATEST_SESSION_PATH.exists():
            latest_payload = json.loads(_LATEST_SESSION_PATH.read_text(encoding="utf-8"))
            session_path = Path(latest_payload["manifest_path"])
        else:
            candidates = sorted(
                _DEFAULT_SESSION_ROOT.glob("session_*/manifest.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"Latest spatial session pointer not found at {_LATEST_SESSION_PATH} and no saved sessions exist under {_DEFAULT_SESSION_ROOT}. Run sourccey-spatial-scan first."
                )
            session_path = candidates[0]
            logger.warning(
                "Latest spatial session pointer was missing; falling back to newest manifest at %s",
                session_path,
            )
    if session_path.is_dir():
        session_path = session_path / "manifest.json"
    if not session_path.exists():
        raise FileNotFoundError(f"Spatial session manifest not found at {session_path}")
    manifest = json.loads(session_path.read_text(encoding="utf-8"))
    return session_path, manifest


def _latest_reconstruction_outputs(output_dir: Path) -> dict[str, Path]:
    return {
        "npz": output_dir / "latest_cloud.npz",
        "ply": output_dir / "latest_cloud.ply",
        "html": output_dir / "latest_view.html",
        "summary": output_dir / "latest_summary.json",
    }


class SpatialSessionWriter:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else _DEFAULT_SESSION_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir: Path | None = None
        self.manifest_path: Path | None = None
        self.records: list[dict[str, Any]] = []

    def start_session(self) -> Path:
        if self.session_dir is not None:
            return self.session_dir
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.root_dir / session_name
        for subdir in ("primary", "companion", "bottom", "preview"):
            (self.session_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_dir / "manifest.json"
        self._write_manifest()
        return self.session_dir

    def _save_frame(self, subdir: str, stem: str, frame: np.ndarray | None) -> str | None:
        frame = _ensure_uint8_bgr(frame)
        if frame is None:
            return None
        assert self.session_dir is not None
        relative_path = Path(subdir) / f"{stem}.jpg"
        output_path = self.session_dir / relative_path
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write frame to {output_path}")
        return relative_path.as_posix()

    def add_frame(
        self,
        *,
        primary_frame: np.ndarray,
        companion_frame: np.ndarray | None,
        bottom_frame: np.ndarray | None,
        preview_frame: np.ndarray | None,
        frame_id: str,
        timestamp: float,
        base_pose: Any | None,
        primary_pose: Any | None,
        companion_pose: Any | None,
        bottom_pose: Any | None,
        primary_camera_info: CameraInfo | None,
    ) -> Path:
        session_dir = self.start_session()
        stem = f"{len(self.records):05d}_{frame_id}"
        record = {
            "frame_index": len(self.records),
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "primary_image": self._save_frame("primary", stem, primary_frame),
            "companion_image": self._save_frame("companion", stem, companion_frame),
            "bottom_image": self._save_frame("bottom", stem, bottom_frame),
            "preview_image": self._save_frame("preview", stem, preview_frame),
            "base_pose": _pose_to_dict(base_pose),
            "primary_pose": _pose_to_dict(primary_pose),
            "companion_pose": _pose_to_dict(companion_pose),
            "bottom_pose": _pose_to_dict(bottom_pose),
            "primary_camera_info": _camera_info_to_dict(primary_camera_info),
        }
        self.records.append(record)
        self._write_manifest()
        logger.info(
            "Spatial session stored frame=%s session=%s total=%d",
            frame_id,
            session_dir,
            len(self.records),
        )
        return session_dir

    def _write_manifest(self) -> None:
        if self.session_dir is None or self.manifest_path is None:
            return
        manifest = {
            "schema": "sourccey.spatial_session.v1",
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "session_dir": str(self.session_dir),
            "frame_count": len(self.records),
            "frames": self.records,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _LATEST_SESSION_PATH.write_text(
            json.dumps(
                {
                    "session_dir": str(self.session_dir),
                    "manifest_path": str(self.manifest_path),
                    "frame_count": len(self.records),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def finalize(self) -> Path | None:
        self._write_manifest()
        return self.manifest_path


def export_session(session: str | Path = "latest") -> dict[str, Any]:
    manifest_path, manifest = _load_manifest(session)
    frames = manifest.get("frames", [])
    export_payload = {
        "session_dir": manifest.get("session_dir"),
        "manifest_path": str(manifest_path),
        "frame_count": int(manifest.get("frame_count", len(frames))),
        "primary_frames": sum(1 for frame in frames if frame.get("primary_image")),
        "companion_frames": sum(1 for frame in frames if frame.get("companion_image")),
        "bottom_frames": sum(1 for frame in frames if frame.get("bottom_image")),
        "latest_frame_id": frames[-1].get("frame_id") if frames else None,
    }
    export_path = Path(manifest.get("session_dir", manifest_path.parent)) / "session_export.json"
    export_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    export_payload["export_path"] = str(export_path)
    return export_payload


def _create_feature_pipeline() -> tuple[Any, Any, str]:
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        return detector, matcher, "sift"
    detector = cv2.ORB_create(nfeatures=6000, fastThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return detector, matcher, "orb"


def _preprocess_for_features(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


def _dedupe_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    unique: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[tuple[int, str], tuple[int, str]]] = set()
    for view_a, view_b in pairs:
        key_a = (int(view_a["frame_index"]), str(view_a["camera_kind"]))
        key_b = (int(view_b["frame_index"]), str(view_b["camera_kind"]))
        key = tuple(sorted((key_a, key_b)))
        if key in seen:
            continue
        seen.add(key)
        unique.append((view_a, view_b))
    return unique


def _filter_reprojection(
    points_world: np.ndarray,
    image_points: np.ndarray,
    pose_dict: dict[str, Any],
    intrinsics: np.ndarray,
    max_error_px: float,
) -> np.ndarray:
    projection = _projection_from_pose_and_intrinsics(pose_dict, intrinsics)
    points_h = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = (projection @ points_h.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected_xy - image_points.astype(np.float64), axis=1)
    return errors <= max_error_px


def _write_plotly_view(points: np.ndarray, colors_rgb: np.ndarray, output_path: Path) -> None:
    import plotly.graph_objects as go

    if len(points) > 35000:
        stride = max(1, len(points) // 35000)
        points = points[::stride]
        colors_rgb = colors_rgb[::stride]

    color_strings = [
        f"rgb({int(r)}, {int(g)}, {int(b)})" for r, g, b in colors_rgb.astype(np.uint8)
    ]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": 2,
                    "opacity": 0.75,
                    "color": color_strings,
                },
            )
        ]
    )
    figure.update_layout(
        title="Sourccey Spatial Reconstruction",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    figure.write_html(str(output_path), include_plotlyjs=True, auto_open=False)


def _camera_info_for_view(
    frame_record: dict[str, Any],
    image: np.ndarray,
    camera_kind: str,
) -> dict[str, Any] | None:
    primary_info = frame_record.get("primary_camera_info")
    width = float(image.shape[1])
    height = float(image.shape[0])

    if camera_kind == "primary" and primary_info is not None:
        return primary_info

    if primary_info is not None and camera_kind == "companion":
        primary_width = max(1.0, float(primary_info.get("width", image.shape[1])))
        primary_height = max(1.0, float(primary_info.get("height", image.shape[0])))
        scale_x = width / primary_width
        scale_y = height / primary_height
        intrinsics = np.array(primary_info["K"], dtype=np.float64).reshape(3, 3)
        intrinsics[0, 0] *= scale_x
        intrinsics[1, 1] *= scale_y
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 2] *= scale_y
    else:
        default_fov = _DEFAULT_BOTTOM_FOV_DEG if camera_kind == "bottom" else _DEFAULT_PRIMARY_FOV_DEG
        intrinsics = _intrinsics_from_fov(width, height, default_fov)

    distortion = [] if primary_info is None else [float(x) for x in primary_info.get("D", [])]
    return {
        "frame_id": f"{camera_kind}_camera_optical",
        "width": int(width),
        "height": int(height),
        "K": intrinsics.reshape(-1).tolist(),
        "D": distortion,
        "R": [] if primary_info is None else [float(x) for x in primary_info.get("R", [])],
        "P": [] if primary_info is None else [float(x) for x in primary_info.get("P", [])],
        "distortion_model": "plumb_bob" if primary_info is None else primary_info.get("distortion_model", "plumb_bob"),
        "ts": 0.0 if primary_info is None else float(primary_info.get("ts", 0.0)),
    }



def _iter_reconstruction_views(session_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    image_specs = (
        ("primary", "primary_image", "primary_pose"),
        ("companion", "companion_image", "companion_pose"),
        ("bottom", "bottom_image", "bottom_pose"),
    )
    for frame_record in manifest.get("frames", []):
        for camera_kind, image_key, pose_key in image_specs:
            image_rel = frame_record.get(image_key)
            pose_dict = frame_record.get(pose_key)
            if not image_rel or pose_dict is None:
                continue
            image_path = session_dir / image_rel
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            camera_info = _camera_info_for_view(frame_record, image, camera_kind)
            if camera_info is None:
                continue
            views.append(
                {
                    "camera_kind": camera_kind,
                    "frame_index": int(frame_record.get("frame_index", 0)),
                    "frame_id": str(frame_record.get("frame_id", "")),
                    "timestamp": float(frame_record.get("timestamp", 0.0)),
                    "image_path": str(image_path),
                    "image": image,
                    "pose": pose_dict,
                    "camera_info": camera_info,
                }
            )
    return views



def _should_pair_views(
    view_a: dict[str, Any],
    view_b: dict[str, Any],
    *,
    min_translation_m: float,
    min_rotation_deg: float,
) -> bool:
    if view_a["camera_kind"] != view_b["camera_kind"] and view_a["frame_index"] == view_b["frame_index"]:
        return True
    translation_delta = _translation_delta_m(view_a["pose"], view_b["pose"])
    rotation_delta = _rotation_delta_deg(view_a["pose"], view_b["pose"])
    return translation_delta >= min_translation_m or rotation_delta >= min_rotation_deg



def _candidate_view_pairs(
    views: list[dict[str, Any]],
    *,
    min_translation_m: float,
    min_rotation_deg: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for view in views:
        by_frame.setdefault(int(view["frame_index"]), {})[str(view["camera_kind"])] = view
        by_camera.setdefault(str(view["camera_kind"]), []).append(view)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    # Highest-value pairs: synchronized multi-camera views from the same stored frame.
    preferred_same_frame_pairs = (("primary", "companion"), ("primary", "bottom"), ("companion", "bottom"))
    for frame_views in by_frame.values():
        for left_kind, right_kind in preferred_same_frame_pairs:
            left_view = frame_views.get(left_kind)
            right_view = frame_views.get(right_kind)
            if left_view is not None and right_view is not None:
                pairs.append((left_view, right_view))

    # Same-camera temporal pairs only when there is real translational baseline.
    # Bottom-only temporal pairs are especially noisy, so demand a larger baseline there.
    for camera_kind, camera_views in by_camera.items():
        ordered = sorted(camera_views, key=lambda view: int(view["frame_index"]))
        for view_a, view_b in zip(ordered, ordered[1:]):
            translation_delta = _translation_delta_m(view_a["pose"], view_b["pose"])
            required_translation = max(min_translation_m, 0.01)
            if camera_kind == "bottom":
                required_translation = max(required_translation * 3.0, 0.03)
            if translation_delta >= required_translation:
                pairs.append((view_a, view_b))

    # Cross-camera temporal pairs only between adjacent stored frames, centered on the forward-looking rig.
    ordered_frames = sorted(by_frame)
    for prev_index, next_index in zip(ordered_frames, ordered_frames[1:]):
        prev_views = by_frame[prev_index]
        next_views = by_frame[next_index]
        for left_kind in ("primary", "companion"):
            for right_kind in ("primary", "companion"):
                if left_kind == right_kind:
                    continue
                left_view = prev_views.get(left_kind)
                right_view = next_views.get(right_kind)
                if left_view is None or right_view is None:
                    continue
                if _should_pair_views(
                    left_view,
                    right_view,
                    min_translation_m=max(min_translation_m, 0.01),
                    min_rotation_deg=min_rotation_deg,
                ):
                    pairs.append((left_view, right_view))

    return _dedupe_pairs(pairs)



def _triangulate_pair(
    view_a: dict[str, Any],
    view_b: dict[str, Any],
    *,
    detector: Any,
    matcher: Any,
    min_matches: int,
    max_pair_range_m: float,
    reprojection_error_px: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    image_a = view_a["image"]
    image_b = view_b["image"]
    gray_a = _preprocess_for_features(image_a)
    gray_b = _preprocess_for_features(image_b)
    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None:
        return None

    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.84 * n.distance:
            good_matches.append(m)
    if len(good_matches) < min_matches:
        return None

    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches])
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches])

    if len(points_a) >= 8:
        _, fundamental_mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, 2.0, 0.995)
        if fundamental_mask is not None:
            fundamental_mask = fundamental_mask.reshape(-1).astype(bool)
            points_a = points_a[fundamental_mask]
            points_b = points_b[fundamental_mask]
    if len(points_a) < min_matches:
        return None

    intrinsics_a = np.array(view_a["camera_info"]["K"], dtype=np.float64).reshape(3, 3)
    intrinsics_b = np.array(view_b["camera_info"]["K"], dtype=np.float64).reshape(3, 3)
    projection_a = _projection_from_pose_and_intrinsics(view_a["pose"], intrinsics_a)
    projection_b = _projection_from_pose_and_intrinsics(view_b["pose"], intrinsics_b)

    triangulated = cv2.triangulatePoints(projection_a, projection_b, points_a.T, points_b.T)
    points_world = (triangulated[:3] / triangulated[3]).T
    finite_mask = np.isfinite(points_world).all(axis=1)
    points_world = points_world[finite_mask]
    points_a = points_a[finite_mask]
    points_b = points_b[finite_mask]
    if len(points_world) == 0:
        return None

    depth_a = _camera_depths(points_world, view_a["pose"])
    depth_b = _camera_depths(points_world, view_b["pose"])
    depth_mask = (depth_a > 0.08) & (depth_b > 0.08) & (depth_a < max_pair_range_m) & (depth_b < max_pair_range_m)
    points_world = points_world[depth_mask]
    points_a = points_a[depth_mask]
    points_b = points_b[depth_mask]
    if len(points_world) < max(6, min_matches // 2):
        return None

    reprojection_mask = _filter_reprojection(
        points_world, points_a, view_a["pose"], intrinsics_a, reprojection_error_px
    ) & _filter_reprojection(
        points_world, points_b, view_b["pose"], intrinsics_b, reprojection_error_px
    )
    points_world = points_world[reprojection_mask]
    points_a = points_a[reprojection_mask]
    if len(points_world) < max(6, min_matches // 2):
        return None

    sample_x = np.clip(np.round(points_a[:, 0]).astype(int), 0, image_a.shape[1] - 1)
    sample_y = np.clip(np.round(points_a[:, 1]).astype(int), 0, image_a.shape[0] - 1)
    colors_rgb = image_a[sample_y, sample_x][:, ::-1]
    return points_world.astype(np.float32), colors_rgb.astype(np.uint8)



def reconstruct_session(
    session: str | Path = "latest",
    *,
    output_dir: str | Path | None = None,
    min_matches: int = 12,
    min_translation_m: float = 0.01,
    min_rotation_deg: float = 2.0,
    max_pair_range_m: float = 8.0,
    reprojection_error_px: float = 4.0,
    voxel_size_m: float = 0.03,
) -> dict[str, Any]:
    manifest_path, manifest = _load_manifest(session)
    session_dir = manifest_path.parent
    views = _iter_reconstruction_views(session_dir, manifest)
    if len(views) < 2:
        raise ValueError("At least two camera views are required for reconstruction.")

    detector, matcher, detector_name = _create_feature_pipeline()
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    used_pairs = 0
    attempted_pairs = 0

    for view_a, view_b in _candidate_view_pairs(
        views,
        min_translation_m=min_translation_m,
        min_rotation_deg=min_rotation_deg,
    ):
        attempted_pairs += 1
        pair_result = _triangulate_pair(
            view_a,
            view_b,
            detector=detector,
            matcher=matcher,
            min_matches=min_matches,
            max_pair_range_m=max_pair_range_m,
            reprojection_error_px=reprojection_error_px,
        )
        if pair_result is None:
            continue
        points_world, colors_rgb = pair_result
        if len(points_world) == 0:
            continue
        point_sets.append(points_world)
        color_sets.append(colors_rgb)
        used_pairs += 1
        logger.info(
            "Spatial reconstruct accepted pair cameras=(%s,%s) frame_ids=(%s,%s) points=%d",
            view_a["camera_kind"],
            view_b["camera_kind"],
            view_a["frame_id"],
            view_b["frame_id"],
            len(points_world),
        )

    if not point_sets:
        raise ValueError(
            f"No triangulated point pairs passed the reconstruction filters (attempted_pairs={attempted_pairs}, views={len(views)})."
        )

    points = np.concatenate(point_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0)

    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        if voxel_size_m > 0:
            cloud = cloud.voxel_down_sample(voxel_size_m)
        if len(cloud.points) > 32:
            cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.clip(np.asarray(cloud.colors) * 255.0, 0, 255).astype(np.uint8)
    except Exception as exc:
        logger.warning("Open3D refinement skipped during spatial reconstruction: %s", exc)

    output_root = Path(output_dir) if output_dir is not None else _DEFAULT_RECON_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = output_root / f"point_cloud_{timestamp}.npz"
    ply_path = output_root / f"point_cloud_{timestamp}.ply"
    html_path = output_root / f"point_cloud_{timestamp}.html"
    summary_path = output_root / f"point_cloud_{timestamp}.json"

    np.savez_compressed(npz_path, points=points, colors=colors)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with ply_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    _write_plotly_view(points, colors, html_path)

    camera_counts: dict[str, int] = {}
    for view in views:
        camera_counts[view["camera_kind"]] = camera_counts.get(view["camera_kind"], 0) + 1

    summary = {
        "schema": "sourccey.spatial_reconstruction.v1",
        "manifest_path": str(manifest_path),
        "session_dir": str(session_dir),
        "view_count": len(views),
        "camera_counts": camera_counts,
        "attempted_pairs": attempted_pairs,
        "used_pairs": used_pairs,
        "point_count": int(len(points)),
        "detector": detector_name,
        "outputs": {
            "npz": str(npz_path),
            "ply": str(ply_path),
            "html": str(html_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest = _latest_reconstruction_outputs(output_root)
    shutil.copyfile(npz_path, latest["npz"])
    shutil.copyfile(ply_path, latest["ply"])
    shutil.copyfile(html_path, latest["html"])
    latest["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["summary_path"] = str(summary_path)
    summary["latest_html"] = str(latest["html"])
    return summary


def resolve_latest_view(output_dir: str | Path | None = None) -> Path:
    output_root = Path(output_dir) if output_dir is not None else _DEFAULT_RECON_ROOT
    latest_html = _latest_reconstruction_outputs(output_root)["html"]
    if latest_html.exists():
        return latest_html
    candidates = sorted(
        output_root.glob("point_cloud_*.html"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        logger.warning(
            "Latest spatial viewer pointer was missing; falling back to newest HTML at %s",
            candidates[0],
        )
        return candidates[0]
    raise FileNotFoundError(
        f"Latest spatial viewer HTML not found at {latest_html} and no point_cloud_*.html files exist under {output_root}. Run spatial reconstruction first."
    )
