from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import re
import threading
import time
from typing import Any

import cv2
import numpy as np
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT, DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DEFAULT_ROOT = DIMOS_PROJECT_ROOT / "assets" / "output" / "sourccey_landmarks"
_DEFAULT_STORE_PATH = _DEFAULT_ROOT / "landmarks.pkl"
_DEFAULT_STATUS_PATH = _DEFAULT_ROOT / "latest_match.json"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return slug.strip("-") or "landmark"


def _image_to_bgr(msg: Image | None) -> np.ndarray | None:
    if msg is None or not hasattr(msg, "data"):
        return None
    frame = np.asarray(msg.data)
    if frame.ndim == 2:
        return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class LandmarkRecord(dict):
    """Plain dict-backed record kept pickle-friendly across refactors."""


class LandmarkMatch(dict):
    """Best-match status payload for local DimOS consumers."""


def load_landmark_records(store_path: str | Path) -> list[LandmarkRecord]:
    path = Path(store_path)
    if not path.exists():
        return []
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    records: list[LandmarkRecord] = []
    for item in raw if isinstance(raw, list) else []:
        records.append(LandmarkRecord(item))
    return records


def save_landmark_records(store_path: str | Path, records: list[LandmarkRecord]) -> None:
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(records), handle)


class SourcceyVisualLandmarkMemoryConfig(ModuleConfig):
    orb_features: int = 800
    ratio_test: float = 0.75
    max_hamming_distance: int = 48
    min_good_matches: int = 22
    match_period_s: float = 0.6
    store_path: str = str(_DEFAULT_STORE_PATH)
    status_path: str = str(_DEFAULT_STATUS_PATH)


class SourcceyVisualLandmarkMemory(Module):
    dedicated_worker = True

    config: SourcceyVisualLandmarkMemoryConfig

    color_image: In[Image]
    odom: In[PoseStamped]

    landmark_pose: Out[PoseStamped]
    landmark_match: Out[LandmarkMatch]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_frame: np.ndarray | None = None
        self._latest_odom: PoseStamped | None = None
        self._records = load_landmark_records(self.config.store_path)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._orb = cv2.ORB_create(nfeatures=max(int(self.config.orb_features), 128))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        Path(self.config.store_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.status_path).parent.mkdir(parents=True, exist_ok=True)
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._match_loop,
            name="sourccey-visual-landmark-memory",
            daemon=True,
        )
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        super().stop()

    @rpc
    def save_landmark(self, name: str, room: str | None = None, note: str | None = None) -> bool:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            odom = self._latest_odom
        if frame is None or odom is None:
            return False

        descriptors = self._extract_descriptors(frame)
        if descriptors is None or len(descriptors) < int(self.config.min_good_matches):
            return False

        preview_dir = Path(self.config.store_path).parent / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{_slugify(name)}_{int(time.time())}.jpg"
        cv2.imwrite(str(preview_path), frame)

        record = LandmarkRecord(
            name=str(name),
            room=None if room is None else str(room),
            note=None if note is None else str(note),
            ts=float(odom.ts),
            pose=(float(odom.x), float(odom.y), float(odom.z)),
            yaw_rad=float(odom.yaw),
            descriptors=descriptors,
            preview_path=str(preview_path),
        )
        self._records.append(record)
        save_landmark_records(self.config.store_path, self._records)
        return True

    @rpc
    def list_landmarks(self) -> list[dict[str, Any]]:
        return [
            {
                "name": record.get("name"),
                "room": record.get("room"),
                "note": record.get("note"),
                "pose": tuple(record.get("pose", (0.0, 0.0, 0.0))),
                "yaw_rad": float(record.get("yaw_rad", 0.0)),
                "preview_path": record.get("preview_path"),
            }
            for record in self._records
        ]

    @rpc
    def clear_landmarks(self) -> int:
        count = len(self._records)
        self._records.clear()
        save_landmark_records(self.config.store_path, self._records)
        return count

    @rpc
    def best_match(self) -> dict[str, Any] | None:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        return self._match_frame(frame)

    def _on_color_image(self, msg: Image) -> None:
        frame = _image_to_bgr(msg)
        if frame is None:
            return
        with self._lock:
            self._latest_frame = frame

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = msg

    def _match_loop(self) -> None:
        period_s = max(0.2, float(self.config.match_period_s))
        while not self._stop_event.is_set():
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None:
                match = self._match_frame(frame)
                if match is not None:
                    pose = PoseStamped(
                        ts=float(match["ts"]),
                        frame_id="map",
                        position=list(match["pose"]),
                        orientation=(0.0, 0.0, math.sin(float(match["yaw_rad"]) / 2.0), math.cos(float(match["yaw_rad"]) / 2.0)),
                    )
                    self.landmark_pose.publish(pose)
                    self.landmark_match.publish(LandmarkMatch(match))
                    self._write_status(match)
            time.sleep(period_s)

    def _extract_descriptors(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            return None
        return descriptors

    def _match_frame(self, frame: np.ndarray) -> dict[str, Any] | None:
        query_descriptors = self._extract_descriptors(frame)
        if query_descriptors is None:
            return None

        best_record: LandmarkRecord | None = None
        best_score = 0
        for record in self._records:
            train_descriptors = record.get("descriptors")
            if not isinstance(train_descriptors, np.ndarray) or len(train_descriptors) == 0:
                continue
            knn_matches = self._matcher.knnMatch(query_descriptors, train_descriptors, k=2)
            good_matches = []
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                first, second = pair
                if (
                    first.distance <= int(self.config.max_hamming_distance)
                    and first.distance < float(self.config.ratio_test) * second.distance
                ):
                    good_matches.append(first)
            score = len(good_matches)
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None or best_score < int(self.config.min_good_matches):
            return None

        return {
            "name": best_record.get("name"),
            "room": best_record.get("room"),
            "note": best_record.get("note"),
            "pose": tuple(best_record.get("pose", (0.0, 0.0, 0.0))),
            "yaw_rad": float(best_record.get("yaw_rad", 0.0)),
            "preview_path": best_record.get("preview_path"),
            "score": best_score,
            "ts": time.time(),
        }

    def _write_status(self, match: dict[str, Any]) -> None:
        Path(self.config.status_path).write_text(json.dumps(match, indent=2), encoding="utf-8")
