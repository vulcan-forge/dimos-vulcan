from __future__ import annotations

import os
import platform
import threading
import time
from typing import Any

import cv2
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SourcceyOpenCVDebugViewerConfig(ModuleConfig):
    status_interval_s: float = 2.0
    poll_interval_s: float = 0.03
    window_prefix: str = Field(default="Sourccey Debug")
    auto_enhance_preview: bool = True
    denoise_bottom_preview: bool = True
    allow_headless_fallback: bool = True
    force_headless_preview: bool = False
    enable_wsl_gui_preview: bool = False


class SourcceyOpenCVDebugViewer(Module):
    dedicated_worker = True

    config: SourcceyOpenCVDebugViewerConfig

    color_image: In[Image]
    companion_image: In[Image]
    bottom_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gui_enabled = True
        self._latest_frames: dict[str, tuple[float, Any] | None] = {
            "Primary": None,
            "Companion": None,
            "Bottom": None,
        }

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(Disposable(self.companion_image.subscribe(self._on_companion_image)))
        self.register_disposable(Disposable(self.bottom_image.subscribe(self._on_bottom_image)))
        self._stop_event.clear()
        self._gui_enabled = self._detect_gui_availability()
        self._thread = threading.Thread(target=self._display_loop, name="sourccey-opencv-debug-viewer", daemon=True)
        self._thread.start()
        if self._gui_enabled:
            logger.info("%s viewer started", self.config.window_prefix)
        else:
            logger.warning(
                "%s viewer started in headless fallback mode; GUI preview disabled but sensor status logging remains active",
                self.config.window_prefix,
            )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        if self._gui_enabled:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        logger.info("%s viewer stopped", self.config.window_prefix)
        super().stop()

    def _on_color_image(self, msg: Image) -> None:
        self._store_frame("Primary", msg)

    def _on_companion_image(self, msg: Image) -> None:
        self._store_frame("Companion", msg)

    def _on_bottom_image(self, msg: Image) -> None:
        self._store_frame("Bottom", msg)

    def _store_frame(self, name: str, msg: Image) -> None:
        try:
            frame = msg.to_opencv().copy()
            frame = self._prepare_preview_frame(name, frame)
        except Exception as exc:
            logger.warning(
                "Failed to convert Sourccey debug image",
                stream=name,
                error=str(exc),
            )
            return
        with self._lock:
            self._latest_frames[name] = (time.time(), frame)

    def _detect_gui_availability(self) -> bool:
        if self.config.force_headless_preview:
            logger.warning("%s GUI preview forced off by configuration", self.config.window_prefix)
            return False

        if os.name != "posix":
            return True

        has_display = bool(os.environ.get("DISPLAY"))
        has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        has_runtime_dir = bool(os.environ.get("XDG_RUNTIME_DIR"))
        is_wsl = "microsoft" in platform.release().lower()

        if is_wsl and not self.config.enable_wsl_gui_preview:
            if self.config.allow_headless_fallback:
                logger.warning(
                    "%s GUI preview disabled under WSL by default; set enable_wsl_gui_preview=true to try native preview",
                    self.config.window_prefix,
                )
                return False
            return True

        if has_display or (has_wayland and has_runtime_dir):
            return True

        if self.config.allow_headless_fallback:
            logger.warning(
                "%s GUI preview disabled: no DISPLAY/WAYLAND environment detected%s",
                self.config.window_prefix,
                " under WSL" if is_wsl else "",
            )
            return False

        return True

    def _prepare_preview_frame(self, name: str, frame: Any) -> Any:
        if not self.config.auto_enhance_preview:
            return frame

        preview = frame
        if name == "Bottom" and self.config.denoise_bottom_preview:
            try:
                preview = cv2.fastNlMeansDenoisingColored(preview, None, 4, 4, 7, 21)
            except Exception:
                pass

        # Lift dark feeds without touching the raw transport by equalizing the
        # luminance channel and then applying a mild gain toward a usable mean.
        lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        enhanced = cv2.merge((l_chan, a_chan, b_chan))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        mean_luma = float(np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
        if mean_luma > 1.0:
            gain = min(2.4, max(1.0, 110.0 / mean_luma))
            enhanced = cv2.convertScaleAbs(enhanced, alpha=gain, beta=0)

        return enhanced

    def _display_loop(self) -> None:
        last_status = 0.0
        poll_interval = max(float(self.config.poll_interval_s), 0.01)
        status_interval = max(float(self.config.status_interval_s), 0.5)
        window_names = {
            name: f"{self.config.window_prefix} - {name}" for name in self._latest_frames
        }

        while not self._stop_event.is_set():
            frames = self._snapshot_frames()
            if self._gui_enabled:
                for name, payload in frames.items():
                    if payload is None:
                        continue
                    _, frame = payload
                    try:
                        cv2.imshow(window_names[name], frame)
                    except Exception as exc:
                        logger.warning("Failed to display Sourccey debug frame", window=name, error=str(exc))
                        self._stop_event.set()
                        break
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("%s viewer received q; closing windows", self.config.window_prefix)
                        self._stop_event.set()
                        break
                except Exception as exc:
                    logger.warning("Sourccey debug viewer waitKey failed", error=str(exc))
                    self._stop_event.set()
                    break

            now = time.time()
            if now - last_status >= status_interval:
                logger.info(self._format_status(frames, now))
                last_status = now
            time.sleep(poll_interval)

    def _snapshot_frames(self) -> dict[str, tuple[float, Any] | None]:
        with self._lock:
            return dict(self._latest_frames)

    def _format_status(self, frames: dict[str, tuple[float, Any] | None], now: float) -> str:
        parts: list[str] = []
        for name, payload in frames.items():
            if payload is None:
                parts.append(f"{name}=missing")
                continue
            ts, frame = payload
            age = max(0.0, now - ts)
            height, width = frame.shape[:2]
            parts.append(f"{name}=ok age={age:.2f}s size={width}x{height}")
        return f"{self.config.window_prefix}: " + " | ".join(parts)
