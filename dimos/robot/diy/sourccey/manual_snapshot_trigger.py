from __future__ import annotations

import os
import select
import sys
import threading
import time
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.std_msgs.Bool import Bool
from dimos.utils.logging_config import setup_logger

from .run_trace import start_new_run, trace_event

logger = setup_logger()


def _console_notice(message: str) -> None:
    print(f"[manual_snapshot_trigger] {message}", flush=True)

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows fallback
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX path
    msvcrt = None  # type: ignore[assignment]


class SourcceyManualSnapshotTriggerConfig(ModuleConfig):
    poll_interval_s: float = Field(default=0.05, ge=0.01, le=0.50)


class SourcceyManualSnapshotTrigger(Module):
    config: SourcceyManualSnapshotTriggerConfig

    snapshot_request: Out[Bool]
    reset_request: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stdin_fd: int | None = None
        self._stdin_settings: list[Any] | None = None
        self._keypress_count = 0

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        start_new_run(component="manual_snapshot_trigger", label="manual_snapshot_mapper")
        trace_event(
            "manual_snapshot_trigger",
            "start",
            controls="space=capture_snapshot r=reset_map q=quit_listener",
            os_name=os.name,
            stdin_is_tty=bool(sys.stdin.isatty()),
            poll_interval_s=float(self.config.poll_interval_s),
        )
        logger.info(
            "Manual snapshot controls ready. Focus this terminal and press SPACE to capture, R to reset."
        )
        _console_notice("started: focus this terminal and press SPACE to capture, R to reset, Q to stop listener")
        self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._thread.start()
        trace_event(
            "manual_snapshot_trigger",
            "keyboard_thread_started",
            thread_name=self._thread.name,
            daemon=bool(self._thread.daemon),
        )
        self.register_disposable(Disposable(self._restore_terminal))

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._restore_terminal()
        trace_event(
            "manual_snapshot_trigger",
            "stop",
            keypress_count=int(self._keypress_count),
            thread_alive=bool(self._thread.is_alive()) if self._thread is not None else False,
        )
        super().stop()

    def _keyboard_loop(self) -> None:
        trace_event(
            "manual_snapshot_trigger",
            "keyboard_loop_enter",
            mode="windows" if (os.name == "nt" and msvcrt is not None) else "posix",
        )
        if os.name == "nt" and msvcrt is not None:
            self._keyboard_loop_windows()
            return
        self._keyboard_loop_posix()

    def _keyboard_loop_windows(self) -> None:
        poll_s = float(self.config.poll_interval_s)
        trace_event("manual_snapshot_trigger", "keyboard_loop_windows_start", poll_interval_s=poll_s)
        while not self._stop_event.is_set():
            if not msvcrt.kbhit():
                time.sleep(poll_s)
                continue
            key = msvcrt.getwch()
            trace_event(
                "manual_snapshot_trigger",
                "key_detected",
                source="windows",
                key=repr(key),
            )
            self._handle_key(key)
        trace_event("manual_snapshot_trigger", "keyboard_loop_windows_exit")

    def _keyboard_loop_posix(self) -> None:
        if termios is None or tty is None:
            trace_event("manual_snapshot_trigger", "stdin_unsupported")
            logger.warning("Manual snapshot trigger unavailable: termios/tty not present.")
            _console_notice("DISABLED: termios/tty not present, so keyboard capture is unavailable")
            return
        if not sys.stdin.isatty():
            trace_event("manual_snapshot_trigger", "stdin_not_tty")
            logger.warning("Manual snapshot trigger unavailable: stdin is not a TTY.")
            _console_notice("DISABLED: stdin is not a TTY, so pressing SPACE in this run will do nothing")
            return

        self._stdin_fd = sys.stdin.fileno()
        self._stdin_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        poll_s = float(self.config.poll_interval_s)
        trace_event(
            "manual_snapshot_trigger",
            "keyboard_loop_posix_start",
            stdin_fd=int(self._stdin_fd),
            poll_interval_s=poll_s,
        )
        try:
            while not self._stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], poll_s)
                if not readable:
                    continue
                key = sys.stdin.read(1)
                trace_event(
                    "manual_snapshot_trigger",
                    "key_detected",
                    source="posix",
                    key=repr(key),
                )
                _console_notice(f"key detected: {key!r}")
                self._handle_key(key)
        finally:
            trace_event("manual_snapshot_trigger", "keyboard_loop_posix_exit")
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self._stdin_fd is None or self._stdin_settings is None or termios is None:
            return
        try:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_settings)
        except Exception:
            pass
        self._stdin_fd = None
        self._stdin_settings = None

    def _handle_key(self, key: str) -> None:
        self._keypress_count += 1
        trace_event(
            "manual_snapshot_trigger",
            "handle_key",
            key=repr(key),
            keypress_count=int(self._keypress_count),
        )
        if key == " ":
            _console_notice("SPACE detected -> publishing snapshot request")
            trace_event("manual_snapshot_trigger", "snapshot_publish_begin")
            self.snapshot_request.publish(Bool(True))
            trace_event("manual_snapshot_trigger", "snapshot_request")
            trace_event("manual_snapshot_trigger", "snapshot_publish_end")
            logger.info("Manual snapshot requested")
            _console_notice("snapshot request published")
            return
        if key in {"r", "R"}:
            _console_notice("R detected -> publishing reset request")
            trace_event("manual_snapshot_trigger", "reset_publish_begin")
            self.reset_request.publish(Bool(True))
            trace_event("manual_snapshot_trigger", "reset_request")
            trace_event("manual_snapshot_trigger", "reset_publish_end")
            logger.info("Manual snapshot map reset requested")
            _console_notice("reset request published")
            return
        if key in {"q", "Q"}:
            self._stop_event.set()
            trace_event("manual_snapshot_trigger", "listener_quit")
            logger.info("Manual snapshot keyboard listener stopped")
            _console_notice("Q detected -> keyboard listener stopping")
            return
        _console_notice(f"ignored key: {key!r}")
        trace_event("manual_snapshot_trigger", "unhandled_key", key=repr(key))
