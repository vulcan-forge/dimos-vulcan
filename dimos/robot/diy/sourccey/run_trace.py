from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


_TRACE_ROOT = Path("artifacts") / "sourccey_run_traces"
_CURRENT_SESSION_PATH = _TRACE_ROOT / "_current_session.json"
_LOCK_PATH = _TRACE_ROOT / "_trace.lock"
_LATEST_DOC_PATH = _TRACE_ROOT / "latest_run_trace.doc"
_LATEST_HTML_PATH = _TRACE_ROOT / "latest_run_trace.html"
_LATEST_JSONL_PATH = _TRACE_ROOT / "latest_run_trace.jsonl"
_SESSION_STALE_S = 900.0


def _ensure_root() -> None:
    _TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    if not _LOCK_PATH.exists():
        _LOCK_PATH.touch()


@contextmanager
def _locked_trace_root() -> Any:
    _ensure_root()
    with _LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_current_session() -> dict[str, Any] | None:
    if not _CURRENT_SESSION_PATH.exists():
        return None
    try:
        return json.loads(_CURRENT_SESSION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _create_session(label: str, started_by: str) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = _TRACE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "run_id": run_id,
        "label": label,
        "started_by": started_by,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_ts": time.time(),
        "last_event_ts": time.time(),
        "run_dir": str(run_dir),
        "jsonl_path": str(run_dir / "run_trace.jsonl"),
        "html_path": str(run_dir / "run_trace.html"),
        "doc_path": str(run_dir / "run_trace.doc"),
    }
    _write_json(_CURRENT_SESSION_PATH, session)
    return session


def _clear_previous_runs() -> None:
    for child in _TRACE_ROOT.iterdir():
        if child in {_LOCK_PATH, _CURRENT_SESSION_PATH}:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass


def _get_or_create_session(*, label: str, component: str) -> dict[str, Any]:
    session = _load_current_session()
    now = time.time()
    if session is None:
        return _create_session(label=label, started_by=component)
    last_event_ts = float(session.get("last_event_ts", session.get("started_ts", 0.0)))
    if (now - last_event_ts) > _SESSION_STALE_S:
        return _create_session(label=label, started_by=component)
    return session


def start_new_run(*, component: str, label: str = "sourccey_run") -> dict[str, Any]:
    with _locked_trace_root():
        _clear_previous_runs()
        session = _create_session(label=label, started_by=component)
        _append_entry_locked(
            session,
            component=component,
            event="run_started",
            fields={"label": label},
        )
        return session


def trace_event(component: str, event: str, **fields: Any) -> None:
    with _locked_trace_root():
        session = _get_or_create_session(label="sourccey_run", component=component)
        _append_entry_locked(session, component=component, event=event, fields=fields)


def _append_entry_locked(
    session: dict[str, Any],
    *,
    component: str,
    event: str,
    fields: dict[str, Any],
) -> None:
    run_dir = Path(str(session["run_dir"]))
    jsonl_path = Path(str(session["jsonl_path"]))
    entry = {
        "ts": time.time(),
        "iso_ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        "fields": _sanitize(fields),
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")
    session["last_event_ts"] = float(entry["ts"])
    _write_json(_CURRENT_SESSION_PATH, session)
    _render_report_locked(run_dir, session)


def _read_entries(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _render_report_locked(run_dir: Path, session: dict[str, Any]) -> None:
    jsonl_path = Path(str(session["jsonl_path"]))
    html_path = Path(str(session["html_path"]))
    doc_path = Path(str(session["doc_path"]))
    entries = _read_entries(jsonl_path)

    rows: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        fields_json = json.dumps(entry.get("fields", {}), indent=2, sort_keys=True)
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{escape(str(entry.get('iso_ts', '')))}</td>"
            f"<td>{escape(str(entry.get('component', '')))}</td>"
            f"<td>{escape(str(entry.get('event', '')))}</td>"
            f"<td><pre>{escape(fields_json)}</pre></td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sourccey Run Trace {escape(str(session['run_id']))}</title>
  <style>
    body {{
      font-family: Consolas, 'Courier New', monospace;
      background: #0b1220;
      color: #e5eefc;
      margin: 24px;
    }}
    h1, h2 {{ color: #ffffff; }}
    .meta {{
      margin-bottom: 18px;
      padding: 12px 16px;
      border: 1px solid #24324a;
      border-radius: 10px;
      background: #101a2c;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #24324a;
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #14213a;
    }}
    tr:nth-child(even) {{
      background: #0f1727;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <h1>Sourccey Run Trace</h1>
  <div class="meta">
    <div><strong>Run ID:</strong> {escape(str(session['run_id']))}</div>
    <div><strong>Label:</strong> {escape(str(session['label']))}</div>
    <div><strong>Started By:</strong> {escape(str(session['started_by']))}</div>
    <div><strong>Hostname:</strong> {escape(str(session['hostname']))}</div>
    <div><strong>Started:</strong> {escape(datetime.fromtimestamp(float(session['started_ts']), tz=timezone.utc).isoformat())}</div>
    <div><strong>Latest Event:</strong> {escape(datetime.fromtimestamp(float(session['last_event_ts']), tz=timezone.utc).isoformat())}</div>
    <div><strong>Raw JSONL:</strong> {escape(str(jsonl_path))}</div>
  </div>
  <h2>Event Timeline</h2>
  <table>
    <thead>
      <tr>
        <th style="width:60px;">#</th>
        <th style="width:240px;">Timestamp (UTC)</th>
        <th style="width:180px;">Component</th>
        <th style="width:220px;">Event</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    doc_path.write_text(html, encoding="utf-8")
    shutil.copyfile(jsonl_path, _LATEST_JSONL_PATH)
    shutil.copyfile(html_path, _LATEST_HTML_PATH)
    shutil.copyfile(doc_path, _LATEST_DOC_PATH)
