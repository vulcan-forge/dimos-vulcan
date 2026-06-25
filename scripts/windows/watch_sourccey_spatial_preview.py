import argparse
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def default_preview_dir(distro: str) -> Path:
    return Path(rf"\\wsl$\{distro}\root\dimos\assets\output\memory\spatial_memory")


class SpatialPreviewApp:
    def __init__(self, root: tk.Tk, preview_dir: Path, poll_ms: int) -> None:
        self.root = root
        self.preview_dir = preview_dir
        self.poll_ms = poll_ms
        self.status_path = preview_dir / "preview_status.json"
        self.image_path = preview_dir / "latest_stored.png"
        self.last_image_mtime = None
        self.photo = None

        self.root.title("Sourccey Spatial Preview")
        self.root.geometry("1100x780")

        self.status_var = tk.StringVar(value=f"Watching {self.preview_dir}")
        self.meta_var = tk.StringVar(value="Waiting for spatial preview artifacts...")

        top = ttk.Frame(root, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.meta_var, font=("Consolas", 10)).pack(anchor="w", pady=(4, 0))

        self.image_label = ttk.Label(root, text="No stored frame yet.", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.refresh()

    def refresh(self) -> None:
        self._refresh_status()
        self._refresh_image()
        self.root.after(self.poll_ms, self.refresh)

    def _refresh_status(self) -> None:
        if not self.status_path.exists():
            self.status_var.set(f"Watching {self.preview_dir}")
            self.meta_var.set("Waiting for spatial preview artifacts...")
            return

        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.meta_var.set(f"Failed to read status: {exc}")
            return

        state = payload.get("state", "unknown")
        stored = payload.get("stored_frame_count", 0)
        processed = payload.get("frame_count", 0)
        frame_id = payload.get("frame_id", "-")
        position = payload.get("position") or {}
        rotation = payload.get("rotation_euler") or {}

        self.status_var.set(f"state={state} stored={stored} processed={processed} frame_id={frame_id}")
        self.meta_var.set(
            "pos=({x:.2f}, {y:.2f}, {z:.2f}) rot=({rx:.2f}, {ry:.2f}, {rz:.2f})".format(
                x=float(position.get("x", 0.0)),
                y=float(position.get("y", 0.0)),
                z=float(position.get("z", 0.0)),
                rx=float(rotation.get("x", 0.0)),
                ry=float(rotation.get("y", 0.0)),
                rz=float(rotation.get("z", 0.0)),
            )
        )

    def _refresh_image(self) -> None:
        if not self.image_path.exists():
            return

        try:
            mtime = self.image_path.stat().st_mtime
        except OSError:
            return

        if self.last_image_mtime == mtime:
            return

        try:
            photo = tk.PhotoImage(file=str(self.image_path))
        except Exception as exc:
            self.meta_var.set(f"Failed to load preview image: {exc}")
            return

        self.photo = photo
        self.image_label.configure(image=self.photo, text="")
        self.last_image_mtime = mtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Sourccey spatial preview artifacts from WSL.")
    parser.add_argument("--distro", default=os.environ.get("DIMOS_WSL_DISTRO", "Ubuntu"))
    parser.add_argument("--poll-ms", type=int, default=750)
    args = parser.parse_args()

    root = tk.Tk()
    app = SpatialPreviewApp(root, default_preview_dir(args.distro), args.poll_ms)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
