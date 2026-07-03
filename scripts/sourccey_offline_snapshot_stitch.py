from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Snapshot:
    request_index: int
    metadata_path: Path
    local_points: np.ndarray
    scan_ts: float


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _transform_points_2d(points_xy: np.ndarray, *, x: float, y: float, yaw: float) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = points_xy @ rotation.T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    return world.astype(np.float32, copy=False)


def _downsample_points(points_xy: np.ndarray, max_points: int) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) <= max_points:
        return points_xy.astype(np.float32, copy=False)
    indices = np.linspace(0, len(points_xy) - 1, max_points, dtype=np.int32)
    return points_xy[indices].astype(np.float32, copy=False)


def _voxelize_points(points_xy: np.ndarray, voxel_m: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    voxel = max(float(voxel_m), 1e-4)
    cells = np.round(points_xy / voxel).astype(np.int32, copy=False)
    _, unique_indices = np.unique(cells, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points_xy[unique_indices].astype(np.float32, copy=False)


def _point_cells(points_xy: np.ndarray, voxel_m: float) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    voxel = max(float(voxel_m), 1e-4)
    cells = np.round(np.asarray(points_xy, dtype=np.float32) / voxel).astype(np.int32, copy=False)
    _, unique_indices = np.unique(cells, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return cells[unique_indices].astype(np.int32, copy=False)


def _cell_set_with_margin(cells: np.ndarray, margin_cells: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    radius = max(int(margin_cells), 0)
    for cx, cy in cells.tolist():
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                expanded.add((int(cx + dx), int(cy + dy)))
    return expanded


def _scan_match_metrics(reference_world: np.ndarray, candidate_world: np.ndarray, *, overlap_radius_m: float) -> tuple[float, float, float]:
    if reference_world.size == 0 or candidate_world.size == 0:
        return float("inf"), 0.0, float("inf")
    deltas = candidate_world[:, None, :] - reference_world[None, :, :]
    min_d2 = np.min(np.sum(deltas * deltas, axis=2), axis=1)
    overlap_radius2 = max(float(overlap_radius_m), 1e-3) ** 2
    overlap_fraction = float(np.mean(min_d2 <= overlap_radius2))
    mean_distance = float(np.mean(np.sqrt(np.clip(min_d2, 0.0, None))))
    score = float(np.mean(np.clip(min_d2, 0.0, 0.25)))
    return score, overlap_fraction, mean_distance


def _load_snapshots(snapshot_dir: Path) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    for metadata_path in sorted(snapshot_dir.glob("snapshot_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        request_index = int(metadata["request_index"])
        local_path = snapshot_dir / f"snapshot_{request_index:03d}_local.npy"
        if not local_path.exists():
            raise FileNotFoundError(f"Missing local snapshot file for request {request_index}: {local_path}")
        local_points = np.load(local_path).astype(np.float32, copy=False)
        snapshots.append(
            Snapshot(
                request_index=request_index,
                metadata_path=metadata_path,
                local_points=_voxelize_points(local_points, 0.02),
                scan_ts=float(metadata["scan_ts"]),
            )
        )
    return snapshots


def _candidate_objective(
    *,
    world_points: np.ndarray,
    reference_points: np.ndarray,
    expected_yaw_rad: float,
    overlap_radius_m: float,
    translation_weight: float,
    yaw_weight_per_deg: float,
    overlap_reward: float,
    coarse_overlap_reward: float,
    coarse_overlap_fraction: float,
) -> dict[str, float]:
    score, overlap, mean_distance = _scan_match_metrics(reference_points, world_points, overlap_radius_m=overlap_radius_m)
    yaw = math.atan2(
        float(world_points[1, 1] - world_points[0, 1]) if len(world_points) > 1 else math.sin(expected_yaw_rad),
        float(world_points[1, 0] - world_points[0, 0]) if len(world_points) > 1 else math.cos(expected_yaw_rad),
    )
    return {
        "score": float(score),
        "overlap_fraction": float(overlap),
        "mean_distance_m": float(mean_distance),
    }


def _coarse_candidate_search(
    *,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    x_center: float,
    y_center: float,
    expected_yaw_rad: float,
    trans_span_m: float,
    trans_step_m: float,
    yaw_span_deg: float,
    yaw_step_deg: float,
    cell_voxel_m: float,
    cell_margin: int,
    overlap_radius_m: float,
    top_k: int,
) -> list[dict[str, Any]]:
    reference_points = _downsample_points(reference_points, 420)
    source_points = _downsample_points(source_points, 260)
    ref_cells = _point_cells(reference_points, cell_voxel_m)
    expanded_ref = _cell_set_with_margin(ref_cells, cell_margin)
    candidates: list[dict[str, Any]] = []

    trans_offsets = np.arange(-trans_span_m, trans_span_m + 1e-9, trans_step_m, dtype=np.float32)
    yaw_offsets_deg = np.arange(-yaw_span_deg, yaw_span_deg + 1e-9, yaw_step_deg, dtype=np.float32)

    voxel = max(float(cell_voxel_m), 1e-4)
    rounded_ref_centroid = np.mean(reference_points, axis=0) if len(reference_points) else np.zeros(2, dtype=np.float32)

    for dyaw_deg in yaw_offsets_deg.tolist():
        yaw = _wrap_angle_rad(float(expected_yaw_rad + math.radians(float(dyaw_deg))))
        rotated = _transform_points_2d(source_points, x=0.0, y=0.0, yaw=yaw)
        rotated_cells = _point_cells(rotated, cell_voxel_m)
        if len(rotated_cells) == 0:
            continue
        rotated_centroid = np.mean(rotated, axis=0) if len(rotated) else np.zeros(2, dtype=np.float32)
        for dx in trans_offsets.tolist():
            for dy in trans_offsets.tolist():
                x = float(x_center + dx)
                y = float(y_center + dy)
                translation_mag = math.hypot(float(x), float(y))
                cell_dx = int(round(float(dx) / voxel))
                cell_dy = int(round(float(dy) / voxel))
                overlap_hits = 0
                for cx, cy in rotated_cells.tolist():
                    if (int(cx + cell_dx), int(cy + cell_dy)) in expanded_ref:
                        overlap_hits += 1
                coarse_overlap_fraction = float(overlap_hits / max(len(rotated_cells), 1))
                translated_centroid = rotated_centroid + np.asarray([x, y], dtype=np.float32)
                centroid_distance = float(np.linalg.norm(translated_centroid - rounded_ref_centroid))
                yaw_error_deg = abs(math.degrees(_wrap_angle_rad(float(yaw - expected_yaw_rad))))
                objective = (
                    (0.85 * centroid_distance)
                    + (0.35 * translation_mag)
                    + (0.022 * yaw_error_deg)
                    - (2.10 * coarse_overlap_fraction)
                )
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "yaw": float(yaw),
                        "yaw_deg": float(math.degrees(float(yaw))),
                        "objective": float(objective),
                        "coarse_overlap_fraction": float(coarse_overlap_fraction),
                        "overlap_fraction": float(coarse_overlap_fraction),
                        "mean_distance_m": float(centroid_distance),
                        "yaw_error_deg": float(yaw_error_deg),
                        "translation_mag_m": float(translation_mag),
                    }
                )
    candidates.sort(
        key=lambda item: (
            item["objective"],
            item["yaw_error_deg"],
            item["translation_mag_m"],
            -item["overlap_fraction"],
            -item["coarse_overlap_fraction"],
        )
    )
    return candidates[: max(int(top_k), 1)]


def _refine_candidate(
    *,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    seed: dict[str, Any],
    expected_yaw_rad: float,
    overlap_radius_m: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    levels = (
        (0.04, 0.01, 6.0, 1.0),
        (0.015, 0.005, 2.0, 0.5),
    )
    source_points = _downsample_points(source_points, 220)
    reference_points = _downsample_points(reference_points, 360)
    best_x = float(seed["x"])
    best_y = float(seed["y"])
    best_yaw = float(seed["yaw"])

    for trans_span, trans_step, yaw_span_deg, yaw_step_deg in levels:
        trans_offsets = np.arange(-trans_span, trans_span + 1e-9, trans_step, dtype=np.float32)
        yaw_offsets_deg = np.arange(-yaw_span_deg, yaw_span_deg + 1e-9, yaw_step_deg, dtype=np.float32)
        for dx in trans_offsets.tolist():
            for dy in trans_offsets.tolist():
                x = float(best_x + dx)
                y = float(best_y + dy)
                translation_mag = math.hypot(float(x), float(y))
                for dyaw_deg in yaw_offsets_deg.tolist():
                    yaw = _wrap_angle_rad(float(best_yaw + math.radians(float(dyaw_deg))))
                    world = _transform_points_2d(source_points, x=x, y=y, yaw=yaw)
                    score, overlap, mean_distance = _scan_match_metrics(reference_points, world, overlap_radius_m=float(overlap_radius_m))
                    yaw_error_deg = abs(math.degrees(_wrap_angle_rad(float(yaw - expected_yaw_rad))))
                    objective = (
                        (1.00 * mean_distance)
                        + (0.28 * translation_mag)
                        + (0.018 * yaw_error_deg)
                        - (0.95 * overlap)
                    )
                    candidate = {
                        "x": x,
                        "y": y,
                        "yaw": float(yaw),
                        "yaw_deg": float(math.degrees(float(yaw))),
                        "score": float(score),
                        "overlap_fraction": float(overlap),
                        "mean_distance_m": float(mean_distance),
                        "yaw_error_deg": float(yaw_error_deg),
                        "translation_mag_m": float(translation_mag),
                        "objective": float(objective),
                    }
                    if best is None or (
                        candidate["objective"] < best["objective"] - 1e-9
                        or (
                            abs(candidate["objective"] - best["objective"]) <= 1e-9
                            and candidate["yaw_error_deg"] < best["yaw_error_deg"]
                        )
                    ):
                        best = candidate
                        best_x = x
                        best_y = y
                        best_yaw = yaw
    if best is None:
        raise RuntimeError("Refinement failed to produce a candidate")
    return best


def _solve_for_direction(
    *,
    snapshots: list[Snapshot],
    expected_step_deg: float,
    overlap_radius_m: float,
    direction_sign: float,
) -> tuple[float, list[Pose2D], list[np.ndarray], list[dict[str, Any]]]:
    poses: list[Pose2D] = [Pose2D(0.0, 0.0, 0.0)]
    transformed_snapshots: list[np.ndarray] = [snapshots[0].local_points.astype(np.float32, copy=False)]
    reports: list[dict[str, Any]] = [
        {
            "request_index": int(snapshots[0].request_index),
            "mode": "anchor",
            "pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
            "local_points": int(len(snapshots[0].local_points)),
            "integrated_points": int(len(transformed_snapshots[0])),
        }
    ]
    global_points = transformed_snapshots[0].astype(np.float32, copy=True)
    total_objective = 0.0

    for idx in range(1, len(snapshots)):
        snapshot = snapshots[idx]
        expected_yaw_deg = float(direction_sign * idx * expected_step_deg)
        expected_yaw_rad = math.radians(expected_yaw_deg)
        source_points = snapshot.local_points.astype(np.float32, copy=False)
        coarse_candidates = _coarse_candidate_search(
            source_points=source_points,
            reference_points=global_points,
            x_center=0.0,
            y_center=0.0,
            expected_yaw_rad=float(expected_yaw_rad),
            trans_span_m=0.18,
            trans_step_m=0.03,
            yaw_span_deg=10.0,
            yaw_step_deg=2.0,
            cell_voxel_m=0.05,
            cell_margin=1,
            overlap_radius_m=float(overlap_radius_m),
            top_k=4,
        )
        refined_candidates: list[dict[str, Any]] = []
        for coarse in coarse_candidates[:4]:
            refined = _refine_candidate(
                source_points=source_points,
                reference_points=global_points,
                seed=coarse,
                expected_yaw_rad=float(expected_yaw_rad),
                overlap_radius_m=float(overlap_radius_m),
            )
            refined["coarse_seed"] = {
                "x": float(coarse["x"]),
                "y": float(coarse["y"]),
                "yaw_deg": float(coarse["yaw_deg"]),
                "objective": float(coarse["objective"]),
                "coarse_overlap_fraction": float(coarse["coarse_overlap_fraction"]),
                "overlap_fraction": float(coarse["overlap_fraction"]),
            }
            refined_candidates.append(refined)
        refined_candidates.sort(
            key=lambda item: (
                item["objective"],
                item["yaw_error_deg"],
                item["translation_mag_m"],
                -item["overlap_fraction"],
            )
        )
        best = refined_candidates[0]
        accepted_pose = Pose2D(x=float(best["x"]), y=float(best["y"]), yaw=float(best["yaw"]))
        accepted_world = _transform_points_2d(
            source_points,
            x=float(accepted_pose.x),
            y=float(accepted_pose.y),
            yaw=float(accepted_pose.yaw),
        )
        accepted_world = _voxelize_points(accepted_world, 0.02)
        global_points = _voxelize_points(np.vstack((global_points, accepted_world)), 0.02)
        poses.append(accepted_pose)
        transformed_snapshots.append(accepted_world)
        total_objective += float(best["objective"])
        reports.append(
            {
                "request_index": int(snapshot.request_index),
                "expected_yaw_deg": float(expected_yaw_deg),
                "chosen_pose": {
                    "x": float(accepted_pose.x),
                    "y": float(accepted_pose.y),
                    "yaw_deg": float(math.degrees(float(accepted_pose.yaw))),
                },
                "fit": {
                    "score": float(best["score"]),
                    "overlap_fraction": float(best["overlap_fraction"]),
                    "mean_distance_m": float(best["mean_distance_m"]),
                    "yaw_error_deg": float(best["yaw_error_deg"]),
                    "translation_mag_m": float(best["translation_mag_m"]),
                    "objective": float(best["objective"]),
                },
                "local_points": int(len(source_points)),
                "integrated_points": int(len(accepted_world)),
                "global_points_after": int(len(global_points)),
                "top_coarse_candidates": coarse_candidates[:6],
                "top_refined_candidates": refined_candidates[:6],
            }
        )
    return float(total_objective), poses, transformed_snapshots, reports


def _solve_snapshot_sequence(*, snapshots: list[Snapshot], expected_step_deg: float, overlap_radius_m: float) -> tuple[str, list[Pose2D], list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    pos_total, pos_poses, pos_transformed, pos_reports = _solve_for_direction(
        snapshots=snapshots,
        expected_step_deg=float(expected_step_deg),
        overlap_radius_m=float(overlap_radius_m),
        direction_sign=1.0,
    )
    neg_total, neg_poses, neg_transformed, neg_reports = _solve_for_direction(
        snapshots=snapshots,
        expected_step_deg=float(expected_step_deg),
        overlap_radius_m=float(overlap_radius_m),
        direction_sign=-1.0,
    )
    direction_report = {
        "positive_ccw_total_objective": float(pos_total),
        "negative_cw_total_objective": float(neg_total),
    }
    if pos_total <= neg_total:
        return "ccw", pos_poses, pos_transformed, pos_reports, direction_report
    return "cw", neg_poses, neg_transformed, neg_reports, direction_report


def _render_svg(*, transformed_snapshots: list[np.ndarray], poses: list[Pose2D], output_path: Path, title: str) -> None:
    canvas_size = 1400
    palette = ["#ffd76e", "#5ac8ff", "#78ff8c", "#ff78dc", "#ffa050", "#a078ff"]
    all_points: list[np.ndarray] = [np.asarray([[0.0, 0.0]], dtype=np.float32)]
    for points in transformed_snapshots:
        if points.size != 0:
            all_points.append(points.astype(np.float32, copy=False))
    merged = np.vstack(all_points)
    min_xy = np.min(merged, axis=0)
    max_xy = np.max(merged, axis=0)
    center = (min_xy + max_xy) * 0.5
    span = max(float(np.max(max_xy - min_xy)), 1.0)
    half = (span * 0.5) + 0.35

    def world_to_px(points_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        normalized = (points_xy - center) / (2.0 * half)
        px = ((normalized[:, 0] + 0.5) * (canvas_size - 1)).astype(np.int32)
        py = ((0.5 - normalized[:, 1]) * (canvas_size - 1)).astype(np.int32)
        return np.column_stack((px, py)).astype(np.int32, copy=False)

    lines: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{canvas_size}' height='{canvas_size}' viewBox='0 0 {canvas_size} {canvas_size}'>",
        f"<rect x='0' y='0' width='{canvas_size}' height='{canvas_size}' fill='#090909' />",
    ]
    for frac in np.linspace(0.1, 0.9, 9):
        x_px = int(frac * (canvas_size - 1))
        y_px = int(frac * (canvas_size - 1))
        lines.append(f"<line x1='{x_px}' y1='0' x2='{x_px}' y2='{canvas_size - 1}' stroke='#2a2a2a' stroke-width='1' />")
        lines.append(f"<line x1='0' y1='{y_px}' x2='{canvas_size - 1}' y2='{y_px}' stroke='#2a2a2a' stroke-width='1' />")

    for index, points in enumerate(transformed_snapshots):
        points_px = world_to_px(points)
        color = palette[index % len(palette)]
        for x_px, y_px in points_px.tolist():
            lines.append(f"<circle cx='{int(x_px)}' cy='{int(y_px)}' r='2' fill='{color}' />")

    origin_px = world_to_px(np.asarray([[0.0, 0.0]], dtype=np.float32))
    if len(origin_px):
        x0, y0 = origin_px[0].tolist()
        lines.append(f"<circle cx='{int(x0)}' cy='{int(y0)}' r='8' fill='none' stroke='#ffffff' stroke-width='2' />")

    for index, pose in enumerate(poses):
        center_px = world_to_px(np.asarray([[float(pose.x), float(pose.y)]], dtype=np.float32))
        if not len(center_px):
            continue
        x0, y0 = center_px[0].tolist()
        color = palette[index % len(palette)]
        lines.append(f"<circle cx='{int(x0)}' cy='{int(y0)}' r='7' fill='none' stroke='{color}' stroke-width='2' />")
        tip = np.asarray([[float(pose.x) + (0.28 * math.cos(float(pose.yaw))), float(pose.y) + (0.28 * math.sin(float(pose.yaw)))]], dtype=np.float32)
        tip_px = world_to_px(tip)
        if len(tip_px):
            xt, yt = tip_px[0].tolist()
            lines.append(f"<line x1='{int(x0)}' y1='{int(y0)}' x2='{int(xt)}' y2='{int(yt)}' stroke='{color}' stroke-width='2' />")
            lines.append(f"<circle cx='{int(xt)}' cy='{int(yt)}' r='3' fill='{color}' />")
        lines.append(f"<text x='{int(x0) + 10}' y='{int(y0) - 10}' font-family='monospace' font-size='20' fill='{color}'>{index + 1}</text>")

    lines.append(f"<text x='28' y='44' font-family='monospace' font-size='28' fill='#ebebeb'>{html.escape(title)}</text>")
    lines.append("<text x='28' y='78' font-family='monospace' font-size='18' fill='#b4b4b4'>different colors = snapshots in capture order; arrows = solved sensor poses</text>")
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(svg_path: Path, report_path: Path, output_path: Path) -> None:
    svg_markup = svg_path.read_text(encoding="utf-8")
    report_markup = html.escape(report_path.read_text(encoding="utf-8"))
    content = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <title>Sourccey Offline Snapshot Stitch</title>
  <style>
    body {{ background:#0a0c10; color:#eef2f7; font-family:Segoe UI, Arial, sans-serif; margin:0; padding:24px; }}
    h1 {{ margin:0 0 8px 0; font-size:36px; }}
    p {{ color:#bfc8d4; }}
    .panel {{ background:#0f141b; border:1px solid #223040; border-radius:16px; padding:16px; margin-top:18px; }}
    pre {{ overflow:auto; white-space:pre-wrap; word-break:break-word; font-family:Consolas, monospace; color:#c8d2df; }}
    svg {{ width:min(92vw, 1200px); height:auto; display:block; }}
  </style>
</head>
<body>
  <h1>Sourccey Offline Snapshot Stitch</h1>
  <p>SVG overlay from the saved manual LiDAR snapshots. Different colors correspond to capture order.</p>
  <div class='panel'>{svg_markup}</div>
  <div class='panel'>
    <h2>Stitch Report</h2>
    <pre>{report_markup}</pre>
  </div>
</body>
</html>
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline stitcher for Sourccey manual LiDAR snapshots.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(r"C:\Users\Theor\Documents\WebsiteCode\VulcanSlam\dimos-vulcan\assets\output\sourccey_manual_snapshots"),
        help="Directory containing snapshot_###.json + local/world NPY files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save stitched previews/reports. Defaults to <snapshot-dir>/offline_stitch.",
    )
    parser.add_argument("--expected-step-deg", type=float, default=90.0)
    parser.add_argument("--overlap-radius-m", type=float, default=0.12)
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else snapshot_dir / "offline_stitch"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = _load_snapshots(snapshot_dir)
    if not snapshots:
        raise SystemExit(f"No snapshots found in {snapshot_dir}")

    print(f"Loaded {len(snapshots)} snapshots from {snapshot_dir}")
    direction_label, poses, transformed_snapshots, reports, direction_report = _solve_snapshot_sequence(
        snapshots=snapshots,
        expected_step_deg=float(args.expected_step_deg),
        overlap_radius_m=float(args.overlap_radius_m),
    )

    svg_path = output_dir / "latest_stitched_overlay.svg"
    html_path = output_dir / "latest_stitched_overlay.html"
    report_path = output_dir / "latest_stitch_report.json"
    points_path = output_dir / "latest_stitched_points.npy"

    merged_world = _voxelize_points(np.vstack(transformed_snapshots), 0.02)
    np.save(points_path, merged_world.astype(np.float32, copy=False))
    _render_svg(
        transformed_snapshots=transformed_snapshots,
        poses=poses,
        output_path=svg_path,
        title=f"Sourccey Offline Snapshot Stitch ({direction_label})",
    )
    report = {
        "schema": "sourccey.offline_snapshot_stitch.v3",
        "snapshot_dir": str(snapshot_dir),
        "output_dir": str(output_dir),
        "parameters": {
            "expected_step_deg": float(args.expected_step_deg),
            "overlap_radius_m": float(args.overlap_radius_m),
        },
        "chosen_direction": direction_label,
        "direction_search": direction_report,
        "snapshot_count": int(len(snapshots)),
        "poses": [
            {
                "snapshot_index": int(index + 1),
                "request_index": int(snapshots[index].request_index),
                "x": float(pose.x),
                "y": float(pose.y),
                "yaw_deg": float(math.degrees(float(pose.yaw))),
            }
            for index, pose in enumerate(poses)
        ],
        "steps": reports,
        "svg_path": str(svg_path),
        "html_path": str(html_path),
        "stitched_points_npy_path": str(points_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_html(svg_path, report_path, html_path)

    print(f"Chosen direction: {direction_label}")
    print(f"Saved overlay SVG to {svg_path}")
    print(f"Saved stitched points to {points_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved browser preview to {html_path}")
    for index, pose in enumerate(poses):
        print(f"snapshot {index + 1}: x={pose.x:.3f} y={pose.y:.3f} yaw_deg={math.degrees(float(pose.yaw)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
