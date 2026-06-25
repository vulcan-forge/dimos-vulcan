from __future__ import annotations

import argparse
import json

from dimos.robot.diy.sourccey.spatial_artifacts import (
    export_session,
    reconstruct_session,
    resolve_latest_view,
)


def _cmd_export(args: argparse.Namespace) -> int:
    payload = export_session(session=args.session)
    print(
        "Sourccey Spatial Export: OK "
        f"session={payload['session_dir']} manifest={payload['manifest_path']} "
        f"frames={payload['frame_count']} export={payload['export_path']}"
    )
    return 0


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    payload = reconstruct_session(
        session=args.session,
        output_dir=args.output_dir,
        min_matches=args.min_matches,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        max_pair_range_m=args.max_pair_range_m,
        reprojection_error_px=args.reprojection_error_px,
        voxel_size_m=args.voxel_size_m,
    )
    print(
        "Sourccey Spatial Reconstruct: OK "
        f"points={payload['point_count']} pairs={payload['used_pairs']} "
        f"npz={payload['outputs']['npz']} ply={payload['outputs']['ply']} html={payload['outputs']['html']}"
    )
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    html_path = resolve_latest_view(output_dir=args.output_dir)
    print(json.dumps({"latest_html": str(html_path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sourccey spatial session export and reconstruction tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export/inspect the latest captured spatial session")
    export_parser.add_argument("--session", default="latest")
    export_parser.set_defaults(handler=_cmd_export)

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Build a sparse point cloud from a captured spatial session")
    reconstruct_parser.add_argument("--session", default="latest")
    reconstruct_parser.add_argument("--output-dir", default=None)
    reconstruct_parser.add_argument("--min-matches", type=int, default=12)
    reconstruct_parser.add_argument("--min-translation-m", type=float, default=0.005)
    reconstruct_parser.add_argument("--min-rotation-deg", type=float, default=2.0)
    reconstruct_parser.add_argument("--max-pair-range-m", type=float, default=8.0)
    reconstruct_parser.add_argument("--reprojection-error-px", type=float, default=4.0)
    reconstruct_parser.add_argument("--voxel-size-m", type=float, default=0.03)
    reconstruct_parser.set_defaults(handler=_cmd_reconstruct)

    view_parser = subparsers.add_parser("view", help="Print the latest generated spatial viewer HTML path")
    view_parser.add_argument("--output-dir", default=None)
    view_parser.set_defaults(handler=_cmd_view)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
