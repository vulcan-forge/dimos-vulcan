from __future__ import annotations

import numpy as np

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.diy.sourccey.connection import SourcceyConnection
from dimos.visualization.vis_module import vis_module


def _convert_camera_info(camera_info):  # type: ignore[no-untyped-def]
    return camera_info.to_rerun(
        image_topic="/world/color_image",
        optical_frame="camera_optical",
    )


def _static_base_link(rr):  # type: ignore[no-untyped-def]
    return [
        rr.Boxes3D(
            half_sizes=[0.20, 0.16, 0.18],
            colors=[(0, 191, 255)],
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _global_map_colors(cloud):  # type: ignore[no-untyped-def]
    import rerun as rr

    points = cloud.points_f32()
    if len(points) == 0:
        return rr.Points3D([])

    intensities = cloud.intensities_f32()
    if intensities is not None and len(intensities) == len(points):
        # Free-space samples are emitted with intensity 0, obstacle endpoints with sensor confidence.
        obstacle_mask = intensities > 0.0
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[~obstacle_mask] = np.array([80, 120, 220], dtype=np.uint8)
        colors[obstacle_mask] = np.array([255, 80, 80], dtype=np.uint8)
        radii = np.where(obstacle_mask, 0.04, 0.022).astype(np.float32)
        return rr.Points3D(positions=points[:, :3], colors=colors, radii=radii)

    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[:, 0] = (30 + z_norm * 30).astype(np.uint8)
    colors[:, 1] = (80 + z_norm * 140).astype(np.uint8)
    colors[:, 2] = (200 - z_norm * 100).astype(np.uint8)
    return rr.Points3D(positions=points[:, :3], colors=colors, radii=0.03)


def _registered_scan_colors(cloud):  # type: ignore[no-untyped-def]
    import rerun as rr

    points = cloud.points_f32()
    if len(points) == 0:
        return rr.Points3D([])
    colors = np.full((len(points), 3), [255, 240, 180], dtype=np.uint8)
    return rr.Points3D(positions=points[:, :3], colors=colors, radii=0.055)


def _global_costmap_mesh(grid):  # type: ignore[no-untyped-def]
    return grid.to_rerun(
        background="#0d0f14",
        opacity=0.92,
        z_offset=0.015,
    )


def _sourccey_rerun_blueprint():  # type: ignore[no-untyped-def]
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/color_image", name="Primary"),
                rrb.Spatial2DView(origin="world/companion_image", name="Companion"),
                rrb.Spatial2DView(origin="world/bottom_image", name="Bottom"),
            ),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.0),
                ),
            ),
            column_shares=[1, 2],
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


rerun_config = {
    "blueprint": _sourccey_rerun_blueprint,
    "visual_override": {
        "world/camera_info": _convert_camera_info,
        "world/global_map": _global_map_colors,
        "world/registered_scan": _registered_scan_colors,
        "world/global_costmap": _global_costmap_mesh,
    },
    "max_hz": {
        "world/color_image": 0,
        "world/companion_image": 0,
        "world/bottom_image": 0,
        "world/global_map": 12,
        "world/registered_scan": 12,
        "world/global_costmap": 8,
    },
    "static": {
        "world/tf/base_link": _static_base_link,
    },
}


sourccey_basic = autoconnect(
    vis_module(
        viewer_backend=global_config.viewer,
        rerun_config=rerun_config,
    ),
    SourcceyConnection.blueprint(),
).global_config(
    n_workers=4,
    robot_model="sourccey",
)
