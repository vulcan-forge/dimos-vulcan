from __future__ import annotations

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
    },
    "max_hz": {
        "world/color_image": 0,
        "world/companion_image": 0,
        "world/bottom_image": 0,
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
