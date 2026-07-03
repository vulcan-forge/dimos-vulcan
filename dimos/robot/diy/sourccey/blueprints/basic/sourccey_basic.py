from __future__ import annotations

import numpy as np

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.diy.sourccey.connection import SourcceyConnection
from dimos.visualization.vis_module import vis_module


def _viewer_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indices]


def _viewer_downsample_indices(length: int, max_points: int) -> np.ndarray | None:
    if length <= max_points:
        return None
    return np.linspace(0, length - 1, max_points, dtype=np.int32)


def _convert_camera_info(camera_info):  # type: ignore[no-untyped-def]
    return camera_info.to_rerun(
        image_topic="/world/color_image",
        optical_frame="camera_optical",
    )


def _static_base_link(rr):  # type: ignore[no-untyped-def]
    vision_radius = 0.52
    vision_z = 0.06
    # Sourccey's practical "forward" in the viewer frame is +Y, not +X.
    # Rotate the cone 90 degrees CCW relative to the canonical Rerun arrow axis
    # so the displayed heading matches the room map the user sees.
    vision_angles = np.linspace(0.0, np.pi, 25, dtype=np.float32)
    arc_strip = np.column_stack(
        (
            vision_radius * np.cos(vision_angles),
            vision_radius * np.sin(vision_angles),
            np.full_like(vision_angles, vision_z),
        )
    ).astype(np.float32)
    center_ray = np.asarray(
        [[0.0, 0.0, vision_z], [0.0, vision_radius, vision_z]],
        dtype=np.float32,
    )
    left_ray = np.asarray(
        [[0.0, 0.0, vision_z], [-vision_radius, 0.0, vision_z]],
        dtype=np.float32,
    )
    right_ray = np.asarray(
        [[0.0, 0.0, vision_z], [vision_radius, 0.0, vision_z]],
        dtype=np.float32,
    )
    return [
        rr.Boxes3D(
            half_sizes=[0.20, 0.16, 0.18],
            colors=[(0, 191, 255)],
        ),
        rr.Arrows3D(
            origins=np.asarray(
                [
                    [0.0, 0.0, 0.08],
                    [0.0, 0.0, 0.08],
                    [0.0, 0.0, 0.08],
                ],
                dtype=np.float32,
            ),
            vectors=np.asarray(
                [
                    [0.0, 0.42, 0.0],
                    [-0.18, 0.32, 0.0],
                    [0.18, 0.32, 0.0],
                ],
                dtype=np.float32,
            ),
            colors=np.asarray(
                [
                    [0, 255, 255],
                    [0, 210, 255],
                    [0, 210, 255],
                ],
                dtype=np.uint8,
            ),
            radii=np.asarray([0.012, 0.008, 0.008], dtype=np.float32),
        ),
        rr.LineStrips3D(
            strips=[arc_strip, center_ray, left_ray, right_ray],
            colors=[
                [0, 255, 255],
                [0, 255, 255],
                [0, 190, 255],
                [0, 190, 255],
            ],
            radii=[0.01, 0.006, 0.005, 0.005],
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _global_map_colors(cloud):  # type: ignore[no-untyped-def]
    import rerun as rr

    points = cloud.points_f32()
    if len(points) == 0:
        return rr.Clear(recursive=False)
    intensities = cloud.intensities_f32()
    if intensities is not None and len(intensities) == len(points):
        obstacle_mask = intensities > 0.0
    else:
        obstacle_mask = points[:, 2] > 0.16

    free_points = points[~obstacle_mask]
    obstacle_points = points[obstacle_mask]

    # Keep free space present as a lightweight floor footprint, but spend most
    # of the visual detail budget on actual geometry and obstacle structure.
    free_points = _viewer_downsample(free_points, 5000)
    obstacle_points = _viewer_downsample(obstacle_points, 7000)

    stacked_points: list[np.ndarray] = []
    stacked_colors: list[np.ndarray] = []
    stacked_radii: list[np.ndarray] = []
    if len(free_points) != 0:
        stacked_points.append(free_points[:, :3])
        stacked_colors.append(
            np.full((len(free_points), 3), [90, 150, 255], dtype=np.uint8)
        )
        stacked_radii.append(np.full((len(free_points),), 0.032, dtype=np.float32))
    if len(obstacle_points) != 0:
        stacked_points.append(obstacle_points[:, :3])
        # Reserve bright green for obstacle-memory points only; the global map's
        # own occupied voxels use a lighter blue so "green" keeps the semantic
        # meaning of "previously seen obstacle that is currently out of sight".
        stacked_colors.append(
            np.full((len(obstacle_points), 3), [170, 210, 255], dtype=np.uint8)
        )
        stacked_radii.append(np.full((len(obstacle_points),), 0.03, dtype=np.float32))

    if not stacked_points:
        return rr.Points3D([])
    return rr.Points3D(
        positions=np.vstack(stacked_points),
        colors=np.vstack(stacked_colors),
        radii=np.concatenate(stacked_radii),
    )


def _registered_scan_colors(cloud):  # type: ignore[no-untyped-def]
    import rerun as rr

    points = cloud.points_f32()
    if len(points) == 0:
        return rr.Clear(recursive=False)
    points = _viewer_downsample(points, 2500)
    colors = np.full((len(points), 3), [255, 240, 180], dtype=np.uint8)
    return rr.Points3D(positions=points[:, :3], colors=colors, radii=0.055)


def _obstacle_memory_colors(cloud):  # type: ignore[no-untyped-def]
    import rerun as rr

    points = cloud.points_f32()
    if len(points) == 0:
        return rr.Clear(recursive=False)
    points = _viewer_downsample(points, 3000)
    colors = np.full((len(points), 3), [0, 255, 70], dtype=np.uint8)
    return rr.Points3D(positions=points[:, :3], colors=colors, radii=0.055)


def _local_map_colors(cloud):  # type: ignore[no-untyped-def]
    import rerun as rr

    points = cloud.points_f32()
    if len(points) == 0:
        return rr.Clear(recursive=False)
    downsample_indices = _viewer_downsample_indices(len(points), 8000)
    if downsample_indices is not None:
        points = points[downsample_indices]

    intensities = cloud.intensities_f32()
    if intensities is not None and downsample_indices is not None:
        intensities = intensities[downsample_indices]
    if intensities is not None and len(intensities) == len(points):
        obstacle_mask = intensities > 0.0
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[~obstacle_mask] = np.array([70, 150, 255], dtype=np.uint8)
        colors[obstacle_mask] = np.array([255, 235, 120], dtype=np.uint8)
        radii = np.where(obstacle_mask, 0.05, 0.03).astype(np.float32)
        return rr.Points3D(positions=points[:, :3], colors=colors, radii=radii)

    colors = np.full((len(points), 3), [100, 180, 255], dtype=np.uint8)
    return rr.Points3D(positions=points[:, :3], colors=colors, radii=0.035)


def _global_costmap_mesh(grid):  # type: ignore[no-untyped-def]
    # The native OccupancyGrid default renderer uses a dark purple free-space
    # palette. When the point-cloud layer briefly drops out during an offboard
    # mapping run, that fallback made the map look "empty" even though free
    # space was still present. Force the costmap layer itself to carry the same
    # blue semantics as the point-cloud view so the room footprint is always
    # visible.
    color_lookup_table = np.zeros((102, 4), dtype=np.uint8)
    color_lookup_table[0] = np.array([0, 0, 0, 255], dtype=np.uint8)
    color_lookup_table[1] = np.array([90, 150, 255, 255], dtype=np.uint8)
    color_lookup_table[2:102] = np.array([125, 185, 255, 255], dtype=np.uint8)
    return grid.to_rerun(
        color_lookup_table=color_lookup_table,
        background="#000000",
        z_offset=0.01,
    )


def _sourccey_rerun_blueprint():  # type: ignore[no-untyped-def]
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            name="3D",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            overrides={
                # The occupancy mesh is useful as a fallback, but in practice it
                # can visually dominate the room and read as a purple fan even
                # when the point-based room map is correct. Keep the point cloud
                # and obstacle layers as the default view while we debug mapping.
                "world/global_costmap": rrb.EntityBehavior(visible=False),
            },
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


rerun_config = {
    "blueprint": _sourccey_rerun_blueprint,
    "visual_override": {
        "world/camera_info": _convert_camera_info,
        "world/color_image": None,
        "world/companion_image": None,
        "world/bottom_image": None,
        "world/lidar": _local_map_colors,
        "world/local_map": _local_map_colors,
        "world/global_map": _global_map_colors,
        "world/registered_scan": _registered_scan_colors,
        "world/obstacle_memory": _obstacle_memory_colors,
        "world/global_costmap": _global_costmap_mesh,
    },
    "max_hz": {
        "world/lidar": 12,
        "world/global_map": 8,
        "world/global_costmap": 8,
        "world/local_map": 12,
        "world/registered_scan": 12,
        "world/obstacle_memory": 12,
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
