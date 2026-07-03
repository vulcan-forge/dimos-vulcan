from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.manual_snapshot_mapper import SourcceyManualSnapshotMapper
from dimos.robot.diy.sourccey.manual_snapshot_trigger import SourcceyManualSnapshotTrigger
from dimos.robot.diy.sourccey.occupancy_grid_exporter import SourcceyOccupancyGridExporter
from dimos.robot.diy.sourccey.lidar_scan_publisher import SourcceyLidarScanPublisher


sourccey_lidar_manual_snapshot = autoconnect(
    sourccey_basic,
    SourcceyLidarScanPublisher.blueprint(
        source="tcp",
        port=8765,
    ),
    SourcceyManualSnapshotTrigger.blueprint(),
    SourcceyManualSnapshotMapper.blueprint(
        forward_angle_deg=270.0,
        valid_angle_half_width_deg=90.0,
        invert_lateral_axis=True,
        lidar_mount_x_m=0.2286,
        lidar_mount_y_m=0.0,
        min_range_m=0.20,
        max_distance_m=8.0,
        min_confidence=5,
        scan_match_enabled=True,
        scan_match_max_points=220,
        scan_match_translation_window_m=2.0,
        scan_match_rotation_window_deg=180.0,
        scan_match_accept_score_m=0.12,
        scan_match_overlap_radius_m=0.10,
        scan_match_min_overlap_fraction=0.10,
        submap_max_points=2600,
        submap_local_radius_m=8.0,
        stationary_keyframe_voxel_m=0.03,
        stationary_keyframe_max_points=420,
        obstacle_memory_enabled=True,
        obstacle_memory_decay_s=180.0,
        obstacle_memory_max_points=2500,
        obstacle_memory_voxel_m=0.05,
        obstacle_memory_height_m=0.32,
        obstacle_memory_visible_clear_radius_m=0.12,
        recent_scans_for_snapshot=1,
        cmd_vel_gate_enabled=False,
    ),
    SourcceyOccupancyGridExporter.blueprint(),
).remappings(
    [
        (SourcceyOccupancyGridExporter, "odom", "localized_pose"),
        (WebsocketVisModule, "snapshot_request", "snapshot_request"),
        (WebsocketVisModule, "reset_request", "reset_request"),
    ]
).global_config(n_workers=6)
