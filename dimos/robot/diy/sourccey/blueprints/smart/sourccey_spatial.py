from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.sourccey.blueprints.basic.sourccey_basic import sourccey_basic
from dimos.robot.diy.sourccey.connection import SourcceyConnection

_spatial_import_error: Exception | None = None

try:
    from dimos.perception.spatial_perception import SpatialMemory
except Exception as exc:
    _spatial_import_error = exc


def _require_spatial_dependencies() -> str | None:
    if _spatial_import_error is None:
        return None
    return (
        "sourccey-spatial requires optional dimos perception/agent dependencies "
        f"that are not currently installed: {_spatial_import_error}"
    )


if _spatial_import_error is None:
    sourccey_spatial = autoconnect(
        sourccey_basic,
        SourcceyConnection.blueprint(publish_mosaic_as_color_image=False),
        SpatialMemory.blueprint(
            embedding_model="mobileclip",
            min_distance_threshold=0.01,
            min_rotation_threshold_deg=5.0,
            min_time_threshold=0.75,
        ),
    ).global_config(n_workers=6)
else:
    sourccey_spatial = autoconnect(
        sourccey_basic,
    ).requirements(_require_spatial_dependencies)
