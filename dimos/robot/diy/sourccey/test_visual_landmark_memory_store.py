from __future__ import annotations

import numpy as np

from dimos.robot.diy.sourccey.visual_landmark_memory import (
    LandmarkRecord,
    load_landmark_records,
    save_landmark_records,
)


def test_landmark_store_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store_path = tmp_path / "landmarks.pkl"
    records = [
        LandmarkRecord(
            name="desk",
            room="office",
            note="front corner",
            ts=1.23,
            pose=(1.0, 2.0, 0.0),
            yaw_rad=0.5,
            descriptors=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            preview_path="desk.jpg",
        )
    ]

    save_landmark_records(store_path, records)
    loaded = load_landmark_records(store_path)

    assert len(loaded) == 1
    assert loaded[0]["name"] == "desk"
    assert loaded[0]["room"] == "office"
    assert np.array_equal(loaded[0]["descriptors"], records[0]["descriptors"])
