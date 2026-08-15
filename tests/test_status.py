from __future__ import annotations

import pytest

from frame_extractor.config import load_default_config
from frame_extractor.status import comparison_label
from frame_extractor.status import tracking_status_lines
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import TriggerDecision


@pytest.mark.parametrize(
    ("value", "threshold", "expected"),
    [
        (1.0, 2.0, "1.00 < 2.00"),
        (2.0, 2.0, "2.00 = 2.00"),
        (3.0, 2.0, "3.00 > 2.00"),
    ],
)
def test_comparison_label(value: float, threshold: float, expected: str) -> None:
    assert comparison_label(value, threshold) == expected


def test_tracking_status_lines_are_complete_and_ordered() -> None:
    config = load_default_config()
    frame_scores = FrameScores(
        frame_index=42,
        timestamp_sec=1.25,
        global_score=config.trigger.main_threshold_original_px + 1.0,
        in_bounds_points=10,
        in_bounds_ratio=config.trigger.min_in_bounds_ratio,
    )
    trigger_decision = TriggerDecision(
        triggered=True,
        reason="main+in_bounds",
        frames_since_keyframe=7,
    )

    assert tracking_status_lines(
        frame_scores,
        trigger_decision,
        config=config,
        keyframe_count=3,
    ) == [
        "frame: 42    time: 1.250s",
        "frames since last trigger: 7",
        "number of keyframes: 3",
        (
            "motion: "
            f"{config.trigger.main_threshold_original_px + 1.0:.2f} > "
            f"{config.trigger.main_threshold_original_px:.2f}"
        ),
        (
            "points: "
            f"{config.trigger.min_in_bounds_ratio:.2f} = "
            f"{config.trigger.min_in_bounds_ratio:.2f}"
        ),
        "trigger: motion+low_points",
    ]
