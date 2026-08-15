from __future__ import annotations

import numpy as np

from frame_extractor.config import FrameExtractorConfig
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import TriggerDecision


def comparison_label(value: float, threshold: float) -> str:
    if np.isclose(value, threshold):
        relation = "="
    elif value < threshold:
        relation = "<"
    else:
        relation = ">"
    return f"{value:.2f} {relation} {threshold:.2f}"


def tracking_status_lines(
    frame_scores: FrameScores,
    trigger_decision: TriggerDecision,
    *,
    config: FrameExtractorConfig,
    keyframe_count: int,
) -> list[str]:
    return [
        f"frame: {frame_scores.frame_index}    time: {frame_scores.timestamp_sec:.3f}s",
        f"frames since last trigger: {trigger_decision.frames_since_keyframe}",
        f"number of keyframes: {keyframe_count}",
        (
            "motion: "
            f"{comparison_label(frame_scores.global_score, config.trigger.main_threshold_original_px)}"
        ),
        (
            "points: "
            f"{comparison_label(frame_scores.in_bounds_ratio, config.trigger.min_in_bounds_ratio)}"
        ),
        f"trigger: {trigger_decision.display_reason}",
    ]
