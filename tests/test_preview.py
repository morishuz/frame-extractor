from __future__ import annotations

from frame_extractor.preview import History
from frame_extractor.preview import PLOT_HISTORY_FRAMES
from frame_extractor.preview import history_append
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import TriggerDecision


def test_history_retains_only_the_rendered_frame_window() -> None:
    history = History()
    total_frames = PLOT_HISTORY_FRAMES + 5

    for frame_index in range(total_frames):
        history_append(
            history,
            FrameScores(
                frame_index=frame_index,
                timestamp_sec=frame_index / 30.0,
                global_score=float(frame_index),
                in_bounds_points=100 - frame_index,
                in_bounds_ratio=0.5,
            ),
            TriggerDecision(
                triggered=True,
                reason="main",
                frames_since_keyframe=1,
            ),
        )

    expected_first_frame = total_frames - PLOT_HISTORY_FRAMES
    assert len(history.frame_indices) == PLOT_HISTORY_FRAMES
    assert len(history.global_scores) == PLOT_HISTORY_FRAMES
    assert len(history.in_bounds_ratios) == PLOT_HISTORY_FRAMES
    assert len(history.trigger_frames) == PLOT_HISTORY_FRAMES
    assert len(history.trigger_reasons) == PLOT_HISTORY_FRAMES
    assert history.frame_indices[0] == expected_first_frame
    assert history.trigger_frames[0] == expected_first_frame
    assert history.frame_indices[-1] == total_frames - 1


def test_trigger_history_is_bounded_when_decoded_indices_repeat() -> None:
    history = History()

    for processed_index in range(PLOT_HISTORY_FRAMES + 5):
        history_append(
            history,
            FrameScores(
                frame_index=0,
                timestamp_sec=processed_index / 30.0,
                global_score=0.0,
                in_bounds_points=1,
                in_bounds_ratio=1.0,
            ),
            TriggerDecision(
                triggered=True,
                reason="interval",
                frames_since_keyframe=1,
            ),
        )

    assert len(history.trigger_frames) == PLOT_HISTORY_FRAMES
    assert len(history.trigger_reasons) == PLOT_HISTORY_FRAMES
