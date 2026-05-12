from __future__ import annotations

from frame_extractor.config import load_default_config
from frame_extractor.config import parse_config
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import decide_trigger
from frame_extractor.tracking import original_px_to_processing_px


def test_default_config_loads_current_yaml_values() -> None:
    config = load_default_config()

    assert config.n_downsample == 2
    assert config.dis.preset == "ultrafast"
    assert config.sampling.grid_step_original_px == 160
    assert config.trigger.main_threshold_original_px == 400.0


def test_original_pixel_values_are_converted_to_processing_pixels() -> None:
    config = parse_config({"n_downsample": 2, "max_step_norm_original_px": 100.0})

    assert original_px_to_processing_px(config.max_step_norm_original_px, config) == 25.0


def test_string_boolean_values_are_parsed_explicitly() -> None:
    config = parse_config({"visualization": {"save_debug_video": "false"}})

    assert not config.visualization.save_debug_video


def test_motion_trigger_uses_minimum_keyframe_age() -> None:
    config = load_default_config()
    scores = FrameScores(
        frame_index=10,
        timestamp_sec=0.4,
        global_score=config.trigger.main_threshold_original_px,
        in_bounds_points=100,
        in_bounds_ratio=1.0,
    )

    early = decide_trigger(scores, config.trigger.min_frames_since_keyframe - 1, config.trigger)
    ready = decide_trigger(scores, config.trigger.min_frames_since_keyframe, config.trigger)

    assert not early.triggered
    assert ready.triggered
    assert ready.display_reason == "motion"


def test_points_trigger_reports_points_reason() -> None:
    config = load_default_config()
    scores = FrameScores(
        frame_index=10,
        timestamp_sec=0.4,
        global_score=0.0,
        in_bounds_points=10,
        in_bounds_ratio=config.trigger.min_in_bounds_ratio - 0.01,
    )

    decision = decide_trigger(scores, config.trigger.min_frames_since_keyframe, config.trigger)

    assert decision.triggered
    assert decision.display_reason == "points"
