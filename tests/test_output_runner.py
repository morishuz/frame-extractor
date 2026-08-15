from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_extractor.config import parse_config
from frame_extractor.output import TimingValidator
from frame_extractor.output import capture_raw_frame_timing
from frame_extractor.runner import run_experiment
from frame_extractor.tracking import FlowStepDiagnostics


EXPECTED_KEYFRAME_MANIFEST_FIELDS = (
    "filename",
    "processed_index",
    "decoded_frame_index",
    "pts",
    "pos_seconds_raw",
    "timing_status",
    "selection_reason",
    "motion_score_px",
    "in_bounds_ratio",
)


class FakeVideoCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fps: float = 10.0,
        pts: list[float] | None = None,
        timestamps_seconds: list[float] | None = None,
        backend: str = "FFMPEG",
        seek_supported: bool = True,
        seek_lands_at: int | None = None,
    ) -> None:
        self.frames = frames
        self.fps = fps
        self.pts = pts if pts is not None else list(range(len(frames)))
        self.timestamps_seconds = (
            timestamps_seconds
            if timestamps_seconds is not None
            else [index / fps for index in range(len(frames))]
        )
        self.backend = backend
        self.seek_supported = seek_supported
        self.seek_lands_at = seek_lands_at
        self.next_index = 0
        self.last_read_index: int | None = None

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            if not self.seek_supported:
                return False
            self.next_index = (
                self.seek_lands_at
                if self.seek_lands_at is not None
                else int(value)
            )
            self.last_read_index = None
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.next_index >= len(self.frames):
            return False, None
        self.last_read_index = self.next_index
        frame = self.frames[self.next_index].copy()
        self.next_index += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.next_index)
        if self.last_read_index is None:
            return 0.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps_seconds[self.last_read_index] * 1000.0
        if prop == getattr(cv2, "CAP_PROP_PTS", -1):
            return float(self.pts[self.last_read_index])
        return 0.0

    def getBackendName(self) -> str:
        return self.backend

    def release(self) -> None:
        pass


class SilentProgress:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def update(self, *_args: object, **_kwargs: object) -> None:
        pass

    def finish(self) -> None:
        pass


class FakeVideoWriter:
    def write(self, _frame: np.ndarray) -> None:
        pass

    def release(self) -> None:
        pass


def _frames(count: int) -> list[np.ndarray]:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return [frame.copy() for _ in range(count)]


def _config(
    *,
    max_frames_since_keyframe: int = 0,
    keyframe_thumbnail_slots: int = 8,
):
    return parse_config(
        {
            "n_downsample": 0,
            "sampling": {
                "grid_step_original_px": 16,
                "min_margin_original_px": 2,
            },
            "trigger": {
                "main_threshold_original_px": 10_000.0,
                "max_frames_since_keyframe": max_frames_since_keyframe,
            },
            "visualization": {
                "keyframe_thumbnail_slots": keyframe_thumbnail_slots,
            },
        }
    )


def _run_with_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capture: FakeVideoCapture,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    max_frames_since_keyframe: int = 0,
    keyframe_thumbnail_slots: int = 8,
    show_preview: bool = False,
):
    import frame_extractor.runner as runner

    monkeypatch.setattr(runner, "open_video", lambda _path: capture)
    monkeypatch.setattr(runner, "create_dis_flow", lambda _config: object())
    monkeypatch.setattr(
        runner,
        "step_tracking",
        lambda state, *_args: FlowStepDiagnostics(
            in_bounds_mask=np.ones(state.alive_mask.shape, dtype=bool)
        ),
    )
    monkeypatch.setattr(runner, "TerminalProgress", SilentProgress)

    return run_experiment(
        "fake.mp4",
        _config(
            max_frames_since_keyframe=max_frames_since_keyframe,
            keyframe_thumbnail_slots=keyframe_thumbnail_slots,
        ),
        output_dir=str(tmp_path),
        show_preview=show_preview,
        start_frame=start_frame,
        max_frames=max_frames,
    )


def _manifest_rows(manifest_path: str | None) -> list[dict[str, str]]:
    assert manifest_path is not None
    with Path(manifest_path).open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        assert tuple(reader.fieldnames or ()) == EXPECTED_KEYFRAME_MANIFEST_FIELDS
        return list(reader)


def _read_and_validate_timing(
    capture: FakeVideoCapture,
    validator: TimingValidator,
    processed_index: int,
):
    ok, _frame = capture.read()
    assert ok
    raw_timing = capture_raw_frame_timing(
        capture,
        reported_fps=capture.fps,
        video_backend=capture.backend,
    )
    return validator.observe(
        raw_timing,
        decoded_frame_index=processed_index,
        processed_index=processed_index,
    )


def test_manifest_records_first_and_final_frames_with_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(
        _frames(4),
        fps=10.0,
        pts=[100, 101, 103, 106],
        timestamps_seconds=[0.0, 0.1, 0.3, 0.6],
    )

    stats = _run_with_capture(monkeypatch, tmp_path, capture)
    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.keyframes_saved == 2
    assert [row["decoded_frame_index"] for row in rows] == ["0", "3"]
    assert [row["processed_index"] for row in rows] == ["0", "3"]
    assert [row["selection_reason"] for row in rows] == ["first", "final"]
    assert rows[0]["pts"] == "100.0"
    assert float(rows[1]["pos_seconds_raw"]) == pytest.approx(0.6)
    assert rows[1]["timing_status"] == "warning"
    assert rows[0]["motion_score_px"] == "0.0"

    summary = Path(stats.summary_path or "").read_text(encoding="utf-8")
    assert "keyframes_csv_schema_version: 3" in summary
    assert "opencv_version: " + cv2.__version__ in summary
    assert "video_backend: FFMPEG" in summary
    assert "reported_fps: 10.000000000" in summary
    assert "pts_time_base_num: 1" in summary
    assert "pts_time_base_den: 10" in summary
    assert "timing_validation_status: warnings" in summary
    assert "timing_flag_counts: pos_gap=2, pts_gap=2" in summary
    assert "raw_pos_timeline_valid: true" in summary

    run_dir = Path(stats.run_dir or "")
    image_paths = sorted((run_dir / "keyframes").glob("*.jpg"))
    assert len(image_paths) == stats.keyframes_saved
    assert {row["filename"] for row in rows} == {
        path.relative_to(run_dir).as_posix() for path in image_paths
    }


def test_final_frame_is_the_end_of_the_requested_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        FakeVideoCapture(_frames(6)),
        start_frame=2,
        max_frames=3,
    )

    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.processed_frames == 3
    assert [row["decoded_frame_index"] for row in rows] == ["2", "4"]
    assert [row["processed_index"] for row in rows] == ["0", "2"]


def test_failed_start_frame_seek_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(_frames(4), seek_supported=False)

    with pytest.raises(RuntimeError, match="could not seek to start frame 2"):
        _run_with_capture(monkeypatch, tmp_path, capture, start_frame=2)


def test_inexact_start_frame_seek_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(_frames(4), seek_lands_at=1)

    with pytest.raises(
        RuntimeError,
        match="seek landed on decoded frame 1; expected 2",
    ):
        _run_with_capture(monkeypatch, tmp_path, capture, start_frame=2)


@pytest.mark.parametrize(
    ("start_frame", "max_frames", "message"),
    [
        (-1, None, "start_frame must be >= 0"),
        (0, 0, "max_frames must be > 0"),
        (0, -1, "max_frames must be > 0"),
    ],
)
def test_invalid_processing_ranges_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    start_frame: int,
    max_frames: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _run_with_capture(
            monkeypatch,
            tmp_path,
            FakeVideoCapture(_frames(2)),
            start_frame=start_frame,
            max_frames=max_frames,
        )


def test_run_directories_are_unique_within_the_same_second(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import frame_extractor.output as output

    class FixedDatetime:
        @staticmethod
        def now() -> datetime:
            return datetime(2026, 8, 15, 12, 34, 56)

    monkeypatch.setattr(output, "datetime", FixedDatetime)

    first = output.make_run_paths(str(tmp_path), save_debug_video=False)
    second = output.make_run_paths(str(tmp_path), save_debug_video=False)

    assert first.run_dir.name == "20260815_123456"
    assert second.run_dir.name == "20260815_123456_01"
    assert first.run_dir != second.run_dir


def test_single_frame_is_saved_once_as_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats = _run_with_capture(monkeypatch, tmp_path, FakeVideoCapture(_frames(1)))

    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.keyframes_saved == 1
    assert rows[0]["selection_reason"] == "first+final"


def test_triggered_final_frame_is_not_saved_twice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        FakeVideoCapture(_frames(3)),
        max_frames_since_keyframe=2,
    )

    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.trigger_count == 1
    assert stats.keyframes_saved == 2
    assert rows[-1]["selection_reason"] == "interval+final"


def test_headless_run_does_not_construct_preview_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import frame_extractor.runner as runner

    def unexpected_history() -> None:
        pytest.fail("History should only be constructed for visual debug runs")

    monkeypatch.setattr(runner, "History", unexpected_history)

    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        FakeVideoCapture(_frames(3)),
        max_frames_since_keyframe=1,
    )

    assert stats.trigger_count == 2
    summary = Path(stats.summary_path or "").read_text(encoding="utf-8")
    assert "trigger_count: 2" in summary


def test_preview_retains_only_visible_keyframe_thumbnails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import frame_extractor.runner as runner

    retained_counts: list[int] = []
    displayed_keyframe_counts: list[int] = []

    monkeypatch.setattr(runner.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(runner.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(runner.cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(runner.cv2, "destroyAllWindows", lambda: None)

    def render_spy(*_args, **kwargs) -> np.ndarray:
        retained_counts.append(len(kwargs["keyframe_thumbnails"]))
        displayed_keyframe_counts.append(kwargs["keyframe_count"])
        return np.zeros((48, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(runner, "_render_preview_frame", render_spy)
    monkeypatch.setattr(runner, "create_video_writer", lambda *_args: FakeVideoWriter())

    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        FakeVideoCapture(_frames(6)),
        max_frames_since_keyframe=1,
        keyframe_thumbnail_slots=2,
        show_preview=True,
    )

    assert stats.trigger_count == 5
    assert retained_counts
    assert max(retained_counts) == 2
    assert retained_counts[-1] == 2
    assert displayed_keyframe_counts[-1] == 6


@pytest.mark.parametrize(
    ("internal_reason", "expected_selection_reason"),
    [
        ("main", "motion"),
        ("in_bounds", "low_points"),
        ("interval", "interval"),
        ("main+in_bounds", "motion+low_points"),
        ("main+interval", "motion+interval"),
        ("in_bounds+interval", "low_points+interval"),
        (
            "main+in_bounds+interval",
            "motion+low_points+interval",
        ),
    ],
)
def test_trigger_causes_are_expressed_in_selection_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    internal_reason: str,
    expected_selection_reason: str,
) -> None:
    import frame_extractor.runner as runner

    def decide_once(frame_scores, frames_since_keyframe, _config):
        if frame_scores.frame_index == 1:
            return runner.TriggerDecision(
                triggered=True,
                reason=internal_reason,
                frames_since_keyframe=frames_since_keyframe,
            )
        return runner.TriggerDecision(
            triggered=False,
            reason="none",
            frames_since_keyframe=frames_since_keyframe,
        )

    monkeypatch.setattr(runner, "decide_trigger", decide_once)

    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        FakeVideoCapture(_frames(3)),
    )
    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert [row["selection_reason"] for row in rows] == [
        "first",
        expected_selection_reason,
        "final",
    ]


def test_final_frame_deduplication_uses_processed_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import frame_extractor.runner as runner

    monkeypatch.setattr(
        runner,
        "capture_decoded_frame_index",
        lambda *_args, **_kwargs: 0,
    )

    stats = _run_with_capture(monkeypatch, tmp_path, FakeVideoCapture(_frames(4)))
    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.keyframes_saved == 2
    assert [row["processed_index"] for row in rows] == ["0", "3"]
    assert [row["decoded_frame_index"] for row in rows] == ["0", "0"]
    assert rows[-1]["selection_reason"] == "final"


def test_user_stop_saves_the_last_displayed_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import frame_extractor.runner as runner

    monkeypatch.setattr(runner.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(runner.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(runner.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(runner.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        runner,
        "_render_preview_frame",
        lambda *_args, **_kwargs: np.zeros((48, 64, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(runner, "create_video_writer", lambda *_args: FakeVideoWriter())

    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        FakeVideoCapture(_frames(4)),
        show_preview=True,
    )

    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.stopped_by_user
    assert stats.processed_frames == 2
    assert [row["decoded_frame_index"] for row in rows] == ["0", "1"]
    assert rows[-1]["selection_reason"] == "final"


def test_timing_falls_back_without_claiming_pts() -> None:
    capture = FakeVideoCapture(
        _frames(1),
        fps=30.0,
        timestamps_seconds=[1.25],
        backend="AVFOUNDATION",
    )
    validator = TimingValidator(
        reported_fps=30.0,
        fallback_fps=30.0,
    )
    timing = _read_and_validate_timing(capture, validator, 0)

    assert timing.pts is None
    assert timing.pts_time_base_num is None
    assert timing.pts_time_base_den is None
    assert timing.pts_seconds is None
    assert timing.pos_msec_raw == pytest.approx(1_250.0)
    assert timing.pos_seconds_raw == pytest.approx(1.25)
    assert timing.effective_timestamp_seconds == pytest.approx(1.25)
    assert timing.effective_timestamp_source == "opencv_pos_msec"
    assert timing.timing_status == "ok"
    assert timing.timing_flags == ()
    assert validator.report().pts_unavailable_frames == 1


def test_stuck_position_timestamp_uses_an_explicit_fallback() -> None:
    capture = FakeVideoCapture(
        _frames(2),
        fps=10.0,
        timestamps_seconds=[0.0, 0.0],
        backend="AVFOUNDATION",
    )
    validator = TimingValidator(
        reported_fps=10.0,
        fallback_fps=10.0,
    )
    first_timing = _read_and_validate_timing(capture, validator, 0)
    second_timing = _read_and_validate_timing(capture, validator, 1)

    assert first_timing.effective_timestamp_source == "opencv_pos_msec"
    assert second_timing.pos_msec_raw == 0.0
    assert second_timing.pos_seconds_raw == 0.0
    assert second_timing.effective_timestamp_seconds == pytest.approx(0.1)
    assert second_timing.effective_timestamp_source == "frame_duration_fallback"
    assert second_timing.timing_status == "invalid"
    assert second_timing.timing_flags == (
        "pos_duplicate",
        "effective_fallback",
    )
    report = validator.report()
    assert not report.raw_pos_timeline_valid
    assert report.flag_counts == (
        ("pos_duplicate", 1),
        ("effective_fallback", 1),
    )


def test_aligned_pts_fallback_stays_in_the_relative_timestamp_domain() -> None:
    capture = FakeVideoCapture(
        _frames(2),
        fps=10.0,
        pts=[100, 101],
        timestamps_seconds=[0.0, 0.0],
    )
    validator = TimingValidator(
        reported_fps=10.0,
        fallback_fps=10.0,
    )
    first_timing = _read_and_validate_timing(capture, validator, 0)
    second_timing = _read_and_validate_timing(capture, validator, 1)

    assert first_timing.effective_timestamp_seconds == 0.0
    assert second_timing.pts_seconds == pytest.approx(10.1)
    assert second_timing.pos_msec_raw == 0.0
    assert second_timing.pos_seconds_raw == 0.0
    assert second_timing.effective_timestamp_seconds == pytest.approx(0.1)
    assert second_timing.effective_timestamp_source == "opencv_pts_aligned"
    assert "pos_duplicate" in second_timing.timing_flags
    assert "pts_pos_disagreement" not in second_timing.timing_flags


def test_timing_recovers_to_raw_position_after_a_transient_duplicate() -> None:
    capture = FakeVideoCapture(
        _frames(3),
        fps=10.0,
        pts=[0, 1, 2],
        timestamps_seconds=[0.0, 0.0, 0.2],
    )
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    first = _read_and_validate_timing(capture, validator, 0)
    duplicate = _read_and_validate_timing(capture, validator, 1)
    recovered = _read_and_validate_timing(capture, validator, 2)

    assert first.effective_timestamp_source == "opencv_pos_msec"
    assert duplicate.effective_timestamp_source == "opencv_pts_aligned"
    assert duplicate.pos_seconds_raw == 0.0
    assert recovered.pos_seconds_raw == pytest.approx(0.2)
    assert recovered.effective_timestamp_seconds == pytest.approx(0.2)
    assert recovered.effective_timestamp_source == "opencv_pos_msec"
    assert "pos_gap" not in recovered.timing_flags


@pytest.mark.parametrize(
    ("raw_position_seconds", "expected_flag"),
    [
        (float("nan"), "pos_nonfinite"),
        (-0.001, "pos_negative"),
    ],
)
def test_invalid_raw_position_is_preserved_and_flagged(
    raw_position_seconds: float,
    expected_flag: str,
) -> None:
    capture = FakeVideoCapture(
        _frames(1),
        fps=10.0,
        timestamps_seconds=[raw_position_seconds],
        backend="AVFOUNDATION",
    )
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    timing = _read_and_validate_timing(capture, validator, 0)

    if math.isnan(raw_position_seconds):
        assert math.isnan(timing.pos_msec_raw)
    else:
        assert timing.pos_msec_raw == pytest.approx(
            raw_position_seconds * 1_000.0
        )
    if math.isnan(raw_position_seconds):
        assert math.isnan(timing.pos_seconds_raw)
    else:
        assert timing.pos_seconds_raw == pytest.approx(raw_position_seconds)
    assert timing.effective_timestamp_source == "frame_index_fps"
    assert timing.timing_status == "invalid"
    assert expected_flag in timing.timing_flags


def test_aligned_pts_does_not_double_count_consecutive_fallbacks() -> None:
    capture = FakeVideoCapture(
        _frames(3),
        fps=10.0,
        pts=[0, 0, 2],
        timestamps_seconds=[0.0, 0.0, 0.0],
    )
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    timings = [
        _read_and_validate_timing(capture, validator, processed_index)
        for processed_index in range(3)
    ]

    assert [timing.effective_timestamp_seconds for timing in timings] == pytest.approx(
        [0.0, 0.1, 0.2]
    )
    assert timings[-1].effective_timestamp_source == "opencv_pts_aligned"


def test_nonfinite_pts_is_preserved_and_flagged() -> None:
    capture = FakeVideoCapture(
        _frames(1),
        fps=10.0,
        pts=[float("nan")],
        timestamps_seconds=[0.0],
    )
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    timing = _read_and_validate_timing(capture, validator, 0)
    report = validator.report()

    assert timing.pts is not None and math.isnan(timing.pts)
    assert timing.pts_seconds is None
    assert timing.effective_timestamp_source == "opencv_pos_msec"
    assert timing.timing_status == "warning"
    assert "pts_nonfinite" in timing.timing_flags
    assert report.pts_available_frames == 0
    assert report.pts_unavailable_frames == 0
    assert report.pts_invalid_frames == 1


def test_raw_pts_is_preserved_when_reported_fps_is_unavailable() -> None:
    capture = FakeVideoCapture(
        _frames(1),
        fps=0.0,
        pts=[7.0],
        timestamps_seconds=[0.0],
    )
    validator = TimingValidator(reported_fps=0.0, fallback_fps=30.0)

    timing = _read_and_validate_timing(capture, validator, 0)
    report = validator.report()

    assert timing.pts == 7.0
    assert timing.pts_time_base_num is None
    assert timing.pts_time_base_den is None
    assert timing.pts_seconds is None
    assert timing.effective_timestamp_source == "opencv_pos_msec"
    assert timing.timing_flags == ("fps_unavailable",)
    assert report.pts_available_frames == 1
    assert report.pts_unavailable_frames == 0
    assert report.pts_invalid_frames == 0


def test_pts_position_disagreement_allows_one_quantized_pts_tick() -> None:
    capture = FakeVideoCapture(
        _frames(3),
        fps=10.0,
        pts=[100, 101, 102],
        timestamps_seconds=[0.0, 0.001, 0.35],
    )
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    first = _read_and_validate_timing(capture, validator, 0)
    within_one_tick = _read_and_validate_timing(capture, validator, 1)
    disagreeing = _read_and_validate_timing(capture, validator, 2)

    assert "pts_pos_disagreement" not in first.timing_flags
    assert "pts_pos_disagreement" not in within_one_tick.timing_flags
    assert "pts_pos_disagreement" in disagreeing.timing_flags


def test_unselected_timing_anomaly_is_aggregated_in_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(
        _frames(4),
        fps=10.0,
        pts=[0, 1, 2, 3],
        timestamps_seconds=[0.0, 0.1, 0.1, 0.3],
    )

    stats = _run_with_capture(monkeypatch, tmp_path, capture)
    rows = _manifest_rows(stats.keyframe_manifest_path)
    summary = Path(stats.summary_path or "").read_text(encoding="utf-8")

    assert len(rows) == 2
    assert all(row["timing_status"] == "ok" for row in rows)
    assert stats.timing_validation_status == "invalid"
    assert stats.raw_pos_timeline_valid is False
    assert "timing_invalid_frames: 1" in summary
    assert "timing_flag_counts: pos_duplicate=1" in summary
    assert "timing_first_issue_frames: pos_duplicate=2" in summary


def test_selected_invalid_timestamp_exports_compact_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(
        _frames(2),
        fps=10.0,
        pts=[0, 1],
        timestamps_seconds=[0.0, 0.0],
    )

    stats = _run_with_capture(monkeypatch, tmp_path, capture)
    rows = _manifest_rows(stats.keyframe_manifest_path)
    summary = Path(stats.summary_path or "").read_text(encoding="utf-8")

    assert [row["timing_status"] for row in rows] == ["ok", "invalid"]
    assert [float(row["pos_seconds_raw"]) for row in rows] == [0.0, 0.0]
    assert "timing_flag_counts: pos_duplicate=1" in summary


def test_summary_marks_pts_time_base_unavailable_without_decoder_pts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(_frames(2), backend="AVFOUNDATION")

    stats = _run_with_capture(monkeypatch, tmp_path, capture)
    summary = Path(stats.summary_path or "").read_text(encoding="utf-8")

    assert "pts_available_frames: 0" in summary
    assert "pts_time_base_num: unavailable" in summary
    assert "pts_time_base_den: unavailable" in summary
