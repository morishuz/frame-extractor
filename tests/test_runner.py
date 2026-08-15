from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_extractor.config import parse_config
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


def _summary_values(summary_path: str | None) -> dict[str, str]:
    assert summary_path is not None
    return dict(
        line.split(": ", maxsplit=1)
        for line in Path(summary_path).read_text(encoding="utf-8").splitlines()
    )


def _summary_counts(value: str) -> dict[str, int]:
    if value == "none":
        return {}
    return {
        name: int(count)
        for item in value.split(", ")
        for name, count in [item.rsplit("=", maxsplit=1)]
    }


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

    summary = _summary_values(stats.summary_path)
    assert summary["keyframes_csv_schema_version"] == "3"
    assert summary["opencv_version"] == cv2.__version__
    assert summary["video_backend"] == "FFMPEG"
    assert float(summary["reported_fps"]) == pytest.approx(10.0)
    assert (summary["pts_time_base_num"], summary["pts_time_base_den"]) == (
        "1",
        "10",
    )
    assert summary["timing_validation_status"] == "warnings"
    assert summary["raw_pos_timeline_valid"] == "true"
    assert _summary_counts(summary["timing_flag_counts"]) == {
        "pos_gap": 2,
        "pts_gap": 2,
    }

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
    capture = FakeVideoCapture(_frames(6))
    stats = _run_with_capture(
        monkeypatch,
        tmp_path,
        capture,
        start_frame=2,
        max_frames=3,
    )

    rows = _manifest_rows(stats.keyframe_manifest_path)

    assert stats.processed_frames == 3
    assert capture.next_index == 5
    assert [row["decoded_frame_index"] for row in rows] == ["2", "4"]
    assert [row["processed_index"] for row in rows] == ["0", "2"]


def test_failed_start_frame_seek_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(_frames(4), seek_supported=False)

    with pytest.raises(RuntimeError, match="could not seek to start frame 2"):
        _run_with_capture(monkeypatch, tmp_path, capture, start_frame=2)


def test_empty_video_preserves_first_frame_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="Could not read the first frame from the video",
    ):
        _run_with_capture(monkeypatch, tmp_path, FakeVideoCapture([]))


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
    import frame_extractor.timing as timing

    monkeypatch.setattr(
        timing,
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
    summary = _summary_values(stats.summary_path)

    assert len(rows) == 2
    assert all(row["timing_status"] == "ok" for row in rows)
    assert stats.timing_validation_status == "invalid"
    assert stats.raw_pos_timeline_valid is False
    assert int(summary["timing_frames_observed"]) == 4
    assert int(summary["timing_invalid_frames"]) == 1
    assert _summary_counts(summary["timing_flag_counts"]) == {
        "pos_duplicate": 1,
        "effective_fallback": 1,
    }
    assert _summary_counts(summary["timing_first_issue_frames"]) == {
        "pos_duplicate": 2,
        "effective_fallback": 2,
    }


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

    assert [row["timing_status"] for row in rows] == ["ok", "invalid"]
    assert [float(row["pos_seconds_raw"]) for row in rows] == [0.0, 0.0]


def test_summary_marks_pts_time_base_unavailable_without_decoder_pts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = FakeVideoCapture(_frames(2), backend="AVFOUNDATION")

    stats = _run_with_capture(monkeypatch, tmp_path, capture)
    summary = _summary_values(stats.summary_path)

    assert int(summary["pts_available_frames"]) == 0
    assert summary["pts_time_base_num"] == "unavailable"
    assert summary["pts_time_base_den"] == "unavailable"
