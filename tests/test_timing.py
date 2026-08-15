from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from frame_extractor.timing import TimingValidator
from frame_extractor.timing import read_timed_frame


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


def _frames(count: int) -> list[np.ndarray]:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return [frame.copy() for _ in range(count)]


def _read_and_validate_timing(
    capture: FakeVideoCapture,
    validator: TimingValidator,
    processed_index: int,
):
    decoded_frame = read_timed_frame(
        capture,
        timing_validator=validator,
        reported_fps=capture.fps,
        video_backend=capture.backend,
        processed_index=processed_index,
        fallback_decoded_frame_index=processed_index,
    )
    assert decoded_frame is not None
    assert decoded_frame.processed_index == processed_index
    return decoded_frame.timing


class RecordingVideoCapture(FakeVideoCapture):
    def __init__(self, frames: list[np.ndarray]) -> None:
        super().__init__(frames)
        self.events: list[str] = []

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.events.append("read")
        return super().read()

    def get(self, prop: int) -> float:
        if prop == getattr(cv2, "CAP_PROP_PTS", -1):
            self.events.append("pts")
        elif prop == cv2.CAP_PROP_POS_MSEC:
            self.events.append("pos_msec")
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            self.events.append("pos_frames")
        return super().get(prop)


def test_read_timed_frame_captures_timing_before_decoded_index() -> None:
    capture = RecordingVideoCapture(_frames(1))
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    decoded_frame = read_timed_frame(
        capture,
        timing_validator=validator,
        reported_fps=10.0,
        video_backend="FFMPEG",
        processed_index=0,
        fallback_decoded_frame_index=0,
    )

    assert decoded_frame is not None
    assert decoded_frame.processed_index == 0
    assert decoded_frame.decoded_frame_index == 0
    assert decoded_frame.frame_bgr.shape == (48, 64, 3)
    assert capture.events == ["read", "pts", "pos_msec", "pos_frames"]
    assert validator.report().frames_observed == 1


def test_read_timed_frame_eof_does_not_observe_timing() -> None:
    capture = RecordingVideoCapture([])
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    decoded_frame = read_timed_frame(
        capture,
        timing_validator=validator,
        reported_fps=10.0,
        video_backend="FFMPEG",
        processed_index=0,
        fallback_decoded_frame_index=0,
    )

    assert decoded_frame is None
    assert capture.events == ["read"]
    assert validator.report().frames_observed == 0


def test_read_timed_frame_validates_required_index_before_observing() -> None:
    capture = RecordingVideoCapture(_frames(1))
    validator = TimingValidator(reported_fps=10.0, fallback_fps=10.0)

    with pytest.raises(
        RuntimeError,
        match="seek landed on decoded frame 0; expected 2",
    ):
        read_timed_frame(
            capture,
            timing_validator=validator,
            reported_fps=10.0,
            video_backend="FFMPEG",
            processed_index=0,
            fallback_decoded_frame_index=2,
            required_decoded_frame_index=2,
        )

    assert capture.events == ["read", "pts", "pos_msec", "pos_frames"]
    assert validator.report().frames_observed == 0


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
    assert set(second_timing.timing_flags) == {
        "pos_duplicate",
        "effective_fallback",
    }
    report = validator.report()
    assert not report.raw_pos_timeline_valid
    assert dict(report.flag_counts) == {
        "pos_duplicate": 1,
        "effective_fallback": 1,
    }


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
