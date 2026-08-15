from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path

import cv2
import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"
VIDEO_PATH = FIXTURE_DIR / "vfr_timing.mov"
FFPROBE_ORACLE_PATH = FIXTURE_DIR / "vfr_timing.ffprobe.json"


def _read_ffprobe_oracle() -> dict[str, object]:
    with FFPROBE_ORACLE_PATH.open(encoding="utf-8") as oracle_file:
        return json.load(oracle_file)


def _decode_opencv_timing() -> tuple[str, float, list[tuple[float, float]]]:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    assert capture.isOpened(), f"OpenCV could not open timing fixture: {VIDEO_PATH}"

    try:
        backend = capture.getBackendName()
        fps = capture.get(cv2.CAP_PROP_FPS)
        timing: list[tuple[float, float]] = []

        while True:
            ok, _frame = capture.read()
            if not ok:
                break

            # These properties deliberately remain adjacent to read(): they
            # describe the frame most recently returned by the backend.
            pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            pts = capture.get(cv2.CAP_PROP_PTS)
            timing.append((pos_msec / 1000.0, pts))
    finally:
        capture.release()

    return backend, fps, timing


def test_opencv_timing_matches_committed_ffprobe_oracle() -> None:
    """Guard the tested OpenCV/FFmpeg interpretation without running ffprobe."""
    oracle = _read_ffprobe_oracle()
    streams = oracle["streams"]
    frames = oracle["frames"]
    assert isinstance(streams, list) and len(streams) == 1
    assert isinstance(frames, list)

    stream = streams[0]
    assert isinstance(stream, dict)
    assert stream["codec_name"] == "h264"
    assert stream["time_base"] == "1/600"
    assert any(frame.get("pict_type") == "B" for frame in frames)
    expected_seconds = [float(frame["pts_time"]) for frame in frames]
    expected_fps = float(Fraction(stream["avg_frame_rate"]))

    backend, reported_fps, opencv_timing = _decode_opencv_timing()

    assert backend.upper() == "FFMPEG"
    assert reported_fps == pytest.approx(expected_fps)
    assert len(opencv_timing) == int(stream["nb_frames"]) == len(expected_seconds)

    pos_seconds = [position for position, _pts in opencv_timing]
    assert all(math.isfinite(timestamp) for timestamp in pos_seconds)
    assert all(
        later > earlier for earlier, later in zip(pos_seconds, pos_seconds[1:])
    )
    assert pos_seconds == pytest.approx(expected_seconds, abs=0.001)

    expected_gaps = [
        later - earlier
        for earlier, later in zip(expected_seconds, expected_seconds[1:])
    ]
    observed_gaps = [
        later - earlier for earlier, later in zip(pos_seconds, pos_seconds[1:])
    ]
    assert len({round(gap, 6) for gap in expected_gaps}) > 1
    assert observed_gaps == pytest.approx(expected_gaps, abs=0.001)

    # CAP_PROP_PTS is quantized into OpenCV's 1/FPS time base. It need not
    # equal the finer ffprobe timestamp, but must identify the same frame to
    # within half of one reported frame period.
    half_frame_period = 0.5 / reported_fps
    pts_seconds = [pts / reported_fps for _position, pts in opencv_timing]
    for quantized, expected in zip(pts_seconds, expected_seconds):
        assert abs(quantized - expected) <= half_frame_period + 1e-6
