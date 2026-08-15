from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    keyframe_dir: Path
    config_path: Path
    summary_path: Path
    debug_video_path: Path | None

    @property
    def keyframe_manifest_path(self) -> Path:
        return self.run_dir / "keyframes.csv"


@dataclass(frozen=True)
class RunStats:
    processed_frames: int
    avg_fps: float
    runtime_seconds: float
    keyframes_saved: int
    trigger_count: int
    stopped_by_user: bool
    run_dir: str | None
    debug_video_path: str | None
    config_path: str | None
    summary_path: str | None
    keyframe_manifest_path: str | None = None
    timing_validation_status: str | None = None
    raw_pos_timeline_valid: bool | None = None


@dataclass(frozen=True)
class RawFrameTiming:
    pts: float | None
    pts_time_base_num: int | None
    pts_time_base_den: int | None
    pts_seconds: float | None
    pos_msec_raw: float
    pos_seconds_raw: float
    video_backend: str
    opencv_version: str
    reported_fps: float | None


@dataclass(frozen=True)
class FrameTiming:
    pts: float | None
    pts_time_base_num: int | None
    pts_time_base_den: int | None
    pts_seconds: float | None
    pos_msec_raw: float
    pos_seconds_raw: float
    effective_timestamp_seconds: float
    effective_timestamp_source: str
    pos_delta_seconds: float | None
    pts_delta_seconds: float | None
    pts_pos_alignment_offset_seconds: float | None
    pts_pos_residual_seconds: float | None
    timing_status: str
    timing_flags: tuple[str, ...]
    video_backend: str
    opencv_version: str
    reported_fps: float | None


@dataclass(frozen=True)
class TimingValidationReport:
    status: str
    frames_observed: int
    ok_frames: int
    warning_frames: int
    invalid_frames: int
    pts_available_frames: int
    pts_unavailable_frames: int
    pts_invalid_frames: int
    pts_time_base_num: int | None
    pts_time_base_den: int | None
    raw_pos_timeline_valid: bool
    effective_source_counts: tuple[tuple[str, int], ...]
    flag_counts: tuple[tuple[str, int], ...]
    first_issue_frames: tuple[tuple[str, int], ...]
    pts_pos_alignment_offset_seconds: float | None
    nominal_frame_period_seconds: float
    single_step_gap_threshold_seconds: float
    pts_pos_disagreement_tolerance_seconds: float
    initial_join_tolerance_seconds: float


@dataclass(frozen=True)
class KeyframeRecord:
    filename: str
    processed_index: int
    decoded_frame_index: int
    pts: float | None
    pos_seconds_raw: float
    timing_status: str
    selection_reason: str
    motion_score_px: float
    in_bounds_ratio: float


KEYFRAMES_CSV_SCHEMA_VERSION = 3
KEYFRAME_MANIFEST_FIELDS = (
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

TIMING_EPSILON_SECONDS = 1e-9
TIMING_GAP_TOLERANCE_FRAMES = 0.5
PTS_POS_QUANTIZATION_TOLERANCE_FRAMES = 1.05
JOIN_TOLERANCE_MAX_SECONDS = 0.015
JOIN_TOLERANCE_FRAME_FRACTION = 0.45

TIMING_FLAG_ORDER = (
    "fps_unavailable",
    "pos_nonfinite",
    "pos_negative",
    "pos_duplicate",
    "pos_regression",
    "pos_gap",
    "pts_nonfinite",
    "pts_duplicate",
    "pts_regression",
    "pts_gap",
    "pts_pos_disagreement",
    "pos_effective_overlap",
    "effective_fallback",
)
INVALID_TIMING_FLAGS = {
    "pos_nonfinite",
    "pos_negative",
    "pos_duplicate",
    "pos_regression",
}
RAW_POS_BLOCKING_FLAGS = INVALID_TIMING_FLAGS
EFFECTIVE_TIMESTAMP_SOURCE_ORDER = (
    "opencv_pos_msec",
    "opencv_pts",
    "opencv_pts_aligned",
    "frame_duration_fallback",
    "frame_index_fps",
)


class TimingValidator:
    """Validate raw decoder timing while keeping a separate usable timeline."""

    def __init__(self, *, reported_fps: float, fallback_fps: float) -> None:
        self.reported_fps = _valid_fps(reported_fps)
        self.nominal_fps = max(float(fallback_fps), 1e-6)
        self.nominal_frame_period_seconds = 1.0 / self.nominal_fps
        self.pts_pos_disagreement_tolerance_seconds = (
            PTS_POS_QUANTIZATION_TOLERANCE_FRAMES
            * self.nominal_frame_period_seconds
        )
        self.initial_join_tolerance_seconds = min(
            JOIN_TOLERANCE_MAX_SECONDS,
            JOIN_TOLERANCE_FRAME_FRACTION * self.nominal_frame_period_seconds,
        )

        self._last_valid_pos_seconds: float | None = None
        self._last_valid_pos_processed_index: int | None = None
        self._last_valid_pts_seconds: float | None = None
        self._last_valid_pts_processed_index: int | None = None
        self._previous_effective_timestamp_seconds: float | None = None
        self._pts_pos_alignment_offset_seconds: float | None = None
        self._pts_to_effective_offset_seconds: float | None = None

        self._frames_observed = 0
        self._pts_available_frames = 0
        self._pts_unavailable_frames = 0
        self._pts_invalid_frames = 0
        self._pts_time_base_num: int | None = None
        self._pts_time_base_den: int | None = None
        self._status_counts: Counter[str] = Counter()
        self._effective_source_counts: Counter[str] = Counter()
        self._flag_counts: Counter[str] = Counter()
        self._first_issue_frames: dict[str, int] = {}

    def observe(
        self,
        raw: RawFrameTiming,
        *,
        decoded_frame_index: int,
        processed_index: int,
    ) -> FrameTiming:
        flags: set[str] = set()
        pos_delta_seconds: float | None = None
        pts_delta_seconds: float | None = None

        if raw.reported_fps is None:
            flags.add("fps_unavailable")

        pos_is_valid = math.isfinite(raw.pos_msec_raw) and raw.pos_msec_raw >= 0.0
        if not math.isfinite(raw.pos_msec_raw):
            flags.add("pos_nonfinite")
        elif raw.pos_msec_raw < 0.0:
            flags.add("pos_negative")
        else:
            if self._last_valid_pos_seconds is None:
                self._last_valid_pos_seconds = raw.pos_seconds_raw
                self._last_valid_pos_processed_index = processed_index
            else:
                pos_delta_seconds = (
                    raw.pos_seconds_raw - self._last_valid_pos_seconds
                )
                if abs(pos_delta_seconds) <= TIMING_EPSILON_SECONDS:
                    flags.add("pos_duplicate")
                elif pos_delta_seconds < -TIMING_EPSILON_SECONDS:
                    flags.add("pos_regression")
                else:
                    assert self._last_valid_pos_processed_index is not None
                    observed_steps = max(
                        1,
                        processed_index - self._last_valid_pos_processed_index,
                    )
                    gap_limit = (
                        observed_steps + TIMING_GAP_TOLERANCE_FRAMES
                    ) * self.nominal_frame_period_seconds
                    if pos_delta_seconds > gap_limit + TIMING_EPSILON_SECONDS:
                        flags.add("pos_gap")
                    self._last_valid_pos_seconds = raw.pos_seconds_raw
                    self._last_valid_pos_processed_index = processed_index

        if raw.pts is None:
            self._pts_unavailable_frames += 1
        elif not math.isfinite(raw.pts):
            flags.add("pts_nonfinite")
            self._pts_invalid_frames += 1
        else:
            self._pts_available_frames += 1
            if (
                self._pts_time_base_num is None
                and raw.pts_time_base_num is not None
                and raw.pts_time_base_den is not None
            ):
                self._pts_time_base_num = raw.pts_time_base_num
                self._pts_time_base_den = raw.pts_time_base_den
            if raw.pts_seconds is not None:
                if self._last_valid_pts_seconds is None:
                    self._last_valid_pts_seconds = raw.pts_seconds
                    self._last_valid_pts_processed_index = processed_index
                else:
                    pts_delta_seconds = raw.pts_seconds - self._last_valid_pts_seconds
                    if abs(pts_delta_seconds) <= TIMING_EPSILON_SECONDS:
                        flags.add("pts_duplicate")
                    elif pts_delta_seconds < -TIMING_EPSILON_SECONDS:
                        flags.add("pts_regression")
                    else:
                        assert self._last_valid_pts_processed_index is not None
                        observed_steps = max(
                            1,
                            processed_index - self._last_valid_pts_processed_index,
                        )
                        gap_limit = (
                            observed_steps + TIMING_GAP_TOLERANCE_FRAMES
                        ) * self.nominal_frame_period_seconds
                        if pts_delta_seconds > gap_limit + TIMING_EPSILON_SECONDS:
                            flags.add("pts_gap")
                        self._last_valid_pts_seconds = raw.pts_seconds
                        self._last_valid_pts_processed_index = processed_index

        pts_pos_residual_seconds: float | None = None
        if raw.pts_seconds is not None and pos_is_valid:
            current_offset = raw.pts_seconds - raw.pos_seconds_raw
            if self._pts_pos_alignment_offset_seconds is None:
                self._pts_pos_alignment_offset_seconds = current_offset
            pts_pos_residual_seconds = (
                current_offset - self._pts_pos_alignment_offset_seconds
            )
            if (
                abs(pts_pos_residual_seconds)
                > self.pts_pos_disagreement_tolerance_seconds
                + TIMING_EPSILON_SECONDS
            ):
                flags.add("pts_pos_disagreement")

        previous_effective = self._previous_effective_timestamp_seconds
        pos_is_blocked = bool(flags & RAW_POS_BLOCKING_FLAGS)
        can_use_pos = pos_is_valid and not pos_is_blocked
        if (
            can_use_pos
            and previous_effective is not None
            and raw.pos_seconds_raw
            <= previous_effective + TIMING_EPSILON_SECONDS
        ):
            can_use_pos = False
            flags.add("pos_effective_overlap")

        aligned_pts_candidate = (
            raw.pts_seconds + self._pts_to_effective_offset_seconds
            if (
                raw.pts_seconds is not None
                and self._pts_to_effective_offset_seconds is not None
            )
            else None
        )

        if can_use_pos:
            effective_timestamp_seconds = raw.pos_seconds_raw
            effective_timestamp_source = "opencv_pos_msec"
        elif previous_effective is None:
            flags.add("effective_fallback")
            if raw.pts_seconds is not None:
                effective_timestamp_seconds = raw.pts_seconds
                effective_timestamp_source = "opencv_pts"
            else:
                effective_timestamp_seconds = (
                    decoded_frame_index * self.nominal_frame_period_seconds
                )
                effective_timestamp_source = "frame_index_fps"
        else:
            flags.add("effective_fallback")
            if (
                aligned_pts_candidate is not None
                and aligned_pts_candidate
                > previous_effective + TIMING_EPSILON_SECONDS
            ):
                effective_timestamp_seconds = aligned_pts_candidate
                effective_timestamp_source = "opencv_pts_aligned"
            else:
                effective_timestamp_seconds = (
                    previous_effective + self.nominal_frame_period_seconds
                )
                effective_timestamp_source = "frame_duration_fallback"

        if (
            self._pts_to_effective_offset_seconds is None
            and raw.pts_seconds is not None
        ):
            self._pts_to_effective_offset_seconds = (
                effective_timestamp_seconds - raw.pts_seconds
            )

        ordered_flags = tuple(flag for flag in TIMING_FLAG_ORDER if flag in flags)
        if flags & INVALID_TIMING_FLAGS:
            timing_status = "invalid"
        elif ordered_flags:
            timing_status = "warning"
        else:
            timing_status = "ok"

        self._frames_observed += 1
        self._status_counts[timing_status] += 1
        self._effective_source_counts[effective_timestamp_source] += 1
        for flag in ordered_flags:
            self._flag_counts[flag] += 1
            self._first_issue_frames.setdefault(flag, decoded_frame_index)
        self._previous_effective_timestamp_seconds = effective_timestamp_seconds

        return FrameTiming(
            pts=raw.pts,
            pts_time_base_num=raw.pts_time_base_num,
            pts_time_base_den=raw.pts_time_base_den,
            pts_seconds=raw.pts_seconds,
            pos_msec_raw=raw.pos_msec_raw,
            pos_seconds_raw=raw.pos_seconds_raw,
            effective_timestamp_seconds=effective_timestamp_seconds,
            effective_timestamp_source=effective_timestamp_source,
            pos_delta_seconds=pos_delta_seconds,
            pts_delta_seconds=pts_delta_seconds,
            pts_pos_alignment_offset_seconds=(
                self._pts_pos_alignment_offset_seconds
            ),
            pts_pos_residual_seconds=pts_pos_residual_seconds,
            timing_status=timing_status,
            timing_flags=ordered_flags,
            video_backend=raw.video_backend,
            opencv_version=raw.opencv_version,
            reported_fps=raw.reported_fps,
        )

    def report(self) -> TimingValidationReport:
        invalid_frames = self._status_counts["invalid"]
        warning_frames = self._status_counts["warning"]
        if invalid_frames:
            status = "invalid"
        elif warning_frames:
            status = "warnings"
        else:
            status = "ok"

        return TimingValidationReport(
            status=status,
            frames_observed=self._frames_observed,
            ok_frames=self._status_counts["ok"],
            warning_frames=warning_frames,
            invalid_frames=invalid_frames,
            pts_available_frames=self._pts_available_frames,
            pts_unavailable_frames=self._pts_unavailable_frames,
            pts_invalid_frames=self._pts_invalid_frames,
            pts_time_base_num=self._pts_time_base_num,
            pts_time_base_den=self._pts_time_base_den,
            raw_pos_timeline_valid=not any(
                self._flag_counts[flag] for flag in INVALID_TIMING_FLAGS
            ),
            effective_source_counts=tuple(
                (source, self._effective_source_counts[source])
                for source in EFFECTIVE_TIMESTAMP_SOURCE_ORDER
                if self._effective_source_counts[source]
            ),
            flag_counts=tuple(
                (flag, self._flag_counts[flag])
                for flag in TIMING_FLAG_ORDER
                if self._flag_counts[flag]
            ),
            first_issue_frames=tuple(
                (flag, self._first_issue_frames[flag])
                for flag in TIMING_FLAG_ORDER
                if flag in self._first_issue_frames
            ),
            pts_pos_alignment_offset_seconds=(
                self._pts_pos_alignment_offset_seconds
            ),
            nominal_frame_period_seconds=self.nominal_frame_period_seconds,
            single_step_gap_threshold_seconds=(
                (1.0 + TIMING_GAP_TOLERANCE_FRAMES)
                * self.nominal_frame_period_seconds
            ),
            pts_pos_disagreement_tolerance_seconds=(
                self.pts_pos_disagreement_tolerance_seconds
            ),
            initial_join_tolerance_seconds=(
                self.initial_join_tolerance_seconds
            ),
        )


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def make_run_paths(base_output_dir: str, save_debug_video: bool) -> RunPaths:
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    suffix = 1
    while True:
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            run_dir = base_dir / f"{run_name}_{suffix:02d}"
            suffix += 1
    keyframe_dir = run_dir / "keyframes"
    keyframe_dir.mkdir()
    return RunPaths(
        run_dir=run_dir,
        keyframe_dir=keyframe_dir,
        config_path=run_dir / "config.yaml",
        summary_path=run_dir / "summary.txt",
        debug_video_path=(run_dir / "debug.mp4") if save_debug_video else None,
    )


def save_keyframe(
    keyframe_dir: Path,
    frame_bgr: np.ndarray,
    image_format: str,
    keyframe_index: int,
    frame_index: int,
) -> Path:
    keyframe_path = keyframe_dir / (
        f"keyframe_{keyframe_index:04d}_{frame_index:06d}.{image_format}"
    )
    ok = cv2.imwrite(str(keyframe_path), frame_bgr)
    if not ok:
        raise IOError(f"Failed to write keyframe: {keyframe_path}")
    return keyframe_path


def capture_raw_frame_timing(
    cap: cv2.VideoCapture,
    *,
    reported_fps: float,
    video_backend: str,
) -> RawFrameTiming:
    valid_reported_fps = _valid_fps(reported_fps)
    pts = _capture_pts(cap, video_backend)
    position_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))

    pts_time_base = (
        _fps_time_base(valid_reported_fps)
        if pts is not None and valid_reported_fps is not None
        else None
    )
    pts_seconds = (
        float(pts * pts_time_base)
        if pts is not None and math.isfinite(pts) and pts_time_base is not None
        else None
    )
    position_seconds = position_msec / 1000.0

    return RawFrameTiming(
        pts=pts,
        pts_time_base_num=pts_time_base.numerator if pts_time_base is not None else None,
        pts_time_base_den=pts_time_base.denominator if pts_time_base is not None else None,
        pts_seconds=pts_seconds,
        pos_msec_raw=position_msec,
        pos_seconds_raw=position_seconds,
        video_backend=video_backend,
        opencv_version=str(cv2.__version__),
        reported_fps=valid_reported_fps,
    )


def capture_decoded_frame_index(
    cap: cv2.VideoCapture,
    fallback_index: int,
    *,
    required_index: int | None = None,
) -> int:
    next_frame_position = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if math.isfinite(next_frame_position):
        decoded_frame_index = int(round(next_frame_position)) - 1
        if required_index is not None:
            if decoded_frame_index != required_index:
                raise RuntimeError(
                    "Video backend seek landed on decoded frame "
                    f"{decoded_frame_index}; expected {required_index}"
                )
            return decoded_frame_index
        if decoded_frame_index >= fallback_index:
            return decoded_frame_index
    if required_index is not None:
        raise RuntimeError(
            "Video backend did not report a decoded frame index after seeking "
            f"to frame {required_index}"
        )
    return fallback_index


def write_keyframe_manifest(path: Path, records: list[KeyframeRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=KEYFRAME_MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field_name: getattr(record, field_name) for field_name in KEYFRAME_MANIFEST_FIELDS}
            )


def write_summary(
    path: Path,
    *,
    input_video: str,
    start_frame: int,
    max_frames: int | None,
    processed_frames: int,
    runtime_seconds: float,
    keyframes_saved: int,
    trigger_count: int,
    opencv_version: str,
    video_backend: str,
    reported_fps: float | None,
    nominal_fps: float,
    timing_validation: TimingValidationReport,
) -> None:
    mean_frames_per_keyframe = processed_frames / max(keyframes_saved, 1)
    summary = (
        f"input_video: {input_video}\n"
        f"start_frame: {start_frame}\n"
        f"max_frames: {max_frames if max_frames is not None else 'None'}\n"
        f"processed_frames: {processed_frames}\n"
        f"runtime_seconds: {runtime_seconds:.3f}\n"
        f"avg_fps: {processed_frames / max(runtime_seconds, 1e-6):.3f}\n"
        f"keyframes_saved: {keyframes_saved}\n"
        f"mean_frames_per_keyframe: {mean_frames_per_keyframe:.3f}\n"
        f"trigger_count: {trigger_count}\n"
        f"keyframes_csv_schema_version: {KEYFRAMES_CSV_SCHEMA_VERSION}\n"
        f"opencv_version: {opencv_version}\n"
        f"video_backend: {video_backend}\n"
        f"reported_fps: {_format_optional_float(reported_fps)}\n"
        f"pts_time_base_num: "
        f"{_format_optional_int(timing_validation.pts_time_base_num)}\n"
        f"pts_time_base_den: "
        f"{_format_optional_int(timing_validation.pts_time_base_den)}\n"
        f"nominal_fps: {nominal_fps:.9f}\n"
        f"nominal_frame_period_seconds: "
        f"{timing_validation.nominal_frame_period_seconds:.9f}\n"
        f"timing_validation_status: {timing_validation.status}\n"
        f"timing_frames_observed: {timing_validation.frames_observed}\n"
        f"timing_ok_frames: {timing_validation.ok_frames}\n"
        f"timing_warning_frames: {timing_validation.warning_frames}\n"
        f"timing_invalid_frames: {timing_validation.invalid_frames}\n"
        f"pts_available_frames: {timing_validation.pts_available_frames}\n"
        f"pts_unavailable_frames: {timing_validation.pts_unavailable_frames}\n"
        f"pts_invalid_frames: {timing_validation.pts_invalid_frames}\n"
        f"raw_pos_timeline_valid: "
        f"{str(timing_validation.raw_pos_timeline_valid).lower()}\n"
        f"timing_effective_source_counts: "
        f"{_format_pairs(timing_validation.effective_source_counts)}\n"
        f"timing_flag_counts: {_format_pairs(timing_validation.flag_counts)}\n"
        f"timing_first_issue_frames: "
        f"{_format_pairs(timing_validation.first_issue_frames)}\n"
        f"pts_pos_alignment_offset_seconds: "
        f"{_format_optional_float(timing_validation.pts_pos_alignment_offset_seconds)}\n"
        f"single_step_gap_threshold_seconds: "
        f"{timing_validation.single_step_gap_threshold_seconds:.9f}\n"
        f"pts_pos_disagreement_tolerance_seconds: "
        f"{timing_validation.pts_pos_disagreement_tolerance_seconds:.9f}\n"
        f"initial_join_tolerance_seconds: "
        f"{timing_validation.initial_join_tolerance_seconds:.9f}\n"
    )
    path.write_text(summary, encoding="utf-8")


def create_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Cannot open debug video writer: {path}")
    return writer


def resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    target_height = max(1, round(height * (max_width / width)))
    return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)


def pad_to_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_height = _even_dimension(height)
    target_width = _even_dimension(width)
    if target_height == height and target_width == width:
        return frame
    padded = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    padded[:height, :width] = frame
    return padded


def _even_dimension(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def video_backend_name(cap: cv2.VideoCapture) -> str:
    try:
        return str(cap.getBackendName())
    except (AttributeError, cv2.error):
        return "unknown"


def _capture_pts(
    cap: cv2.VideoCapture,
    video_backend: str,
) -> float | None:
    pts_property = getattr(cv2, "CAP_PROP_PTS", None)
    if pts_property is None or video_backend.upper() != "FFMPEG":
        return None
    return float(cap.get(pts_property))


def _fps_time_base(reported_fps: float | None) -> Fraction | None:
    if reported_fps is None:
        return None
    fps = Fraction(reported_fps).limit_denominator(1_000_000)
    return Fraction(fps.denominator, fps.numerator)


def _valid_fps(reported_fps: float) -> float | None:
    if math.isfinite(reported_fps) and reported_fps > 0.0:
        return float(reported_fps)
    return None


def _format_optional_float(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.9f}"


def _format_optional_int(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def _format_pairs(pairs: tuple[tuple[str, int], ...]) -> str:
    if not pairs:
        return "none"
    return ", ".join(f"{name}={value}" for name, value in pairs)
