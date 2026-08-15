from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from frame_extractor.timing import TimingValidationReport


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


def _format_optional_float(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.9f}"


def _format_optional_int(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def _format_pairs(pairs: tuple[tuple[str, int], ...]) -> str:
    if not pairs:
        return "none"
    return ", ".join(f"{name}={value}" for name, value in pairs)
