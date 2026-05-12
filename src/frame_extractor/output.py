from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    keyframe_dir = run_dir / "keyframes"
    run_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
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
) -> None:
    keyframe_path = keyframe_dir / f"keyframe_{keyframe_index:04d}_{frame_index:06d}.{image_format}"
    ok = cv2.imwrite(str(keyframe_path), frame_bgr)
    if not ok:
        raise IOError(f"Failed to write keyframe: {keyframe_path}")


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
