from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from metrics import KeyframeDecision, survival_ratio
from tracker import TrackingFrameResult


THUMBNAIL_GAP = 6
THUMBNAIL_PADDING = 8


def draw_text_lines(
    img: np.ndarray,
    lines: Iterable[tuple[str, tuple[int, int, int]]],
    *,
    origin: tuple[int, int] = (20, 30),
    ystep: int = 36,
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    x0, y0 = origin
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x0, y0 + i * ystep),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_tracks(
    frame_bgr: np.ndarray,
    result: TrackingFrameResult,
    *,
    keyframes_saved: int = 0,
    decision: KeyframeDecision | None = None,
    line_color: tuple[int, int, int] = (255, 255, 255),
    anchor_color: tuple[int, int, int] = (255, 0, 0),
    tracked_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    canvas = frame_bgr.copy()

    for p0, p1, valid in zip(result.anchor_points, result.tracked_points, result.tracked_mask):
        if not valid:
            continue
        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        cv2.line(canvas, (x0, y0), (x1, y1), line_color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, anchor_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, tracked_color, -1, lineType=cv2.LINE_AA)

    white = (255, 255, 255)
    red = (0, 0, 255)

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"frame: {result.frame_index}", white),
        (f"keyframes: {keyframes_saved}", white),
        (f"tracks: {result.tracked_count}/{result.total_points}", white),
    ]

    if decision is None:
        lines.append((f"survivors: {survival_ratio(result) * 100.0:.1f}%", white))
    else:
        surv_color = red if decision.trigger_survival else white
        motion_color = red if decision.trigger_motion else white
        interval_color = red if decision.trigger_interval else white
        lines.append(
            (
                f"survivors: {decision.survival_percent:.1f}% / {decision.min_survival_percent:.1f}%",
                surv_color,
            )
        )
        lines.append(
            (
                f"motion: {decision.motion_normalized:.4f} / {decision.t_motion:.4f}",
                motion_color,
            )
        )
        lines.append(
            (
                f"frames since kf: {decision.frames_since_keyframe} / {decision.max_frames_since_keyframe}",
                interval_color,
            )
        )

    draw_text_lines(canvas, lines)
    return canvas


def thumbnail_size_for_frame(width: int, height: int, *, scale: int = 16) -> tuple[int, int]:
    return (max(1, width // scale), max(1, height // scale))


def thumbnail_strip_size(
    thumbnail_size: tuple[int, int],
    *,
    max_thumbnails: int = 10,
) -> tuple[int, int]:
    thumb_w, thumb_h = thumbnail_size
    width = (THUMBNAIL_PADDING * 2) + (thumb_w * max_thumbnails) + (THUMBNAIL_GAP * (max_thumbnails - 1))
    height = (THUMBNAIL_PADDING * 2) + thumb_h
    return (width, height)


def make_keyframe_thumbnail(frame_bgr: np.ndarray, thumbnail_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame_bgr, thumbnail_size, interpolation=cv2.INTER_AREA)


def resize_to_width(frame_bgr: np.ndarray, output_width: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if width == output_width:
        return frame_bgr
    output_height = max(1, round(height * (output_width / width)))
    return cv2.resize(frame_bgr, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def append_thumbnail_strip(
    frame_bgr: np.ndarray,
    thumbnails: list[np.ndarray],
    *,
    output_width: int,
    strip_height: int,
) -> np.ndarray:
    frame_h, frame_w = frame_bgr.shape[:2]
    canvas = np.zeros((frame_h + strip_height, output_width, 3), dtype=frame_bgr.dtype)
    canvas[:frame_h, :frame_w] = frame_bgr

    y = frame_h + THUMBNAIL_PADDING
    x = THUMBNAIL_PADDING
    max_thumb_h = max(0, strip_height - (THUMBNAIL_PADDING * 2))

    for thumb in thumbnails:
        thumb_h, thumb_w = thumb.shape[:2]
        visible_w = min(thumb_w, output_width - x)
        visible_h = min(thumb_h, max_thumb_h)
        if visible_w <= 0 or visible_h <= 0:
            break
        canvas[y : y + visible_h, x : x + visible_w] = thumb[:visible_h, :visible_w]
        x += thumb_w + THUMBNAIL_GAP

    return canvas
