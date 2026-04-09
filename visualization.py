from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from metrics import KeyframeDecision, survival_ratio
from tracker import TrackingFrameResult


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
