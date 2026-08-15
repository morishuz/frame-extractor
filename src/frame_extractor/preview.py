from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import Sequence

import cv2
import numpy as np

from frame_extractor.config import FrameExtractorConfig
from frame_extractor.tracking import FlowStepDiagnostics
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import TrackingState
from frame_extractor.tracking import TriggerDecision
from frame_extractor.tracking import comparison_label


DASHBOARD_PLOT_HEIGHT = 150
PANEL_PADDING = 10
PLOT_GAP = 10
THUMBNAIL_GAP = 8
TEXT_OVERLAY_ALPHA = 0.5
PLOT_HISTORY_FRAMES = 180


@dataclass
class History:
    frame_indices: list[int] = field(default_factory=list)
    global_scores: list[float] = field(default_factory=list)
    in_bounds_ratios: list[float] = field(default_factory=list)
    trigger_frames: list[int] = field(default_factory=list)
    trigger_reasons: list[str] = field(default_factory=list)


@dataclass
class PreviewKeyframe:
    frame_bgr: np.ndarray
    keyframe_index: int
    frame_index: int
    rendered_thumbnail: np.ndarray | None = None
    rendered_thumbnail_size: tuple[int, int] | None = None


def history_append(history: History, frame_scores: FrameScores, trigger_decision: TriggerDecision) -> None:
    history.frame_indices.append(frame_scores.frame_index)
    history.global_scores.append(frame_scores.global_score)
    history.in_bounds_ratios.append(frame_scores.in_bounds_ratio)
    if trigger_decision.triggered:
        history.trigger_frames.append(frame_scores.frame_index)
        history.trigger_reasons.append(trigger_decision.reason)

    overflow = len(history.frame_indices) - PLOT_HISTORY_FRAMES
    if overflow > 0:
        del history.frame_indices[:overflow]
        del history.global_scores[:overflow]
        del history.in_bounds_ratios[:overflow]

    trigger_overflow = len(history.trigger_frames) - PLOT_HISTORY_FRAMES
    if trigger_overflow > 0:
        del history.trigger_frames[:trigger_overflow]
        del history.trigger_reasons[:trigger_overflow]

    if history.frame_indices:
        earliest_frame = history.frame_indices[0]
        while history.trigger_frames and history.trigger_frames[0] < earliest_frame:
            del history.trigger_frames[0]
            del history.trigger_reasons[0]


def render_tracking_view(
    frame_bgr: np.ndarray,
    state: TrackingState,
    diagnostics: FlowStepDiagnostics,
    frame_scores: FrameScores,
    trigger_decision: TriggerDecision,
    config: FrameExtractorConfig,
    keyframe_count: int,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    displayed_mask = diagnostics.in_bounds_mask & state.alive_mask
    track_color = _rgb_to_bgr(config.visualization.point_color_rgb)

    if config.visualization.show_displacement_vectors:
        displacement_indices = np.flatnonzero(displayed_mask)
        for idx in displacement_indices:
            start = state.origin_points[idx]
            end = state.current_points[idx]
            x0, y0 = int(round(float(start[0]))), int(round(float(start[1])))
            x1, y1 = int(round(float(end[0]))), int(round(float(end[1])))
            cv2.line(canvas, (x0, y0), (x1, y1), track_color, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x0, y0), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    for idx, point in enumerate(state.current_points):
        if not displayed_mask[idx]:
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(
            canvas,
            (x, y),
            config.visualization.point_radius,
            track_color,
            -1,
            lineType=cv2.LINE_AA,
        )

    trigger_color = (0, 0, 255) if trigger_decision.triggered else (255, 255, 255)
    lines = [
        (
            f"frame: {frame_scores.frame_index}    time: {frame_scores.timestamp_sec:.3f}s",
            (255, 255, 255),
        ),
        (
            f"frames since last trigger: {trigger_decision.frames_since_keyframe}",
            (255, 255, 255),
        ),
        (
            f"number of keyframes: {keyframe_count}",
            (255, 255, 255),
        ),
        (
            f"motion: {comparison_label(frame_scores.global_score, config.trigger.main_threshold_original_px)}",
            (255, 255, 255),
        ),
        (
            f"points: {comparison_label(frame_scores.in_bounds_ratio, config.trigger.min_in_bounds_ratio)}",
            (255, 255, 255),
        ),
        (
            f"trigger: {trigger_decision.display_reason}",
            trigger_color,
        ),
    ]
    _draw_text_block(canvas, lines)
    return canvas


def render_debug_dashboard(
    *,
    width: int,
    history: History,
    config: FrameExtractorConfig,
    thumbnails: list[PreviewKeyframe],
    highlight_latest_thumbnail: bool,
) -> np.ndarray:
    plot_row = np.zeros((DASHBOARD_PLOT_HEIGHT, width, 3), dtype=np.uint8)
    plot_w = max(1, (width - (2 * PANEL_PADDING) - PLOT_GAP) // 2)
    trigger_events = list(zip(history.trigger_frames, history.trigger_reasons))
    threshold_line_color = _rgb_to_bgr(config.visualization.threshold_line_color_rgb)
    trigger_line_color = _rgb_to_bgr(config.visualization.trigger_line_color_rgb)
    left_plot = _render_live_metric_plot(
        title="motion",
        frame_indices=history.frame_indices,
        values=history.global_scores,
        threshold=config.trigger.main_threshold_original_px,
        trigger_events=trigger_events,
        width=plot_w,
        height=DASHBOARD_PLOT_HEIGHT,
        value_color=_rgb_to_bgr(config.visualization.motion_plot_color_rgb),
        threshold_line_color=threshold_line_color,
        trigger_line_color=trigger_line_color,
    )
    right_plot = _render_live_metric_plot(
        title="points",
        frame_indices=history.frame_indices,
        values=history.in_bounds_ratios,
        threshold=config.trigger.min_in_bounds_ratio,
        trigger_events=trigger_events,
        width=plot_w,
        height=DASHBOARD_PLOT_HEIGHT,
        value_color=_rgb_to_bgr(config.visualization.points_plot_color_rgb),
        threshold_line_color=threshold_line_color,
        trigger_line_color=trigger_line_color,
        y_min=0.0,
        y_max=1.0,
    )
    x0 = PANEL_PADDING
    plot_row[:, x0 : x0 + plot_w] = left_plot
    x1 = x0 + plot_w + PLOT_GAP
    plot_row[:, x1 : x1 + plot_w] = right_plot

    thumb_row = _render_thumbnail_row(
        thumbnails,
        width=width,
        n_slots=config.visualization.keyframe_thumbnail_slots,
        highlight_latest=highlight_latest_thumbnail,
    )
    return np.vstack([thumb_row, plot_row])


def compose_debug_frame(frame_bgr: np.ndarray, dashboard: np.ndarray) -> np.ndarray:
    _frame_h, frame_w = frame_bgr.shape[:2]
    dash_h, dash_w = dashboard.shape[:2]
    if dash_w != frame_w:
        dashboard = cv2.resize(dashboard, (frame_w, dash_h), interpolation=cv2.INTER_AREA)
    return np.vstack([frame_bgr, dashboard])


def _draw_text_block(
    canvas: np.ndarray,
    lines: Iterable[tuple[str, tuple[int, int, int]]],
    *,
    origin: tuple[int, int] = (14, 46),
    line_step: int = 22,
    font_scale: float = 0.58,
) -> None:
    line_list = list(lines)
    if not line_list:
        return

    x0, y0 = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text, _color in line_list]
    max_text_w = max(width for width, _height in text_sizes)
    text_h = max(height for _width, height in text_sizes)
    pad_x = 8
    pad_y = 7
    overlay_x0 = max(0, x0 - pad_x)
    overlay_y0 = max(0, y0 - text_h - pad_y)
    overlay_x1 = min(canvas.shape[1], x0 + max_text_w + pad_x)
    overlay_y1 = min(canvas.shape[0], y0 + (len(line_list) - 1) * line_step + pad_y)

    if overlay_x1 > overlay_x0 and overlay_y1 > overlay_y0:
        overlay = canvas[overlay_y0:overlay_y1, overlay_x0:overlay_x1].copy()
        overlay[:] = (0, 0, 0)
        canvas[overlay_y0:overlay_y1, overlay_x0:overlay_x1] = cv2.addWeighted(
            overlay,
            TEXT_OVERLAY_ALPHA,
            canvas[overlay_y0:overlay_y1, overlay_x0:overlay_x1],
            1.0 - TEXT_OVERLAY_ALPHA,
            0.0,
        )

    for idx, (text, color) in enumerate(line_list):
        y = y0 + idx * line_step
        cv2.putText(canvas, text, (x0, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _make_keyframe_thumbnail(
    frame_bgr: np.ndarray,
    *,
    keyframe_index: int,
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    thumb_w = max(1, width)
    thumb_h = max(1, height)
    thumb = _resize_exact(frame_bgr, thumb_w, thumb_h)
    label = f"{keyframe_index}:{frame_index}"
    label_h = min(22, thumb_h)
    label_overlay = thumb[:label_h].copy()
    label_overlay[:] = (0, 0, 0)
    thumb[:label_h] = cv2.addWeighted(
        label_overlay,
        TEXT_OVERLAY_ALPHA,
        thumb[:label_h],
        1.0 - TEXT_OVERLAY_ALPHA,
        0.0,
    )
    cv2.putText(
        thumb,
        label,
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return thumb


def _resize_exact(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / src_w, target_h / src_h)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def _render_thumbnail_row(
    thumbnails: list[PreviewKeyframe],
    *,
    width: int,
    n_slots: int,
    highlight_latest: bool,
) -> np.ndarray:
    n_slots = max(1, n_slots)
    visible_thumbnails = thumbnails[-n_slots:]
    available_w = max(1, width - (2 * PANEL_PADDING) - (THUMBNAIL_GAP * (n_slots - 1)))
    target_w = max(1, available_w // n_slots)
    if visible_thumbnails:
        thumb_h, thumb_w = visible_thumbnails[-1].frame_bgr.shape[:2]
        aspect_h_over_w = thumb_h / max(float(thumb_w), 1.0)
    else:
        aspect_h_over_w = 9.0 / 16.0
    target_h = max(1, int(round(target_w * aspect_h_over_w)))
    row_h = target_h + (2 * PANEL_PADDING)
    row = np.zeros((row_h, width, 3), dtype=np.uint8)
    y = PANEL_PADDING
    used_w = (target_w * n_slots) + (THUMBNAIL_GAP * (n_slots - 1))
    x = max(PANEL_PADDING, (width - used_w) // 2)

    for slot_idx in range(n_slots):
        slot_x = x + slot_idx * (target_w + THUMBNAIL_GAP)
        cv2.rectangle(row, (slot_x, y), (slot_x + target_w - 1, y + target_h - 1), (35, 35, 35), 1)
        if slot_idx >= len(visible_thumbnails):
            continue

        preview_keyframe = visible_thumbnails[slot_idx]
        target_size = (target_w, target_h)
        if (
            preview_keyframe.rendered_thumbnail is None
            or preview_keyframe.rendered_thumbnail_size != target_size
        ):
            preview_keyframe.rendered_thumbnail = _make_keyframe_thumbnail(
                preview_keyframe.frame_bgr,
                keyframe_index=preview_keyframe.keyframe_index,
                frame_index=preview_keyframe.frame_index,
                width=target_w,
                height=target_h,
            )
            preview_keyframe.rendered_thumbnail_size = target_size
        thumb = preview_keyframe.rendered_thumbnail
        row[y : y + target_h, slot_x : slot_x + target_w] = thumb
        is_latest = highlight_latest and slot_idx == len(visible_thumbnails) - 1
        border_color = (0, 0, 255) if is_latest else (70, 70, 70)
        border_thickness = 3 if is_latest else 1
        cv2.rectangle(
            row,
            (slot_x, y),
            (slot_x + target_w - 1, y + target_h - 1),
            border_color,
            border_thickness,
        )
    return row


def _render_live_metric_plot(
    *,
    title: str,
    frame_indices: Sequence[int],
    values: Sequence[float],
    threshold: float,
    trigger_events: Sequence[tuple[int, str]],
    width: int,
    height: int,
    value_color: tuple[int, int, int],
    threshold_line_color: tuple[int, int, int],
    trigger_line_color: tuple[int, int, int],
    y_min: float = 0.0,
    y_max: float | None = None,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    cv2.putText(panel, title, (PANEL_PADDING, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    if not frame_indices or not values:
        return panel

    plot_left = PANEL_PADDING
    plot_right = width - PANEL_PADDING
    plot_top = 34
    plot_bottom = height - PANEL_PADDING
    plot_w = max(1, plot_right - plot_left)
    plot_h = max(1, plot_bottom - plot_top)

    recent_count = min(len(frame_indices), PLOT_HISTORY_FRAMES)
    xs = list(frame_indices[-recent_count:])
    ys = list(values[-recent_count:])
    x_min = min(xs)
    x_max = max(xs)
    if x_min == x_max:
        x_max += 1

    if y_max is None:
        y_max = max(threshold, max(ys), 1.0) * 1.15
    if y_max <= y_min:
        y_max = y_min + 1.0

    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (55, 55, 55), 1)

    def to_xy(frame_idx: int, value: float) -> tuple[int, int]:
        x_frac = (frame_idx - x_min) / max(float(x_max - x_min), 1.0)
        y_frac = (float(value) - y_min) / max(float(y_max - y_min), 1e-6)
        y_frac = float(np.clip(y_frac, 0.0, 1.0))
        x = plot_left + int(round(x_frac * plot_w))
        y = plot_bottom - int(round(y_frac * plot_h))
        return x, y

    threshold_y = to_xy(x_min, threshold)[1]
    cv2.line(panel, (plot_left, threshold_y), (plot_right, threshold_y), threshold_line_color, 1, cv2.LINE_AA)

    recent_trigger_events = [(frame, reason) for frame, reason in trigger_events if x_min <= frame <= x_max]
    for frame, _reason in recent_trigger_events:
        x = to_xy(frame, y_min)[0]
        cv2.line(panel, (x, plot_top), (x, plot_bottom), trigger_line_color, 1, cv2.LINE_AA)

    points = np.array([to_xy(frame, value) for frame, value in zip(xs, ys)], dtype=np.int32)
    if points.shape[0] >= 2:
        cv2.polylines(panel, [points], False, value_color, 2, cv2.LINE_AA)
    else:
        cv2.circle(panel, tuple(points[0]), 2, value_color, -1, cv2.LINE_AA)

    label = comparison_label(ys[-1], threshold)
    cv2.putText(panel, label, (plot_right - 135, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def _rgb_to_bgr(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = color_rgb
    return blue, green, red
