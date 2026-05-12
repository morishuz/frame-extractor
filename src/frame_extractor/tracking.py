from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from frame_extractor.config import DISConfig
from frame_extractor.config import FrameExtractorConfig
from frame_extractor.config import TriggerConfig


@dataclass
class TrackingState:
    origin_points: np.ndarray
    current_points: np.ndarray
    alive_mask: np.ndarray


@dataclass(frozen=True)
class FlowStepDiagnostics:
    in_bounds_mask: np.ndarray


@dataclass(frozen=True)
class FrameScores:
    frame_index: int
    timestamp_sec: float
    global_score: float
    in_bounds_points: int
    in_bounds_ratio: float


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    reason: str
    frames_since_keyframe: int

    @property
    def display_reason(self) -> str:
        if not self.triggered:
            return ""
        if "in_bounds" in self.reason:
            return "points"
        return "motion"


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must be grayscale (H,W) or BGR (H,W,3)")


def downsample_frame(frame: np.ndarray, n_downsample: int) -> np.ndarray:
    out = frame
    for _ in range(max(0, int(n_downsample))):
        height, width = out.shape[:2]
        if height <= 1 or width <= 1:
            break
        out = cv2.pyrDown(out)
    return out


def create_dis_flow(config: DISConfig) -> cv2.DISOpticalFlow:
    preset_map = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    flow = cv2.DISOpticalFlow.create(preset_map[config.preset])
    flow.setFinestScale(config.finest_scale)
    flow.setPatchSize(config.patch_size)
    flow.setPatchStride(config.patch_stride)
    flow.setGradientDescentIterations(config.gradient_descent_iterations)
    flow.setVariationalRefinementIterations(config.variational_refinement_iterations)
    flow.setUseSpatialPropagation(config.use_spatial_propagation)
    return flow


def processing_scale(config: FrameExtractorConfig) -> float:
    return float(2 ** max(0, int(config.n_downsample)))


def original_px_to_processing_px(value: float, config: FrameExtractorConfig) -> float:
    return float(value) / processing_scale(config)


def original_px_to_processing_int(value: float, config: FrameExtractorConfig, *, minimum: int) -> int:
    return max(minimum, int(round(original_px_to_processing_px(value, config))))


def initialize_tracking_state(frame_gray: np.ndarray, config: FrameExtractorConfig) -> TrackingState:
    height, width = frame_gray.shape[:2]
    origin_points = _initialize_sample_points(width, height, config)
    n_points = origin_points.shape[0]
    return TrackingState(
        origin_points=origin_points.copy(),
        current_points=origin_points.copy(),
        alive_mask=np.ones((n_points,), dtype=bool),
    )


def step_tracking(
    state: TrackingState,
    prev_gray: np.ndarray,
    current_gray: np.ndarray,
    flow_forward_solver: cv2.DISOpticalFlow,
    config: FrameExtractorConfig,
) -> FlowStepDiagnostics:
    flow_forward = flow_forward_solver.calc(prev_gray, current_gray, None)

    current_points = state.current_points.astype(np.float32, copy=True)
    sampled_forward_flow, forward_valid_mask = _bilinear_sample_flow(flow_forward, current_points)
    sampled_forward_flow = _clip_vectors(
        sampled_forward_flow,
        original_px_to_processing_px(config.max_step_norm_original_px, config),
    )

    next_points = current_points.copy()
    next_points[forward_valid_mask] = current_points[forward_valid_mask] + sampled_forward_flow[forward_valid_mask]

    height, width = current_gray.shape[:2]
    in_bounds_mask = _inside_image(next_points, width, height)

    state.current_points = next_points
    lost_mask = _beyond_lost_border(
        next_points,
        width,
        height,
        original_px_to_processing_px(config.sampling.lost_border_original_px, config),
    )
    state.alive_mask &= ~lost_mask

    return FlowStepDiagnostics(in_bounds_mask=in_bounds_mask)


def compute_frame_scores(
    state: TrackingState,
    diagnostics: FlowStepDiagnostics,
    frame_index: int,
    timestamp_sec: float,
    config: FrameExtractorConfig,
) -> FrameScores:
    displacement = (
        np.linalg.norm(state.current_points - state.origin_points, axis=1) * processing_scale(config)
    ).astype(np.float32)
    score_mask = state.alive_mask & diagnostics.in_bounds_mask & np.isfinite(displacement)

    global_score = _percentile(displacement[score_mask], config.scoring.percentile)

    total_points = max(int(state.origin_points.shape[0]), 1)
    in_bounds_points = int((state.alive_mask & diagnostics.in_bounds_mask).sum())
    in_bounds_ratio = float(in_bounds_points) / float(total_points)

    return FrameScores(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        global_score=global_score,
        in_bounds_points=in_bounds_points,
        in_bounds_ratio=in_bounds_ratio,
    )


def decide_trigger(frame_scores: FrameScores, frames_since_keyframe: int, config: TriggerConfig) -> TriggerDecision:
    trigger_allowed = frames_since_keyframe >= config.min_frames_since_keyframe
    reasons: list[str] = []
    if trigger_allowed and frame_scores.global_score >= config.main_threshold_original_px:
        reasons.append("main")
    if trigger_allowed and frame_scores.in_bounds_ratio < config.min_in_bounds_ratio:
        reasons.append("in_bounds")
    if config.max_frames_since_keyframe > 0 and frames_since_keyframe >= config.max_frames_since_keyframe:
        reasons.append("interval")
    return TriggerDecision(
        triggered=bool(reasons),
        reason="+".join(reasons) if reasons else "none",
        frames_since_keyframe=frames_since_keyframe,
    )


def comparison_label(value: float, threshold: float) -> str:
    if np.isclose(value, threshold):
        relation = "="
    elif value < threshold:
        relation = "<"
    else:
        relation = ">"
    return f"{value:.2f} {relation} {threshold:.2f}"


def _grid_axis(length: int, step: int, margin: int) -> np.ndarray:
    start = min(max(margin, 0), max(length - 1, 0))
    stop = max(start + 1, length - max(margin, 0))
    axis = np.arange(start, stop, max(1, step), dtype=np.float32)
    if axis.size == 0:
        axis = np.array([max(0.0, (length - 1) / 2.0)], dtype=np.float32)
    return axis


def _make_grid_points(width: int, height: int, *, step: int, margin: int) -> np.ndarray:
    xs = _grid_axis(width, step, margin)
    ys = _grid_axis(height, step, margin)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return points.astype(np.float32)


def _initialize_sample_points(width: int, height: int, config: FrameExtractorConfig) -> np.ndarray:
    sampling = config.sampling
    return _make_grid_points(
        width,
        height,
        step=original_px_to_processing_int(sampling.grid_step_original_px, config, minimum=1),
        margin=original_px_to_processing_int(sampling.min_margin_original_px, config, minimum=0),
    )


def _inside_image(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    valid = np.isfinite(points).all(axis=1)
    valid &= (x >= 0.0) & (x <= float(width - 1))
    valid &= (y >= 0.0) & (y <= float(height - 1))
    return valid


def _beyond_lost_border(points: np.ndarray, width: int, height: int, lost_border_px: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    return (
        ~np.isfinite(points).all(axis=1)
        | (x < -lost_border_px)
        | (x > float(width - 1) + lost_border_px)
        | (y < -lost_border_px)
        | (y > float(height - 1) + lost_border_px)
    )


def _bilinear_sample_flow(flow: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    n_points = points.shape[0]
    values = np.zeros((n_points, 2), dtype=np.float32)
    valid = np.zeros((n_points,), dtype=bool)
    if n_points == 0:
        return values, valid

    height, width = flow.shape[:2]
    if height == 0 or width == 0:
        return values, valid

    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    valid = np.isfinite(points).all(axis=1)
    valid &= (x >= 0.0) & (x <= float(width - 1))
    valid &= (y >= 0.0) & (y <= float(height - 1))
    if not np.any(valid):
        return values, valid

    idx = np.flatnonzero(valid)
    x_valid = x[idx]
    y_valid = y[idx]

    x0 = np.floor(x_valid).astype(np.int32)
    y0 = np.floor(y_valid).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    dx = (x_valid - x0.astype(np.float32)).reshape(-1, 1)
    dy = (y_valid - y0.astype(np.float32)).reshape(-1, 1)

    q00 = flow[y0, x0].astype(np.float32)
    q10 = flow[y0, x1].astype(np.float32)
    q01 = flow[y1, x0].astype(np.float32)
    q11 = flow[y1, x1].astype(np.float32)

    top = q00 * (1.0 - dx) + q10 * dx
    bottom = q01 * (1.0 - dx) + q11 * dx
    values[idx] = top * (1.0 - dy) + bottom * dy
    return values, valid


def _clip_vectors(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0 or vectors.size == 0:
        return vectors.astype(np.float32, copy=True)
    clipped = vectors.astype(np.float32, copy=True)
    norms = np.linalg.norm(clipped, axis=1)
    needs_clip = norms > max_norm
    if np.any(needs_clip):
        scale = (max_norm / np.maximum(norms[needs_clip], 1e-6)).astype(np.float32)
        clipped[needs_clip] *= scale[:, None]
    return clipped


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        return 0.0
    return float(np.percentile(values[valid], percentile))
