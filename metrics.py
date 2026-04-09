from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tracker import TrackingFrameResult


class KeyframeDecider(Protocol):
    def evaluate(self, result: TrackingFrameResult, frame_shape_hw: tuple[int, int]) -> "KeyframeDecision":
        """Compute decision state and whether current frame should become a keyframe."""


@dataclass(frozen=True)
class KeyframeDecision:
    survival_percent: float
    min_survival_percent: float
    motion_normalized: float
    t_motion: float
    frames_since_keyframe: int
    max_frames_since_keyframe: int
    trigger_survival: bool
    trigger_motion: bool
    trigger_interval: bool

    @property
    def should_select(self) -> bool:
        return self.trigger_survival or self.trigger_motion or self.trigger_interval


def survival_ratio(result: TrackingFrameResult) -> float:
    total = result.total_points
    if total <= 0:
        return 0.0
    return float(result.tracked_count) / float(total)


def surviving_displacements_px(result: TrackingFrameResult) -> np.ndarray:
    if result.total_points <= 0:
        return np.empty((0,), dtype=np.float32)
    valid = result.tracked_mask
    if not np.any(valid):
        return np.empty((0,), dtype=np.float32)
    deltas = result.tracked_points[valid] - result.anchor_points[valid]
    return np.linalg.norm(deltas, axis=1).astype(np.float32)


def robust_motion_normalized(
    result: TrackingFrameResult,
    frame_shape_hw: tuple[int, int],
    *,
    percentile: float = 70.0,
) -> float:
    distances = surviving_displacements_px(result)
    if distances.size == 0:
        return 0.0
    h, w = frame_shape_hw
    diag = float(np.hypot(w, h))
    if diag <= 0.0:
        return 0.0
    d_robust_px = float(np.percentile(distances, percentile))
    return d_robust_px / diag


@dataclass
class SurvivalMotionIntervalDecider:
    min_survival_percent: float = 70.0
    motion_percentile: float = 70.0
    t_motion: float = 0.01
    max_frames_since_keyframe: int = 200
    _last_keyframe_idx: int | None = None

    def evaluate(self, result: TrackingFrameResult, frame_shape_hw: tuple[int, int]) -> KeyframeDecision:
        if self._last_keyframe_idx is None:
            self._last_keyframe_idx = result.frame_index
            return KeyframeDecision(
                survival_percent=survival_ratio(result) * 100.0,
                min_survival_percent=self.min_survival_percent,
                motion_normalized=robust_motion_normalized(
                    result,
                    frame_shape_hw,
                    percentile=self.motion_percentile,
                ),
                t_motion=self.t_motion,
                frames_since_keyframe=0,
                max_frames_since_keyframe=self.max_frames_since_keyframe,
                trigger_survival=False,
                trigger_motion=False,
                trigger_interval=False,
            )

        frames_since_last = result.frame_index - self._last_keyframe_idx
        surv_pct = survival_ratio(result) * 100.0
        d_robust_norm = robust_motion_normalized(
            result,
            frame_shape_hw,
            percentile=self.motion_percentile,
        )

        trigger_survival = bool(result.total_points > 0 and surv_pct < self.min_survival_percent)
        trigger_motion = bool(d_robust_norm > self.t_motion)
        trigger_interval = bool(
            self.max_frames_since_keyframe > 0 and frames_since_last >= self.max_frames_since_keyframe
        )

        decision = KeyframeDecision(
            survival_percent=surv_pct,
            min_survival_percent=self.min_survival_percent,
            motion_normalized=d_robust_norm,
            t_motion=self.t_motion,
            frames_since_keyframe=frames_since_last,
            max_frames_since_keyframe=self.max_frames_since_keyframe,
            trigger_survival=trigger_survival,
            trigger_motion=trigger_motion,
            trigger_interval=trigger_interval,
        )

        if decision.should_select:
            self._last_keyframe_idx = result.frame_index

        return decision
