from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import TrackingConfig
from features import (
    compute_feature_count,
    detect_gftt_points,
)
from optical_flow import LKTrackResult, ensure_gray, image_diag, track_points_lk


@dataclass
class TrackingFrameResult:
    frame_index: int
    anchor_points: np.ndarray
    tracked_points: np.ndarray
    tracked_mask: np.ndarray

    @property
    def total_points(self) -> int:
        return int(self.anchor_points.shape[0])

    @property
    def tracked_count(self) -> int:
        return int(self.tracked_mask.sum())


class LKTracker:
    """
    Tracks anchor points through subsequent frames using LK optical flow.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self._anchor_gray: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._anchor_points: Optional[np.ndarray] = None
        self._tracked_points: Optional[np.ndarray] = None
        self._tracked_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._anchor_gray = None
        self._prev_gray = None
        self._anchor_points = None
        self._tracked_points = None
        self._tracked_mask = None

    def has_anchor(self) -> bool:
        return self._anchor_gray is not None and self._anchor_points is not None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> TrackingFrameResult:
        gray = ensure_gray(frame)

        if not self.has_anchor():
            self._initialize_anchor(gray)
            return self._build_result(frame_index)

        next_points = self._track_to_current(gray)
        self._tracked_points = next_points.pts_next
        self._tracked_mask = next_points.status
        self._prev_gray = gray

        return self._build_result(frame_index)

    def reset_anchor_from_frame(self, frame: np.ndarray) -> None:
        """Recompute anchor points on the given frame and set it as the new anchor."""
        gray = ensure_gray(frame)
        self._initialize_anchor(gray)

    def _initialize_anchor(self, frame_gray: np.ndarray) -> None:
        height, width = frame_gray.shape
        points = self._detect_anchor_points(frame_gray, height=height, width=width)

        self._anchor_gray = frame_gray
        self._prev_gray = frame_gray
        self._anchor_points = points
        self._tracked_points = points.copy()
        self._tracked_mask = np.ones((points.shape[0],), dtype=bool)

    def _detect_anchor_points(self, frame_gray: np.ndarray, *, height: int, width: int) -> np.ndarray:
        gftt_cfg = self.config.gftt
        max_corners = compute_feature_count(
            height,
            width,
            features_per_mp=gftt_cfg.features_per_mp,
            min_features=gftt_cfg.min_features,
            max_features=gftt_cfg.max_features,
        )
        min_distance_px = gftt_cfg.min_distance_frac * image_diag(height, width)
        return detect_gftt_points(
            frame_gray,
            max_corners=max_corners,
            quality_level=gftt_cfg.quality_level,
            min_distance_px=min_distance_px,
            block_size=gftt_cfg.block_size,
            use_harris_detector=gftt_cfg.use_harris_detector,
            k=gftt_cfg.k,
        )

    def _track_to_current(self, current_gray: np.ndarray) -> LKTrackResult:
        lk_cfg = self.config.lk
        height, width = current_gray.shape
        fb_thresh_px = lk_cfg.fb_thresh_frac * image_diag(height, width)
        return track_points_lk(
            self._tracked_points,
            self._prev_gray,
            current_gray,
            win_size=lk_cfg.win_size,
            max_level=lk_cfg.max_level,
            criteria=lk_cfg.criteria,
            use_forward_backward_check=lk_cfg.use_reverse_tracking_check,
            fb_thresh_px=fb_thresh_px,
            min_eig_threshold=lk_cfg.min_eig_threshold,
        )

    def _build_result(self, frame_index: int) -> TrackingFrameResult:
        return TrackingFrameResult(
            frame_index=frame_index,
            anchor_points=self._anchor_points,
            tracked_points=self._tracked_points,
            tracked_mask=self._tracked_mask,
        )
