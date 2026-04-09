from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must be grayscale (H,W) or BGR (H,W,3)")


def image_diag(height: int, width: int) -> float:
    return float(np.hypot(width, height))


@dataclass
class LKTrackResult:
    pts_next: np.ndarray
    status: np.ndarray


def track_points_lk(
    pts_prev: np.ndarray,
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    *,
    win_size: Tuple[int, int],
    max_level: int,
    criteria: Tuple[int, int, float],
    use_forward_backward_check: bool,
    fb_thresh_px: float,
    min_eig_threshold: float,
) -> LKTrackResult:
    if pts_prev.ndim != 2 or pts_prev.shape[1] != 2:
        raise ValueError("pts_prev must have shape (N,2)")

    n_points = pts_prev.shape[0]
    prev_gray = ensure_gray(frame_prev)
    next_gray = ensure_gray(frame_next)

    valid = np.isfinite(pts_prev).all(axis=1)

    pts_next_full = np.full((n_points, 2), np.nan, dtype=np.float32)
    status_full = np.zeros((n_points,), dtype=bool)

    if not np.any(valid):
        return LKTrackResult(pts_next_full, status_full)

    idx = np.flatnonzero(valid)
    pts_prev_valid = pts_prev[idx].astype(np.float32)
    p0 = pts_prev_valid.reshape(-1, 1, 2)

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=criteria,
        flags=0,
        minEigThreshold=min_eig_threshold,
    )

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

    n_valid = len(idx)
    st_valid = np.zeros((n_valid,), dtype=bool)
    pts_next_valid = np.full((n_valid, 2), np.nan, dtype=np.float32)

    if p1 is not None and st is not None:
        st_valid = st.reshape(-1).astype(bool)
        pts_next_valid = p1.reshape(-1, 2).astype(np.float32)
        pts_next_valid[~st_valid] = np.nan

    if use_forward_backward_check:
        good_forward = st_valid & np.isfinite(pts_next_valid).all(axis=1)

        if np.any(good_forward):
            p1_good = pts_next_valid[good_forward].reshape(-1, 1, 2)
            p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, p1_good, None, **lk_params)

            if p0_back is None or st_back is None:
                st_valid[good_forward] = False
                pts_next_valid[good_forward] = np.nan
            else:
                st_back = st_back.reshape(-1).astype(bool)
                p0_back = p0_back.reshape(-1, 2).astype(np.float32)
                pts_prev_good = pts_prev_valid[good_forward]

                fb = np.linalg.norm(p0_back - pts_prev_good, axis=1).astype(np.float32)
                keep = st_back & (fb <= fb_thresh_px)
                reject_mask = good_forward.copy()
                reject_mask[good_forward] = ~keep

                st_valid[reject_mask] = False
                pts_next_valid[reject_mask] = np.nan

    pts_next_full[idx] = pts_next_valid
    status_full[idx] = st_valid

    return LKTrackResult(pts_next_full, status_full)
