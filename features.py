from __future__ import annotations

import cv2
import numpy as np


def compute_feature_count(height: int, width: int, *, features_per_mp: int, min_features: int, max_features: int) -> int:
    mp = (height * width) / 1_000_000.0
    raw = int(features_per_mp * mp)
    return int(np.clip(raw, min_features, max_features))


def detect_gftt_points(
    img_gray: np.ndarray,
    *,
    max_corners: int,
    quality_level: float,
    min_distance_px: float,
    block_size: int,
    use_harris_detector: bool,
    k: float,
) -> np.ndarray:
    if img_gray.ndim != 2:
        raise ValueError("detect_gftt_points expects grayscale image (H,W)")

    corners = cv2.goodFeaturesToTrack(
        img_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance_px,
        blockSize=block_size,
        useHarrisDetector=use_harris_detector,
        k=k,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)
