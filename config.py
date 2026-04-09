from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import cv2
import yaml


@dataclass(frozen=True)
class GFTTConfig:
    features_per_mp: int = 800
    min_features: int = 600
    max_features: int = 4000
    quality_level: float = 0.01
    min_distance_frac: float = 0.001
    block_size: int = 3
    use_harris_detector: bool = False
    k: float = 0.04


@dataclass(frozen=True)
class LKConfig:
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    use_reverse_tracking_check: bool = True
    fb_thresh_frac: float = 0.001
    min_eig_threshold: float = 1e-4


@dataclass(frozen=True)
class TrackingConfig:
    gftt: GFTTConfig = field(default_factory=GFTTConfig)
    lk: LKConfig = field(default_factory=LKConfig)


@dataclass(frozen=True)
class AppConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    min_survival_percent: float = 30.0
    motion_percentile: float = 50.0
    t_motion: float = 0.015
    max_frames_since_keyframe: int = 1000
    n_downsample: int = 2
    keyframe_image_format: str = "jpg"


DEFAULT_APP_CONFIG = AppConfig()


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping")
    return value


def _parse_win_size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("'lk.win_size' must be a 2-item list like [21, 21]")


def _normalize_image_format(value: Any) -> str:
    fmt = str(value).lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"png", "jpg"}:
        raise ValueError("keyframe_image_format must be one of: png, jpg, jpeg")
    return fmt


def _parse_app_config(raw: dict[str, Any]) -> AppConfig:
    defaults = DEFAULT_APP_CONFIG
    tracking_defaults = defaults.tracking
    gftt_defaults = tracking_defaults.gftt
    lk_defaults = tracking_defaults.lk

    if raw.get("detector") is not None and str(raw["detector"]).lower() != "gftt":
        raise ValueError("Only GFTT anchor detection is supported; remove `detector: sift` from the config.")
    if raw.get("sift") is not None:
        raise ValueError("SIFT config is no longer supported; remove the `sift:` section from the config.")

    gftt_raw = _as_mapping(raw.get("gftt"), "gftt")
    lk_raw = _as_mapping(raw.get("lk"), "lk")

    gftt = GFTTConfig(
        features_per_mp=int(gftt_raw.get("features_per_mp", gftt_defaults.features_per_mp)),
        min_features=int(gftt_raw.get("min_features", gftt_defaults.min_features)),
        max_features=int(gftt_raw.get("max_features", gftt_defaults.max_features)),
        quality_level=float(gftt_raw.get("quality_level", gftt_defaults.quality_level)),
        min_distance_frac=float(gftt_raw.get("min_distance_frac", gftt_defaults.min_distance_frac)),
        block_size=int(gftt_raw.get("block_size", gftt_defaults.block_size)),
        use_harris_detector=bool(gftt_raw.get("use_harris_detector", gftt_defaults.use_harris_detector)),
        k=float(gftt_raw.get("k", gftt_defaults.k)),
    )
    lk = LKConfig(
        win_size=_parse_win_size(lk_raw.get("win_size"), lk_defaults.win_size),
        max_level=int(lk_raw.get("max_level", lk_defaults.max_level)),
        use_reverse_tracking_check=bool(
            lk_raw.get(
                "use_reverse_tracking_check",
                lk_raw.get("use_forward_backward_check", lk_defaults.use_reverse_tracking_check),
            )
        ),
        fb_thresh_frac=float(lk_raw.get("fb_thresh_frac", lk_defaults.fb_thresh_frac)),
        min_eig_threshold=float(lk_raw.get("min_eig_threshold", lk_defaults.min_eig_threshold)),
    )

    min_survival_percent = float(raw.get("min_survival_percent", defaults.min_survival_percent))
    if not (0.0 <= min_survival_percent <= 100.0):
        raise ValueError("min_survival_percent must be in [0, 100]")

    motion_percentile = float(raw.get("motion_percentile", defaults.motion_percentile))
    if not (0.0 <= motion_percentile <= 100.0):
        raise ValueError("motion_percentile must be in [0, 100]")

    t_motion = float(raw.get("t_motion", defaults.t_motion))
    if t_motion < 0.0:
        raise ValueError("t_motion must be >= 0")

    max_frames_since_keyframe = int(raw.get("max_frames_since_keyframe", defaults.max_frames_since_keyframe))
    if max_frames_since_keyframe < 0:
        raise ValueError("max_frames_since_keyframe must be >= 0")

    n_downsample = int(raw.get("n_downsample", defaults.n_downsample))
    if n_downsample < 0:
        raise ValueError("n_downsample must be >= 0")

    if gftt.quality_level <= 0.0:
        raise ValueError("gftt.quality_level must be > 0")
    if gftt.min_distance_frac < 0.0:
        raise ValueError("gftt.min_distance_frac must be >= 0")
    if gftt.block_size <= 0:
        raise ValueError("gftt.block_size must be > 0")

    return AppConfig(
        tracking=TrackingConfig(gftt=gftt, lk=lk),
        min_survival_percent=min_survival_percent,
        motion_percentile=motion_percentile,
        t_motion=t_motion,
        max_frames_since_keyframe=max_frames_since_keyframe,
        n_downsample=n_downsample,
        keyframe_image_format=_normalize_image_format(raw.get("keyframe_image_format", defaults.keyframe_image_format)),
    )


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _parse_app_config(_as_mapping(loaded, "root"))


def load_default_app_config() -> AppConfig:
    return DEFAULT_APP_CONFIG


def dump_app_config_yaml(config: AppConfig) -> str:
    data = asdict(config)
    tracking = data.pop("tracking")
    tracking["lk"]["win_size"] = list(config.tracking.lk.win_size)
    tracking["lk"].pop("criteria", None)
    data.update(tracking)
    return yaml.safe_dump(data, sort_keys=False)
