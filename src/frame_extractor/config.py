from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml


REPO_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class DISConfig:
    preset: str = "ultrafast"
    finest_scale: int = 1
    patch_size: int = 8
    patch_stride: int = 8
    gradient_descent_iterations: int = 12
    variational_refinement_iterations: int = 0
    use_spatial_propagation: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    grid_step_original_px: int = 160
    min_margin_original_px: int = 16
    lost_border_original_px: float = 24.0


@dataclass(frozen=True)
class ScoringConfig:
    percentile: float = 80.0


@dataclass(frozen=True)
class TriggerConfig:
    min_frames_since_keyframe: int = 2
    main_threshold_original_px: float = 400.0
    min_in_bounds_ratio: float = 0.40
    max_frames_since_keyframe: int = 1000


@dataclass(frozen=True)
class VisualizationConfig:
    show_displacement_vectors: bool = False
    point_radius: int = 3
    keyframe_thumbnail_slots: int = 8
    preview_max_width: int = 5000
    save_debug_video: bool = True
    point_color_rgb: tuple[int, int, int] = (0, 0, 0)
    motion_plot_color_rgb: tuple[int, int, int] = (255, 190, 70)
    points_plot_color_rgb: tuple[int, int, int] = (120, 220, 90)
    threshold_line_color_rgb: tuple[int, int, int] = (40, 120, 255)
    trigger_line_color_rgb: tuple[int, int, int] = (255, 0, 0)


@dataclass(frozen=True)
class OutputConfig:
    image_format: str = "jpg"


@dataclass(frozen=True)
class FrameExtractorConfig:
    dis: DISConfig = field(default_factory=DISConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    n_downsample: int = 2
    max_step_norm_original_px: float = 100.0


DEFAULT_CONFIG = FrameExtractorConfig()


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping")
    return value


def _normalize_positive_int(value: Any, name: str, default: int, *, allow_zero: bool = False) -> int:
    if value is None:
        return default
    parsed = int(value)
    if allow_zero:
        if parsed < 0:
            raise ValueError(f"{name} must be >= 0")
    elif parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _normalize_non_negative_float(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _normalize_ratio(value: Any, name: str, default: float, *, upper: float = 1.0) -> float:
    if value is None:
        return default
    parsed = float(value)
    if not (0.0 < parsed < upper):
        raise ValueError(f"{name} must be in (0, {upper})")
    return parsed


def _normalize_choice(value: Any, name: str, default: str, allowed: set[str]) -> str:
    if value is None:
        return default
    parsed = str(value).lower().strip()
    if parsed not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {allowed_text}")
    return parsed


def _normalize_bool(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        parsed = value.lower().strip()
        if parsed in {"true", "yes", "1", "on"}:
            return True
        if parsed in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _normalize_image_format(value: Any, default: str) -> str:
    if value is None:
        return default
    fmt = str(value).lower().strip()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"png", "jpg"}:
        raise ValueError("output.image_format must be one of: png, jpg, jpeg")
    return fmt


def _normalize_rgb_color(value: Any, name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an RGB list with three values, e.g. [255, 0, 0]")
    color = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"{name} RGB values must be in [0, 255]")
    return color


def parse_config(raw: dict[str, Any]) -> FrameExtractorConfig:
    defaults = DEFAULT_CONFIG
    dis_raw = _as_mapping(raw.get("dis"), "dis")
    sampling_raw = _as_mapping(raw.get("sampling"), "sampling")
    scoring_raw = _as_mapping(raw.get("scoring"), "scoring")
    trigger_raw = _as_mapping(raw.get("trigger"), "trigger")
    vis_raw = _as_mapping(raw.get("visualization"), "visualization")
    output_raw = _as_mapping(raw.get("output"), "output")

    dis = DISConfig(
        preset=_normalize_choice(
            dis_raw.get("preset"),
            "dis.preset",
            defaults.dis.preset,
            {"ultrafast", "fast", "medium"},
        ),
        finest_scale=_normalize_positive_int(
            dis_raw.get("finest_scale"),
            "dis.finest_scale",
            defaults.dis.finest_scale,
            allow_zero=True,
        ),
        patch_size=_normalize_positive_int(
            dis_raw.get("patch_size"),
            "dis.patch_size",
            defaults.dis.patch_size,
        ),
        patch_stride=_normalize_positive_int(
            dis_raw.get("patch_stride"),
            "dis.patch_stride",
            defaults.dis.patch_stride,
        ),
        gradient_descent_iterations=_normalize_positive_int(
            dis_raw.get("gradient_descent_iterations"),
            "dis.gradient_descent_iterations",
            defaults.dis.gradient_descent_iterations,
            allow_zero=True,
        ),
        variational_refinement_iterations=_normalize_positive_int(
            dis_raw.get("variational_refinement_iterations"),
            "dis.variational_refinement_iterations",
            defaults.dis.variational_refinement_iterations,
            allow_zero=True,
        ),
        use_spatial_propagation=_normalize_bool(
            dis_raw.get("use_spatial_propagation"),
            "dis.use_spatial_propagation",
            defaults.dis.use_spatial_propagation,
        ),
    )

    sampling = SamplingConfig(
        grid_step_original_px=_normalize_positive_int(
            sampling_raw.get("grid_step_original_px"),
            "sampling.grid_step_original_px",
            defaults.sampling.grid_step_original_px,
        ),
        min_margin_original_px=_normalize_positive_int(
            sampling_raw.get("min_margin_original_px"),
            "sampling.min_margin_original_px",
            defaults.sampling.min_margin_original_px,
            allow_zero=True,
        ),
        lost_border_original_px=_normalize_non_negative_float(
            sampling_raw.get("lost_border_original_px"),
            "sampling.lost_border_original_px",
            defaults.sampling.lost_border_original_px,
        ),
    )

    scoring = ScoringConfig(
        percentile=float(scoring_raw.get("percentile", defaults.scoring.percentile)),
    )

    trigger = TriggerConfig(
        min_frames_since_keyframe=_normalize_positive_int(
            trigger_raw.get("min_frames_since_keyframe"),
            "trigger.min_frames_since_keyframe",
            defaults.trigger.min_frames_since_keyframe,
            allow_zero=True,
        ),
        main_threshold_original_px=_normalize_non_negative_float(
            trigger_raw.get("main_threshold_original_px"),
            "trigger.main_threshold_original_px",
            defaults.trigger.main_threshold_original_px,
        ),
        min_in_bounds_ratio=_normalize_ratio(
            trigger_raw.get("min_in_bounds_ratio"),
            "trigger.min_in_bounds_ratio",
            defaults.trigger.min_in_bounds_ratio,
            upper=1.000001,
        ),
        max_frames_since_keyframe=_normalize_positive_int(
            trigger_raw.get("max_frames_since_keyframe"),
            "trigger.max_frames_since_keyframe",
            defaults.trigger.max_frames_since_keyframe,
            allow_zero=True,
        ),
    )

    visualization = VisualizationConfig(
        show_displacement_vectors=_normalize_bool(
            vis_raw.get("show_displacement_vectors"),
            "visualization.show_displacement_vectors",
            defaults.visualization.show_displacement_vectors,
        ),
        point_radius=_normalize_positive_int(
            vis_raw.get("point_radius"),
            "visualization.point_radius",
            defaults.visualization.point_radius,
        ),
        keyframe_thumbnail_slots=_normalize_positive_int(
            vis_raw.get("keyframe_thumbnail_slots"),
            "visualization.keyframe_thumbnail_slots",
            defaults.visualization.keyframe_thumbnail_slots,
        ),
        preview_max_width=_normalize_positive_int(
            vis_raw.get("preview_max_width"),
            "visualization.preview_max_width",
            defaults.visualization.preview_max_width,
        ),
        save_debug_video=_normalize_bool(
            vis_raw.get("save_debug_video"),
            "visualization.save_debug_video",
            defaults.visualization.save_debug_video,
        ),
        point_color_rgb=_normalize_rgb_color(
            vis_raw.get("point_color_rgb"),
            "visualization.point_color_rgb",
            defaults.visualization.point_color_rgb,
        ),
        motion_plot_color_rgb=_normalize_rgb_color(
            vis_raw.get("motion_plot_color_rgb"),
            "visualization.motion_plot_color_rgb",
            defaults.visualization.motion_plot_color_rgb,
        ),
        points_plot_color_rgb=_normalize_rgb_color(
            vis_raw.get("points_plot_color_rgb"),
            "visualization.points_plot_color_rgb",
            defaults.visualization.points_plot_color_rgb,
        ),
        threshold_line_color_rgb=_normalize_rgb_color(
            vis_raw.get("threshold_line_color_rgb"),
            "visualization.threshold_line_color_rgb",
            defaults.visualization.threshold_line_color_rgb,
        ),
        trigger_line_color_rgb=_normalize_rgb_color(
            vis_raw.get("trigger_line_color_rgb"),
            "visualization.trigger_line_color_rgb",
            defaults.visualization.trigger_line_color_rgb,
        ),
    )

    output = OutputConfig(
        image_format=_normalize_image_format(
            output_raw.get("image_format"),
            defaults.output.image_format,
        ),
    )

    if not (0.0 <= scoring.percentile <= 100.0):
        raise ValueError("scoring.percentile must be in [0, 100]")

    n_downsample = _normalize_positive_int(
        raw.get("n_downsample"),
        "n_downsample",
        defaults.n_downsample,
        allow_zero=True,
    )
    max_step_norm_original_px = _normalize_non_negative_float(
        raw.get("max_step_norm_original_px"),
        "max_step_norm_original_px",
        defaults.max_step_norm_original_px,
    )

    return FrameExtractorConfig(
        dis=dis,
        sampling=sampling,
        scoring=scoring,
        trigger=trigger,
        visualization=visualization,
        output=output,
        n_downsample=n_downsample,
        max_step_norm_original_px=max_step_norm_original_px,
    )


def load_config(path: str | Path) -> FrameExtractorConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return parse_config(_as_mapping(loaded, "root"))


def load_default_config() -> FrameExtractorConfig:
    if REPO_DEFAULT_CONFIG_PATH.exists():
        return load_config(REPO_DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG


def dump_config_yaml(config: FrameExtractorConfig) -> str:
    return yaml.safe_dump(asdict(config), sort_keys=False)
