from __future__ import annotations

import argparse

from config import dump_app_config_yaml, load_app_config, load_default_app_config
from metrics import SurvivalMotionIntervalDecider
from pipeline import TrackingPipeline
from tracker import LKTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Intelligently extract keyframes for SfM workflows to reduce processing time "
            "in COLMAP and similar reconstruction pipelines"
        )
    )
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Defaults to built-in values.",
    )
    parser.add_argument(
        "--keyframe-directory",
        type=str,
        default=None,
        help="Base output directory for timestamped keyframe folder, run metadata, and optional debug video",
    )
    parser.add_argument(
        "--show-flow-vis",
        action="store_true",
        help="Render optical-flow visualization and show the preview window",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to process")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config
    app_config = load_app_config(config_path) if config_path is not None else load_default_app_config()
    tracker = LKTracker(config=app_config.tracking)
    decider = SurvivalMotionIntervalDecider(
        min_survival_percent=app_config.min_survival_percent,
        motion_percentile=app_config.motion_percentile,
        t_motion=app_config.t_motion,
        max_frames_since_keyframe=app_config.max_frames_since_keyframe,
    )

    pipeline = TrackingPipeline(
        args.input_video,
        tracker=tracker,
        keyframe_decider=decider,
        keyframe_directory=args.keyframe_directory,
        effective_config_yaml=dump_app_config_yaml(app_config),
        n_downsample=app_config.n_downsample,
        keyframe_image_format=app_config.keyframe_image_format,
        show_flow_vis=args.show_flow_vis,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    stats = pipeline.run()
    print(
        f"Done. Processed={stats.processed_frames}, "
        f"Resets={stats.keyframe_resets}, "
        f"Saved={stats.keyframes_saved}, "
        f"StoppedByUser={stats.stopped_by_user}, "
        f"Avg FPS={stats.avg_fps:.2f}, "
        f"Runtime={stats.runtime_seconds:.3f}s"
    )
    if stats.keyframe_dir is not None:
        print(f"Keyframe directory: {stats.keyframe_dir}")
    if stats.debug_video_path is not None:
        print(f"Debug video: {stats.debug_video_path}")
    if stats.run_config_path is not None:
        print(f"Run config: {stats.run_config_path}")
    if stats.run_summary_path is not None:
        print(f"Run summary: {stats.run_summary_path}")


if __name__ == "__main__":
    main()
