from __future__ import annotations

import argparse

from frame_extractor.config import load_config
from frame_extractor.config import load_default_config
from frame_extractor.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract SfM-friendly keyframes from video using dense DIS optical flow."
    )
    parser.add_argument("input_video", type=str, help="Path to the input video")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config. Defaults to built-in DIS keyframe settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional base output directory for logs, debug video, and keyframes",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Display a live debug window while processing",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to process",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--duration-frames",
        type=int,
        default=None,
        help="Alias for --max-frames: process this many frames starting at --start-frame",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.duration_frames is not None and args.max_frames is not None and args.duration_frames != args.max_frames:
        raise ValueError("Use either --max-frames or --duration-frames, or pass the same value to both.")
    max_frames = args.duration_frames if args.duration_frames is not None else args.max_frames

    try:
        config = load_config(args.config) if args.config is not None else load_default_config()
    except FileNotFoundError as exc:
        message = str(exc)
        if args.config == "configs/dis_flow_experiment.yaml":
            message += "\nThe DIS config is now the default config. Use --config configs/default.yaml, or omit --config."
        raise SystemExit(message) from None

    run_experiment(
        args.input_video,
        config,
        output_dir=args.output_dir,
        show_preview=args.show_preview,
        start_frame=args.start_frame,
        max_frames=max_frames,
    )


if __name__ == "__main__":
    main()
