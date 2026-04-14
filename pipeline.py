from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional

import cv2
from tqdm.auto import tqdm

from metrics import KeyframeDecider, KeyframeDecision
from tracker import LKTracker
from visualization import (
    append_thumbnail_strip,
    draw_tracks,
    make_keyframe_thumbnail,
    resize_to_width,
    thumbnail_size_for_frame,
    thumbnail_strip_size,
)


MAX_DEBUG_THUMBNAILS = 10


@dataclass
class RunStats:
    processed_frames: int
    avg_fps: float
    runtime_seconds: float
    keyframe_resets: int
    keyframes_saved: int
    stopped_by_user: bool
    keyframe_dir: Optional[str]
    debug_video_path: Optional[str]
    run_config_path: Optional[str]
    run_summary_path: Optional[str]


@dataclass
class OutputSession:
    writer: Optional[cv2.VideoWriter]
    keyframe_dir: Optional[Path]
    debug_video_path: Optional[Path]
    run_config_path: Optional[Path]
    run_summary_path: Optional[Path]
    show_preview: bool
    preview_window_name: str = "frame-extractor debug"


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def _create_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer: {path}")
    return writer


def _even_dimension(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _planned_total_frames(cap: cv2.VideoCapture, start_frame: int, max_frames: Optional[int]) -> Optional[int]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        remaining = max(total - start_frame, 0)
        return remaining if max_frames is None else min(remaining, max_frames)
    return max_frames


def _processing_size(width: int, height: int, n_downsample: int) -> tuple[int, int]:
    out_w, out_h = width, height
    for _ in range(max(0, int(n_downsample))):
        if out_w <= 1 or out_h <= 1:
            break
        out_w = max(1, out_w // 2)
        out_h = max(1, out_h // 2)
    return (out_w, out_h)


def _downsample_for_processing(frame, n_downsample: int):
    out = frame
    for _ in range(max(0, int(n_downsample))):
        h, w = out.shape[:2]
        if w <= 1 or h <= 1:
            break
        out = cv2.pyrDown(out)
    return out


def _start_output_session(
    *,
    keyframe_directory: Optional[str],
    effective_config_yaml: str,
    show_flow_vis: bool,
    fps: float,
    width: int,
    height: int,
) -> OutputSession:
    if not keyframe_directory and not show_flow_vis:
        return _empty_output_session()

    writer = None
    keyframe_dir = None
    debug_video_path = None
    run_config_path = None
    run_summary_path = None
    show_preview = bool(show_flow_vis)

    if keyframe_directory:
        base_dir = Path(keyframe_directory)
        base_dir.mkdir(parents=True, exist_ok=True)
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        keyframe_dir = base_dir / run_name
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        run_config_path = base_dir / f"{run_name}.yaml"
        run_summary_path = base_dir / f"{run_name}.txt"
        if show_flow_vis:
            debug_video_path = base_dir / f"{run_name}.mp4"
            writer = _create_video_writer(str(debug_video_path), fps, width, height)
        run_config_path.write_text(effective_config_yaml, encoding="utf-8")

    session = OutputSession(
        writer=writer,
        keyframe_dir=keyframe_dir,
        debug_video_path=debug_video_path,
        run_config_path=run_config_path,
        run_summary_path=run_summary_path,
        show_preview=show_preview,
    )
    if session.show_preview:
        cv2.namedWindow(session.preview_window_name, cv2.WINDOW_NORMAL)
    return session


def _empty_output_session() -> OutputSession:
    return OutputSession(
        writer=None,
        keyframe_dir=None,
        debug_video_path=None,
        run_config_path=None,
        run_summary_path=None,
        show_preview=False,
    )


def _save_keyframe(
    *,
    keyframe_dir: Optional[Path],
    keyframe_index: int,
    frame_idx: int,
    image_format: str,
    frame,
) -> int:
    if keyframe_dir is None:
        return keyframe_index
    keyframe_name = f"keyframe_{keyframe_index:04d}_{frame_idx:05d}.{image_format}"
    keyframe_path = keyframe_dir / keyframe_name
    ok_write = cv2.imwrite(str(keyframe_path), frame)
    if not ok_write:
        raise IOError(f"Failed to write keyframe image: {keyframe_path}")
    return keyframe_index + 1


def _emit(message: str) -> None:
    # Keep progress bar rendering intact while writing log lines.
    tqdm.write(message)


def _write_run_summary(
    *,
    path: Optional[Path],
    input_video: str,
    start_frame: int,
    max_frames: Optional[int],
    total_video_frames: int,
    planned_processing_frames: Optional[int],
    runtime_seconds: float,
    keyframes_saved: int,
) -> None:
    if path is None:
        return
    summary = (
        f"input_video: {input_video}\n"
        f"start_frame: {start_frame}\n"
        f"max_frames: {max_frames if max_frames is not None else 'None'}\n"
        f"total_video_frames: {total_video_frames}\n"
        f"runtime_seconds: {runtime_seconds:.3f}\n"
        f"frames_extracted: {keyframes_saved} / "
        f"{planned_processing_frames if planned_processing_frames is not None else 'unknown'}\n"
    )
    path.write_text(summary, encoding="utf-8")


class TrackingPipeline:
    def __init__(
        self,
        input_video: str,
        *,
        tracker: LKTracker,
        keyframe_decider: KeyframeDecider,
        keyframe_directory: Optional[str] = None,
        effective_config_yaml: str = "",
        n_downsample: int = 0,
        keyframe_image_format: str = "png",
        show_flow_vis: bool = False,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
    ):
        self.input_video = input_video
        self.tracker = tracker
        self.keyframe_decider = keyframe_decider
        self.keyframe_directory = keyframe_directory
        self.effective_config_yaml = effective_config_yaml
        self.n_downsample = int(n_downsample)
        self.keyframe_image_format = keyframe_image_format
        self.show_flow_vis = bool(show_flow_vis)
        self.start_frame = int(start_frame)
        self.max_frames = None if max_frames is None else int(max_frames)

    def run(self) -> RunStats:
        cap = _open_video(self.input_video)
        pbar = None
        output = _empty_output_session()

        try:
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            proc_width, proc_height = _processing_size(
                in_width,
                in_height,
                self.n_downsample,
            )
            thumbnail_size = thumbnail_size_for_frame(in_width, in_height, scale=16)
            strip_width, strip_height = thumbnail_strip_size(
                thumbnail_size,
                max_thumbnails=MAX_DEBUG_THUMBNAILS,
            )
            render_width = _even_dimension(max(proc_width, strip_width))
            scaled_debug_height = max(1, round(proc_height * (render_width / proc_width)))
            render_height = _even_dimension(scaled_debug_height + strip_height)

            output = _start_output_session(
                keyframe_directory=self.keyframe_directory,
                effective_config_yaml=self.effective_config_yaml,
                show_flow_vis=self.show_flow_vis,
                fps=fps,
                width=render_width,
                height=render_height,
            )

            total_for_progress = _planned_total_frames(cap, self.start_frame, self.max_frames)
            pbar = tqdm(
                total=total_for_progress,
                unit="frame",
                desc="Tracking",
                dynamic_ncols=True,
            )

            frame_idx = self.start_frame
            processed = 0
            keyframe_resets = 0
            keyframes_saved = 0
            recent_keyframe_thumbnails: list = []
            stopped_by_user = False
            t0 = perf_counter()

            try:
                while True:
                    if self.max_frames is not None and processed >= self.max_frames:
                        break

                    ok, frame = cap.read()
                    if not ok:
                        break
                    proc_frame = _downsample_for_processing(frame, self.n_downsample)

                    result = self.tracker.process_frame(proc_frame, frame_idx)

                    current_frame_number = processed + 1

                    # Always keep the first processed frame as keyframe 0.
                    if processed == 0:
                        recent_keyframe_thumbnails.insert(
                            0,
                            make_keyframe_thumbnail(frame, thumbnail_size),
                        )
                        del recent_keyframe_thumbnails[MAX_DEBUG_THUMBNAILS:]
                        keyframes_saved = _save_keyframe(
                            keyframe_dir=output.keyframe_dir,
                            keyframe_index=keyframes_saved,
                            frame_idx=frame_idx,
                            image_format=self.keyframe_image_format,
                            frame=frame,
                        )

                    decision = self.keyframe_decider.evaluate(result, proc_frame.shape[:2])

                    if decision.should_select:
                        # Recompute anchor points on this frame and restart tracking from here.
                        self.tracker.reset_anchor_from_frame(proc_frame)
                        keyframe_resets += 1
                        recent_keyframe_thumbnails.insert(
                            0,
                            make_keyframe_thumbnail(frame, thumbnail_size),
                        )
                        del recent_keyframe_thumbnails[MAX_DEBUG_THUMBNAILS:]
                        keyframes_saved = _save_keyframe(
                            keyframe_dir=output.keyframe_dir,
                            keyframe_index=keyframes_saved,
                            frame_idx=frame_idx,
                            image_format=self.keyframe_image_format,
                            frame=frame,
                        )

                    render_frame = None
                    if output.writer is not None or output.show_preview:
                        render_frame = draw_tracks(
                            proc_frame,
                            result,
                            keyframes_saved=keyframes_saved,
                            decision=decision,
                        )
                        render_frame = resize_to_width(render_frame, render_width)
                        render_frame = append_thumbnail_strip(
                            render_frame,
                            recent_keyframe_thumbnails,
                            output_width=render_width,
                            strip_height=render_height - render_frame.shape[0],
                        )
                    if output.writer is not None and render_frame is not None:
                        output.writer.write(render_frame)

                    if output.show_preview and render_frame is not None:
                        cv2.imshow(output.preview_window_name, render_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            stopped_by_user = True
                            _emit("Stop requested by user (Esc). Finalizing output video...")
                            break
                        if cv2.getWindowProperty(output.preview_window_name, cv2.WND_PROP_VISIBLE) < 1:
                            stopped_by_user = True
                            _emit("Preview window closed by user. Finalizing output video...")
                            break

                    processed += 1
                    frame_idx += 1
                    pbar.update(1)
                    elapsed = perf_counter() - t0
                    pbar.set_postfix_str(
                        f"keyframes={keyframes_saved} "
                        f"frame={current_frame_number}"
                    )
            except KeyboardInterrupt:
                stopped_by_user = True
                _emit("Stop requested by user (Ctrl+C). Finalizing output video...")

            elapsed = perf_counter() - t0
            avg_fps = processed / elapsed if elapsed > 0 else 0.0
            _write_run_summary(
                path=output.run_summary_path,
                input_video=self.input_video,
                start_frame=self.start_frame,
                max_frames=self.max_frames,
                total_video_frames=total_video_frames,
                planned_processing_frames=total_for_progress,
                runtime_seconds=elapsed,
                keyframes_saved=keyframes_saved,
            )
            return RunStats(
                processed_frames=processed,
                avg_fps=avg_fps,
                runtime_seconds=elapsed,
                keyframe_resets=keyframe_resets,
                keyframes_saved=keyframes_saved,
                stopped_by_user=stopped_by_user,
                keyframe_dir=str(output.keyframe_dir) if output.keyframe_dir is not None else None,
                debug_video_path=str(output.debug_video_path) if output.debug_video_path is not None else None,
                run_config_path=str(output.run_config_path) if output.run_config_path is not None else None,
                run_summary_path=str(output.run_summary_path) if output.run_summary_path is not None else None,
            )
        finally:
            if pbar is not None:
                pbar.close()
            if output.show_preview:
                cv2.destroyWindow(output.preview_window_name)
            cap.release()
            if output.writer is not None:
                output.writer.release()
