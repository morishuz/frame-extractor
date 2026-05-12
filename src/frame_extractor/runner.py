from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from frame_extractor.config import FrameExtractorConfig
from frame_extractor.config import dump_config_yaml
from frame_extractor.output import RunPaths
from frame_extractor.output import RunStats
from frame_extractor.output import create_video_writer
from frame_extractor.output import make_run_paths
from frame_extractor.output import open_video
from frame_extractor.output import pad_to_even
from frame_extractor.output import resize_to_max_width
from frame_extractor.output import save_keyframe
from frame_extractor.output import write_summary
from frame_extractor.preview import History
from frame_extractor.preview import PreviewKeyframe
from frame_extractor.preview import compose_debug_frame
from frame_extractor.preview import history_append
from frame_extractor.preview import render_debug_dashboard
from frame_extractor.preview import render_tracking_view
from frame_extractor.terminal import TerminalProgress
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import TriggerDecision
from frame_extractor.tracking import compute_frame_scores
from frame_extractor.tracking import create_dis_flow
from frame_extractor.tracking import decide_trigger
from frame_extractor.tracking import downsample_frame
from frame_extractor.tracking import ensure_gray
from frame_extractor.tracking import initialize_tracking_state
from frame_extractor.tracking import step_tracking


def run_experiment(
    input_video: str,
    config: FrameExtractorConfig,
    *,
    output_dir: str | None,
    show_preview: bool,
    start_frame: int,
    max_frames: int | None,
) -> RunStats:
    visual_debug_enabled = bool(show_preview)
    save_outputs = output_dir is not None
    save_debug_video = save_outputs and visual_debug_enabled and config.visualization.save_debug_video

    run_paths = make_run_paths(output_dir, save_debug_video) if output_dir is not None else None
    if run_paths is not None:
        run_paths.config_path.write_text(dump_config_yaml(config), encoding="utf-8")

    cap = open_video(input_video)
    preview_window_name = "Frame Extractor"
    writer: cv2.VideoWriter | None = None
    stopped_by_user = False

    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 30.0
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            progress_total_frames = max_frames
        elif source_frame_count > start_frame:
            progress_total_frames = source_frame_count - start_frame
        else:
            progress_total_frames = None

        ok, first_frame_full = cap.read()
        if not ok:
            raise RuntimeError("Could not read the first frame from the video")

        processed_first_frame = downsample_frame(first_frame_full, config.n_downsample)
        prev_gray = ensure_gray(processed_first_frame)
        state = initialize_tracking_state(prev_gray, config)

        virtual_keyframe_index = 0
        saved_keyframe_count = 0
        keyframe_thumbnails: list[PreviewKeyframe] = []
        saved_keyframe_count += _save_keyframe(
            run_paths,
            first_frame_full,
            config,
            virtual_keyframe_index,
            start_frame,
        )
        if visual_debug_enabled:
            keyframe_thumbnails.append(
                PreviewKeyframe(
                    frame_bgr=processed_first_frame,
                    keyframe_index=virtual_keyframe_index,
                    frame_index=start_frame,
                )
            )

        if visual_debug_enabled:
            cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)

        flow_forward_solver = create_dis_flow(config.dis)
        history = History()
        frames_since_keyframe = 0
        processed_frames = 1
        current_frame_index = start_frame

        start_time = perf_counter()
        terminal_progress = TerminalProgress(
            input_name=Path(input_video).name,
            output_name=run_paths.run_dir.name if run_paths is not None else "none",
            total_frames=progress_total_frames,
            no_output_warning=not save_outputs,
        )
        initial_point_count = int(state.origin_points.shape[0])
        terminal_progress.update(
            FrameScores(
                frame_index=current_frame_index,
                timestamp_sec=current_frame_index / max(fps, 1e-6),
                global_score=0.0,
                in_bounds_points=initial_point_count,
                in_bounds_ratio=1.0 if initial_point_count > 0 else 0.0,
            ),
            TriggerDecision(
                triggered=False,
                reason="none",
                frames_since_keyframe=frames_since_keyframe,
            ),
            config=config,
            processed_frames=processed_frames,
            keyframe_count=virtual_keyframe_index + 1,
        )

        while True:
            if max_frames is not None and processed_frames >= max_frames:
                break

            ok, current_frame_full = cap.read()
            if not ok:
                break

            current_frame_index += 1
            processed_frames += 1

            current_frame = downsample_frame(current_frame_full, config.n_downsample)
            current_gray = ensure_gray(current_frame)

            diagnostics = step_tracking(
                state,
                prev_gray,
                current_gray,
                flow_forward_solver,
                config,
            )
            timestamp_sec = current_frame_index / max(fps, 1e-6)
            frame_scores = compute_frame_scores(
                state,
                diagnostics,
                current_frame_index,
                timestamp_sec,
                config,
            )
            trigger_decision = decide_trigger(frame_scores, frames_since_keyframe + 1, config.trigger)
            history_append(history, frame_scores, trigger_decision)

            pending_keyframe_index = virtual_keyframe_index + 1
            highlight_latest_thumbnail = False
            if visual_debug_enabled and trigger_decision.triggered:
                keyframe_thumbnails.append(
                    PreviewKeyframe(
                        frame_bgr=current_frame,
                        keyframe_index=pending_keyframe_index,
                        frame_index=current_frame_index,
                    )
                )
                highlight_latest_thumbnail = True

            if visual_debug_enabled:
                debug_frame = _render_preview_frame(
                    current_frame,
                    state,
                    diagnostics,
                    frame_scores,
                    trigger_decision,
                    config,
                    history=history,
                    keyframe_thumbnails=keyframe_thumbnails,
                    highlight_latest_thumbnail=highlight_latest_thumbnail,
                )

                if writer is None and run_paths is not None and run_paths.debug_video_path is not None:
                    writer = create_video_writer(
                        run_paths.debug_video_path,
                        fps,
                        debug_frame.shape[1],
                        debug_frame.shape[0],
                    )
                if writer is not None:
                    writer.write(debug_frame)

                cv2.imshow(preview_window_name, debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    stopped_by_user = True
                    break

            if trigger_decision.triggered:
                virtual_keyframe_index += 1
                saved_keyframe_count += _save_keyframe(
                    run_paths,
                    current_frame_full,
                    config,
                    virtual_keyframe_index,
                    current_frame_index,
                )
                state = initialize_tracking_state(current_gray, config)
                frames_since_keyframe = 0
            else:
                frames_since_keyframe += 1

            terminal_progress.update(
                frame_scores,
                trigger_decision,
                config=config,
                processed_frames=processed_frames,
                keyframe_count=virtual_keyframe_index + 1,
            )
            prev_gray = current_gray

        runtime_seconds = perf_counter() - start_time
        terminal_progress.finish()

        if run_paths is not None:
            write_summary(
                run_paths.summary_path,
                input_video=input_video,
                start_frame=start_frame,
                max_frames=max_frames,
                processed_frames=processed_frames,
                runtime_seconds=runtime_seconds,
                keyframes_saved=saved_keyframe_count,
                trigger_count=len(history.trigger_frames),
            )

        return RunStats(
            processed_frames=processed_frames,
            avg_fps=processed_frames / max(runtime_seconds, 1e-6),
            runtime_seconds=runtime_seconds,
            keyframes_saved=saved_keyframe_count,
            trigger_count=len(history.trigger_frames),
            stopped_by_user=stopped_by_user,
            run_dir=str(run_paths.run_dir) if run_paths is not None else None,
            debug_video_path=(
                str(run_paths.debug_video_path)
                if run_paths is not None and run_paths.debug_video_path
                else None
            ),
            config_path=str(run_paths.config_path) if run_paths is not None else None,
            summary_path=str(run_paths.summary_path) if run_paths is not None else None,
        )
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if visual_debug_enabled:
            cv2.destroyAllWindows()


def _save_keyframe(
    run_paths: RunPaths | None,
    frame_bgr,
    config: FrameExtractorConfig,
    keyframe_index: int,
    frame_index: int,
) -> int:
    if run_paths is None:
        return 0
    save_keyframe(
        run_paths.keyframe_dir,
        frame_bgr,
        config.output.image_format,
        keyframe_index,
        frame_index,
    )
    return 1


def _render_preview_frame(
    current_frame,
    state,
    diagnostics,
    frame_scores: FrameScores,
    trigger_decision: TriggerDecision,
    config: FrameExtractorConfig,
    history: History,
    keyframe_thumbnails: list[PreviewKeyframe],
    highlight_latest_thumbnail: bool,
):
    tracking_view = render_tracking_view(
        current_frame,
        state,
        diagnostics,
        frame_scores,
        trigger_decision,
        config,
        len(keyframe_thumbnails),
    )
    dashboard = render_debug_dashboard(
        width=tracking_view.shape[1],
        history=history,
        config=config,
        thumbnails=keyframe_thumbnails,
        highlight_latest_thumbnail=highlight_latest_thumbnail,
    )
    debug_frame = compose_debug_frame(tracking_view, dashboard)
    debug_frame = resize_to_max_width(debug_frame, config.visualization.preview_max_width)
    return pad_to_even(debug_frame)
