from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from frame_extractor.config import FrameExtractorConfig
from frame_extractor.config import dump_config_yaml
from frame_extractor.output import FrameTiming
from frame_extractor.output import KeyframeRecord
from frame_extractor.output import RunPaths
from frame_extractor.output import RunStats
from frame_extractor.output import TimingValidator
from frame_extractor.output import capture_decoded_frame_index
from frame_extractor.output import capture_raw_frame_timing
from frame_extractor.output import create_video_writer
from frame_extractor.output import make_run_paths
from frame_extractor.output import open_video
from frame_extractor.output import pad_to_even
from frame_extractor.output import resize_to_max_width
from frame_extractor.output import save_keyframe
from frame_extractor.output import video_backend_name
from frame_extractor.output import write_keyframe_manifest
from frame_extractor.output import write_summary
from frame_extractor.preview import History
from frame_extractor.preview import PreviewKeyframe
from frame_extractor.preview import compose_debug_frame
from frame_extractor.preview import history_append
from frame_extractor.preview import render_debug_dashboard
from frame_extractor.preview import render_tracking_view
from frame_extractor.terminal import TerminalProgress
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import FlowStepDiagnostics
from frame_extractor.tracking import TrackingState
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
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be > 0")

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
        if start_frame > 0 and not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(f"Video backend could not seek to start frame {start_frame}")

        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = (
            reported_fps
            if math.isfinite(reported_fps) and reported_fps > 0.0
            else 30.0
        )
        video_backend = video_backend_name(cap)
        timing_validator = TimingValidator(
            reported_fps=reported_fps,
            fallback_fps=fps,
        )
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
        first_frame_raw_timing = capture_raw_frame_timing(
            cap,
            reported_fps=reported_fps,
            video_backend=video_backend,
        )
        first_frame_index = capture_decoded_frame_index(
            cap,
            start_frame,
            required_index=start_frame if start_frame > 0 else None,
        )
        first_frame_timing = timing_validator.observe(
            first_frame_raw_timing,
            decoded_frame_index=first_frame_index,
            processed_index=0,
        )

        processed_first_frame = downsample_frame(first_frame_full, config.n_downsample)
        prev_gray = ensure_gray(processed_first_frame)
        state = initialize_tracking_state(prev_gray, config)
        initial_point_count = int(state.origin_points.shape[0])
        initial_frame_scores = FrameScores(
            frame_index=first_frame_index,
            timestamp_sec=first_frame_timing.effective_timestamp_seconds,
            global_score=0.0,
            in_bounds_points=initial_point_count,
            in_bounds_ratio=1.0 if initial_point_count > 0 else 0.0,
        )
        initial_trigger_decision = TriggerDecision(
            triggered=False,
            reason="none",
            frames_since_keyframe=0,
        )

        virtual_keyframe_index = 0
        keyframe_records: list[KeyframeRecord] = []
        keyframe_thumbnails: list[PreviewKeyframe] = []
        first_keyframe_record = _save_keyframe(
            run_paths,
            first_frame_full,
            config,
            virtual_keyframe_index,
            0,
            first_frame_index,
            first_frame_timing,
            selection_reason="first",
            frame_scores=initial_frame_scores,
        )
        if first_keyframe_record is not None:
            keyframe_records.append(first_keyframe_record)
        if visual_debug_enabled:
            keyframe_thumbnails.append(
                PreviewKeyframe(
                    frame_bgr=processed_first_frame,
                    keyframe_index=virtual_keyframe_index,
                    frame_index=first_frame_index,
                )
            )

        if visual_debug_enabled:
            cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)

        flow_forward_solver = create_dis_flow(config.dis)
        history = History() if visual_debug_enabled else None
        trigger_count = 0
        frames_since_keyframe = 0
        processed_frames = 1
        current_frame_index = first_frame_index
        current_frame_full = first_frame_full
        current_frame_timing = first_frame_timing
        current_frame_scores = initial_frame_scores
        current_trigger_decision = initial_trigger_decision

        start_time = perf_counter()
        terminal_progress = TerminalProgress(
            input_name=Path(input_video).name,
            output_name=run_paths.run_dir.name if run_paths is not None else "none",
            total_frames=progress_total_frames,
            no_output_warning=not save_outputs,
        )
        terminal_progress.update(
            initial_frame_scores,
            current_trigger_decision,
            config=config,
            processed_frames=processed_frames,
            keyframe_count=virtual_keyframe_index + 1,
        )

        while True:
            if max_frames is not None and processed_frames >= max_frames:
                break

            ok, next_frame_full = cap.read()
            if not ok:
                break
            current_frame_full = next_frame_full

            current_frame_raw_timing = capture_raw_frame_timing(
                cap,
                reported_fps=reported_fps,
                video_backend=video_backend,
            )
            current_frame_index = capture_decoded_frame_index(cap, current_frame_index + 1)
            processed_frames += 1
            current_frame_timing = timing_validator.observe(
                current_frame_raw_timing,
                decoded_frame_index=current_frame_index,
                processed_index=processed_frames - 1,
            )

            current_frame = downsample_frame(current_frame_full, config.n_downsample)
            current_gray = ensure_gray(current_frame)

            diagnostics = step_tracking(
                state,
                prev_gray,
                current_gray,
                flow_forward_solver,
                config,
            )
            current_frame_scores = compute_frame_scores(
                state,
                diagnostics,
                current_frame_index,
                current_frame_timing.effective_timestamp_seconds,
                config,
            )
            current_trigger_decision = decide_trigger(
                current_frame_scores,
                frames_since_keyframe + 1,
                config.trigger,
            )
            if current_trigger_decision.triggered:
                trigger_count += 1
            if history is not None:
                history_append(history, current_frame_scores, current_trigger_decision)

            pending_keyframe_index = virtual_keyframe_index + 1
            highlight_latest_thumbnail = False
            if visual_debug_enabled and current_trigger_decision.triggered:
                keyframe_thumbnails.append(
                    PreviewKeyframe(
                        frame_bgr=current_frame,
                        keyframe_index=pending_keyframe_index,
                        frame_index=current_frame_index,
                    )
                )
                thumbnail_limit = max(
                    1,
                    config.visualization.keyframe_thumbnail_slots,
                )
                if len(keyframe_thumbnails) > thumbnail_limit:
                    del keyframe_thumbnails[:-thumbnail_limit]
                highlight_latest_thumbnail = True

            if visual_debug_enabled:
                assert history is not None
                debug_frame = _render_preview_frame(
                    current_frame,
                    state,
                    diagnostics,
                    current_frame_scores,
                    current_trigger_decision,
                    config,
                    keyframe_count=(
                        virtual_keyframe_index
                        + 1
                        + int(current_trigger_decision.triggered)
                    ),
                    history=history,
                    keyframe_thumbnails=keyframe_thumbnails,
                    highlight_latest_thumbnail=highlight_latest_thumbnail,
                )

                if (
                    writer is None
                    and run_paths is not None
                    and run_paths.debug_video_path is not None
                ):
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

            if current_trigger_decision.triggered:
                virtual_keyframe_index += 1
                keyframe_record = _save_keyframe(
                    run_paths,
                    current_frame_full,
                    config,
                    virtual_keyframe_index,
                    processed_frames - 1,
                    current_frame_index,
                    current_frame_timing,
                    selection_reason=current_trigger_decision.display_reason,
                    frame_scores=current_frame_scores,
                )
                if keyframe_record is not None:
                    keyframe_records.append(keyframe_record)
                state = initialize_tracking_state(current_gray, config)
                frames_since_keyframe = 0
            else:
                frames_since_keyframe += 1

            terminal_progress.update(
                current_frame_scores,
                current_trigger_decision,
                config=config,
                processed_frames=processed_frames,
                keyframe_count=virtual_keyframe_index + 1,
            )
            prev_gray = current_gray
            if stopped_by_user:
                break

        runtime_seconds = perf_counter() - start_time
        terminal_progress.finish()
        timing_validation = timing_validator.report()

        if run_paths is not None:
            if keyframe_records[-1].processed_index == processed_frames - 1:
                final_record = keyframe_records[-1]
                keyframe_records[-1] = replace(
                    final_record,
                    selection_reason=_add_selection_reason(
                        final_record.selection_reason,
                        "final",
                    ),
                )
            else:
                virtual_keyframe_index += 1
                final_record = _save_keyframe(
                    run_paths,
                    current_frame_full,
                    config,
                    virtual_keyframe_index,
                    processed_frames - 1,
                    current_frame_index,
                    current_frame_timing,
                    selection_reason="final",
                    frame_scores=current_frame_scores,
                )
                if final_record is not None:
                    keyframe_records.append(final_record)

            write_keyframe_manifest(run_paths.keyframe_manifest_path, keyframe_records)
            write_summary(
                run_paths.summary_path,
                input_video=input_video,
                start_frame=start_frame,
                max_frames=max_frames,
                processed_frames=processed_frames,
                runtime_seconds=runtime_seconds,
                keyframes_saved=len(keyframe_records),
                trigger_count=trigger_count,
                opencv_version=first_frame_timing.opencv_version,
                video_backend=first_frame_timing.video_backend,
                reported_fps=first_frame_timing.reported_fps,
                nominal_fps=fps,
                timing_validation=timing_validation,
            )

        return RunStats(
            processed_frames=processed_frames,
            avg_fps=processed_frames / max(runtime_seconds, 1e-6),
            runtime_seconds=runtime_seconds,
            keyframes_saved=len(keyframe_records),
            trigger_count=trigger_count,
            stopped_by_user=stopped_by_user,
            run_dir=str(run_paths.run_dir) if run_paths is not None else None,
            debug_video_path=(
                str(run_paths.debug_video_path)
                if run_paths is not None and run_paths.debug_video_path
                else None
            ),
            keyframe_manifest_path=(
                str(run_paths.keyframe_manifest_path) if run_paths is not None else None
            ),
            config_path=str(run_paths.config_path) if run_paths is not None else None,
            summary_path=str(run_paths.summary_path) if run_paths is not None else None,
            timing_validation_status=timing_validation.status,
            raw_pos_timeline_valid=timing_validation.raw_pos_timeline_valid,
        )
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if visual_debug_enabled:
            cv2.destroyAllWindows()


def _save_keyframe(
    run_paths: RunPaths | None,
    frame_bgr: np.ndarray,
    config: FrameExtractorConfig,
    keyframe_index: int,
    processed_index: int,
    frame_index: int,
    frame_timing: FrameTiming,
    *,
    selection_reason: str,
    frame_scores: FrameScores,
) -> KeyframeRecord | None:
    if run_paths is None:
        return None
    keyframe_path = save_keyframe(
        run_paths.keyframe_dir,
        frame_bgr,
        config.output.image_format,
        keyframe_index,
        frame_index,
    )
    return KeyframeRecord(
        filename=keyframe_path.relative_to(run_paths.run_dir).as_posix(),
        processed_index=processed_index,
        decoded_frame_index=frame_index,
        pts=frame_timing.pts,
        pos_seconds_raw=frame_timing.pos_seconds_raw,
        timing_status=frame_timing.timing_status,
        selection_reason=selection_reason,
        motion_score_px=frame_scores.global_score,
        in_bounds_ratio=frame_scores.in_bounds_ratio,
    )


def _add_selection_reason(existing_reason: str, added_reason: str) -> str:
    reasons = existing_reason.split("+")
    if added_reason not in reasons:
        reasons.append(added_reason)
    return "+".join(reasons)


def _render_preview_frame(
    current_frame: np.ndarray,
    state: TrackingState,
    diagnostics: FlowStepDiagnostics,
    frame_scores: FrameScores,
    trigger_decision: TriggerDecision,
    config: FrameExtractorConfig,
    keyframe_count: int,
    history: History,
    keyframe_thumbnails: list[PreviewKeyframe],
    highlight_latest_thumbnail: bool,
) -> np.ndarray:
    tracking_view = render_tracking_view(
        current_frame,
        state,
        diagnostics,
        frame_scores,
        trigger_decision,
        config,
        keyframe_count,
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
