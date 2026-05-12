from __future__ import annotations

import sys
from dataclasses import dataclass

from frame_extractor.config import FrameExtractorConfig
from frame_extractor.tracking import FrameScores
from frame_extractor.tracking import TriggerDecision
from frame_extractor.tracking import comparison_label


ANSI_CLEAR_LINE = "\x1b[2K"
ANSI_RED = "\x1b[31m"
ANSI_RESET = "\x1b[0m"


@dataclass
class TerminalProgress:
    input_name: str
    output_name: str
    total_frames: int | None
    no_output_warning: bool
    started: bool = False
    rendered_line_count: int = 0

    def update(
        self,
        frame_scores: FrameScores,
        trigger_decision: TriggerDecision,
        *,
        config: FrameExtractorConfig,
        processed_frames: int,
        keyframe_count: int,
    ) -> None:
        if self.started:
            sys.stdout.write(f"\x1b[{self.rendered_line_count}F")
        lines = _terminal_status_lines(
            frame_scores,
            trigger_decision,
            config=config,
            keyframe_count=keyframe_count,
            input_name=self.input_name,
            output_name=self.output_name,
            no_output_warning=self.no_output_warning,
        )
        lines.append(_progress_bar(processed_frames, self.total_frames))
        for line in lines:
            sys.stdout.write(f"{ANSI_CLEAR_LINE}{line}\n")
        sys.stdout.flush()
        self.started = True
        self.rendered_line_count = len(lines)

    def finish(self) -> None:
        if self.started:
            sys.stdout.flush()


def _terminal_status_lines(
    frame_scores: FrameScores,
    trigger_decision: TriggerDecision,
    *,
    config: FrameExtractorConfig,
    keyframe_count: int,
    input_name: str,
    output_name: str,
    no_output_warning: bool,
) -> list[str]:
    file_line = f"input: {input_name}    output: {output_name}"
    if no_output_warning:
        file_line += f"    {ANSI_RED}WARNING: no --output-dir, no files written{ANSI_RESET}"
    return [
        file_line,
        f"frame: {frame_scores.frame_index}    time: {frame_scores.timestamp_sec:.3f}s",
        f"frames since last trigger: {trigger_decision.frames_since_keyframe}",
        f"number of keyframes: {keyframe_count}",
        f"motion: {comparison_label(frame_scores.global_score, config.trigger.main_threshold_original_px)}",
        f"points: {comparison_label(frame_scores.in_bounds_ratio, config.trigger.min_in_bounds_ratio)}",
        f"trigger: {trigger_decision.display_reason}",
    ]


def _progress_bar(processed_frames: int, total_frames: int | None, *, width: int = 42) -> str:
    if total_frames is None or total_frames <= 0:
        return f"[{'?' * width}] {processed_frames} frames"
    progress = min(max(processed_frames / max(total_frames, 1), 0.0), 1.0)
    filled = int(round(progress * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {processed_frames}/{total_frames} frames ({progress * 100.0:5.1f}%)"
