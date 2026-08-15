#!/usr/bin/env bash
set -euo pipefail

fixture_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
video_path="${fixture_dir}/vfr_timing.mov"
oracle_path="${fixture_dir}/vfr_timing.ffprobe.json"

for command_name in ffmpeg ffprobe; do
    if ! command -v "${command_name}" >/dev/null; then
        echo "Missing required developer command: ${command_name}" >&2
        exit 1
    fi
done

if ! ffmpeg -hide_banner -encoders 2>/dev/null \
    | grep -E '[[:space:]]libx264[[:space:]]' >/dev/null; then
    echo "This optional fixture regeneration requires an FFmpeg build with libx264." >&2
    exit 1
fi

# Retain frames 0, 1, 3, 4, and 7 from a 25 fps source. Their presentation
# times are 0, 40, 120, 160, and 280 ms, giving deliberately uneven gaps.
# H.264 with B-frames and a 600 Hz MOV track time base exercises realistic
# decoder timing behavior more thoroughly than an intra-frame-only fixture.
ffmpeg \
    -y \
    -hide_banner \
    -loglevel error \
    -f lavfi \
    -i 'testsrc2=size=64x48:rate=25:duration=0.36' \
    -vf "select='eq(n,0)+eq(n,1)+eq(n,3)+eq(n,4)+eq(n,7)'" \
    -fps_mode vfr \
    -c:v libx264 \
    -preset veryslow \
    -crf 18 \
    -g 30 \
    -bf 2 \
    -pix_fmt yuv420p \
    -video_track_timescale 600 \
    -an \
    -metadata creation_time=1970-01-01T00:00:00Z \
    "${video_path}"

ffprobe \
    -v error \
    -select_streams v:0 \
    -show_entries 'frame=pts,pts_time,pict_type:stream=codec_name,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration_ts,duration,nb_frames' \
    -of json \
    "${video_path}" >"${oracle_path}"
