"""
splice_video.py — Trim and sample a large gameplay recording into a
                   shorter training-ready video.

Useful when a raw recording is too large for direct frame extraction
(e.g. memory errors with moviepy on a 4+ hour video in Colab).

Usage:
    python scripts/splice_video.py \\
        --input  data/raw/<ClassName>/recording.mp4 \\
        --output data/raw/<ClassName>/training_clip.mp4 \\
        --n_clips 60 \\
        --clip_duration 60

Arguments:
    --input          Path to the source video file
    --output         Path for the spliced output video
    --n_clips        Number of evenly-spaced clips to extract  (default: 60)
    --clip_duration  Duration of each extracted clip in seconds (default: 60)
    --source_duration  Total duration of the source video in seconds.
                     If omitted, moviepy will detect it automatically.

Example (mirrors the original 4.5-hour → 1-hour workflow):
    python scripts/splice_video.py \\
        --input  data/raw/ClassA/raw_recording.mp4 \\
        --output data/raw/ClassA/training_data_1hr.mp4 \\
        --n_clips 60 \\
        --clip_duration 60
"""

import argparse
from pathlib import Path


def splice_video(
    input_path: Path,
    output_path: Path,
    n_clips: int = 60,
    clip_duration: float = 60.0,
    source_duration: float | None = None,
) -> None:
    """
    Extract n_clips evenly-spaced clips of clip_duration seconds each
    from input_path and concatenate them into output_path.

    Parameters
    ----------
    input_path      : Path to the source video.
    output_path     : Destination path for the spliced video.
    n_clips         : How many clips to extract.
    clip_duration   : Length of each clip in seconds.
    source_duration : Total length of source video in seconds.
                      Auto-detected if not supplied.
    """
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        raise SystemExit(
            "moviepy is required.  Install it with:  pip install moviepy"
        )

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[splice_video] Loading: {input_path}")
    video = VideoFileClip(str(input_path))

    total_duration = source_duration if source_duration else video.duration
    print(f"  Source duration : {total_duration:.1f}s  "
          f"({total_duration / 3600:.2f}h)")
    print(f"  Clips to extract: {n_clips} × {clip_duration}s each")

    interval = total_duration / n_clips
    print(f"  Interval between clip starts: {interval:.1f}s")

    clips = []
    for i in range(n_clips):
        start = i * interval
        end = min(start + clip_duration, total_duration)
        clips.append(video.subclip(start, end))
        print(f"  Clip {i + 1:>3}/{n_clips}  [{start:.1f}s – {end:.1f}s]")

    print(f"\n  Concatenating {len(clips)} clips...")
    final_video = concatenate_videoclips(clips)

    print(f"  Writing output: {output_path}")
    final_video.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        logger="bar",
    )

    # Release resources
    final_video.close()
    video.close()

    output_duration = final_video.duration if hasattr(final_video, "duration") else n_clips * clip_duration
    print(f"\n  Done.  Output saved to: {output_path}")
    print(f"  Approximate output duration: {n_clips * clip_duration:.0f}s "
          f"({n_clips * clip_duration / 3600:.2f}h)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retro Game Classifier — Video Splicing Utility"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the source video file"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Destination path for the spliced output video"
    )
    parser.add_argument(
        "--n_clips", type=int, default=60,
        help="Number of evenly-spaced clips to extract (default: 60)"
    )
    parser.add_argument(
        "--clip_duration", type=float, default=60.0,
        help="Duration of each clip in seconds (default: 60)"
    )
    parser.add_argument(
        "--source_duration", type=float, default=None,
        help="Total source video duration in seconds (auto-detected if omitted)"
    )
    args = parser.parse_args()

    splice_video(
        input_path=args.input,
        output_path=args.output,
        n_clips=args.n_clips,
        clip_duration=args.clip_duration,
        source_duration=args.source_duration,
    )


if __name__ == "__main__":
    main()
