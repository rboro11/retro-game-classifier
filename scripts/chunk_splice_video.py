"""
chunk_splice_video.py
---------------------
Takes a full-length gameplay recording and produces a spliced training video
by extracting N evenly-spaced 60-second chunks and concatenating them.

Usage:
    python chunk_splice_video.py --input SMB3_1.mp4 --output training_data_SMB3_1_spliced.mp4
    python chunk_splice_video.py --input SMB3_1.mp4 --output training_data_SMB3_1_spliced.mp4 --chunks 60 --interval 270 --duration 60

Arguments:
    --input      Path to the source gameplay video (required)
    --output     Path for the spliced output video (required)
    --chunks     Number of chunks to extract (default: 60)
    --interval   Seconds between chunk start times (default: 270 = every 4.5 min)
    --duration   Duration of each chunk in seconds (default: 60)
"""

import os
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk and splice a gameplay video for training data.")
    parser.add_argument("--input",    required=True,  help="Path to source video")
    parser.add_argument("--output",   required=True,  help="Path for spliced output video")
    parser.add_argument("--chunks",   type=int, default=60,  help="Number of chunks to extract (default: 60)")
    parser.add_argument("--interval", type=int, default=270, help="Seconds between chunk starts (default: 270)")
    parser.add_argument("--duration", type=int, default=60,  help="Duration of each chunk in seconds (default: 60)")
    return parser.parse_args()


def run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg error:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)


def main():
    args = parse_args()

    video_input  = Path(args.input)
    video_output = Path(args.output)
    list_file    = Path("concat_list.txt")

    if not video_input.exists():
        raise FileNotFoundError(f"Input video not found: {video_input}")

    print(f"Processing video : {video_input}")
    print(f"Output           : {video_output}")
    print(f"Chunks           : {args.chunks} x {args.duration}s every {args.interval}s\n")

    # 1. Clean up any leftover files
    for f in [video_output, list_file, Path("temp_result.mp4")]:
        if f.exists():
            f.unlink()

    # 2. Extract chunks
    chunks = []
    print(f"Extracting {args.chunks} chunks...")

    for i in range(args.chunks):
        start_s    = i * args.interval
        chunk_name = Path(f"chunk_{i:02d}.mp4")

        run_ffmpeg([
            "ffmpeg", "-ss", str(start_s),
            "-t", str(args.duration),
            "-i", str(video_input),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an", "-y", str(chunk_name),
            "-hide_banner", "-loglevel", "error"
        ])

        if chunk_name.exists() and chunk_name.stat().st_size > 0:
            chunks.append(chunk_name)

        if i % 10 == 0:
            print(f"  Progress: {i}/{args.chunks} chunks extracted...")

    print(f"\n{len(chunks)} chunks extracted successfully.")

    # 3. Concatenate chunks
    if not chunks:
        print("❌ No chunks were created. Check your input video and parameters.")
        return

    with open(list_file, "w") as f:
        for chunk in chunks:
            f.write(f"file '{chunk}'\n")

    print("Merging chunks...")
    run_ffmpeg([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy", "-y", str(video_output),
        "-hide_banner", "-loglevel", "error"
    ])

    # 4. Cleanup temp files
    for chunk in chunks:
        chunk.unlink()
    list_file.unlink()

    # 5. Verify output
    if video_output.exists() and video_output.stat().st_size > 1000:
        size_mb = video_output.stat().st_size / 1e6
        print(f"\n✅ Success! Spliced video saved as {video_output} ({size_mb:.2f} MB)")
    else:
        print("\n❌ Error: Output file is missing or empty.")


if __name__ == "__main__":
    main()
