import cv2
import os
import shutil
import argparse
from pathlib import Path

def extract_frames_from_video(video_path, output_base_dir, frame_skip):
    """
    Extracts frames from a single video file and saves them into a subdirectory
    named after the video file (without extension).
    """
    video_filename = Path(video_path).stem # Get filename without extension
    video_output_dir = os.path.join(output_base_dir, video_filename)

    if os.path.exists(video_output_dir):
        print(f"Output directory {video_output_dir} already exists. Clearing it.")
        shutil.rmtree(video_output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    print(f"Processing video: {video_path}")
    print(f"Saving frames to: {video_output_dir}")

    cap = cv2.VideoCapture(str(video_path)) # Path object needs to be str
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(video_output_dir, f'frame_{saved_frame_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1
        if frame_count % 100 == 0: # Log progress every 100 processed frames
            print(f"  Video '{video_filename}': Processed {frame_count} frames, saved {saved_frame_count} frames.")

    cap.release()
    print(f"  Video '{video_filename}': Finished. Extracted {saved_frame_count} frames.")
    return saved_frame_count

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos in a directory.")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing video files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_raw_frames",
        help="Directory where extracted frames will be saved (in subfolders per video)."
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=10,
        help="Extract every Nth frame. Default is 10."
    )
    parser.add_argument(
        "--video_extensions",
        nargs='+', # Allows multiple extensions
        default=['.mp4', '.avi', '.mov', '.mkv'],
        help="List of video file extensions to process (e.g., .mp4 .avi). Default: .mp4 .avi .mov .mkv"
    )

    args = parser.parse_args()

    input_video_dir = Path(args.video_dir)
    output_frames_base_dir = Path(args.output_dir)

    if not input_video_dir.is_dir():
        print(f"Error: Input video directory '{input_video_dir}' not found or is not a directory.")
        return

    os.makedirs(output_frames_base_dir, exist_ok=True)

    total_frames_extracted_all_videos = 0
    video_files_processed = 0

    print(f"Scanning for videos in: {input_video_dir}")
    print(f"Looking for extensions: {args.video_extensions}")

    for video_file_path in input_video_dir.rglob('*'): # rglob searches recursively
        if video_file_path.is_file() and video_file_path.suffix.lower() in args.video_extensions:
            print("-" * 30)
            num_extracted = extract_frames_from_video(
                video_file_path,
                output_frames_base_dir,
                args.frame_skip
            )
            total_frames_extracted_all_videos += num_extracted
            video_files_processed += 1
            print("-" * 30)

    if video_files_processed == 0:
        print("No video files found with the specified extensions in the directory.")
    else:
        print(f"\nFinished processing all videos.")
        print(f"Processed {video_files_processed} video file(s).")
        print(f"Total frames extracted across all videos: {total_frames_extracted_all_videos}")
        print(f"All frames saved under base directory: {output_frames_base_dir}")

if __name__ == "__main__":
    main()
