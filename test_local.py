import cv2
import os
import argparse
from ultralytics import YOLO
from pathlib import Path

def draw_boxes_on_image(image, results, model_names):
    """
    Draws bounding boxes and labels on an image based on YOLO results.
    This function provides more control than results[0].plot() if needed,
    but for most cases, results[0].plot() is simpler.
    """
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model_names[cls]} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put label
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    return image


def process_image(model, image_path, output_dir, conf_threshold):
    """
    Processes a single image, performs inference, and saves the annotated image.
    """
    print(f"Processing image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Perform inference
    # Setting verbose=False to reduce console output during prediction
    results = model.predict(source=img, conf=conf_threshold, verbose=False)

    # Ultralytics' plot() method is very convenient for drawing
    annotated_img = results[0].plot() # This returns a BGR numpy array with boxes

    # --- OR ---
    # If you want more custom drawing, use the draw_boxes_on_image function:
    # img_copy_for_custom_drawing = img.copy()
    # annotated_img = draw_boxes_on_image(img_copy_for_custom_drawing, results, model.names)
    # --- --- ---

    # Save the annotated image
    output_filename = f"{image_path.stem}_annotated{image_path.suffix}"
    output_path = Path(output_dir) / output_filename
    cv2.imwrite(str(output_path), annotated_img)
    print(f"Saved annotated image to: {output_path}")

    # Optionally display the image
    # cv2.imshow("Annotated Image", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def process_video(
    model,
    video_path,
    output_dir,
    conf_threshold,
    save_as_video=False,
    save_frames=False,
    frame_skip=1,
):
    """
    Processes a video, performs inference frame by frame,
    and saves annotated frames or an annotated video.
    """
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = None
    out_video_writer = None

    if save_as_video:
        output_video_filename = f"{video_path.stem}_annotated.mp4"
        output_video_path = Path(output_dir) / output_video_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Codec
        out_video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (frame_width, frame_height)
        )
        print(f"Will save annotated video to: {output_video_path}")

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Perform inference
            results = model.predict(
                source=frame, conf=conf_threshold, verbose=False
            )
            annotated_frame = results[0].plot() # BGR numpy array

            if save_frames:
                frame_filename = f"{video_path.stem}_frame_{saved_frame_count:05d}_annotated.jpg"
                frame_output_path = Path(output_dir) / "frames" / frame_filename
                frame_output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_output_path), annotated_frame)
                saved_frame_count += 1

            if out_video_writer:
                out_video_writer.write(annotated_frame)

            # Optionally display the frame (can be slow for videos)
            # cv2.imshow("Annotated Video", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        frame_count += 1
        if frame_count % (fps * 2) == 0: # Print progress every 2 seconds of video
            print(f"Processed {frame_count} frames...")


    cap.release()
    if out_video_writer:
        out_video_writer.release()
        print(f"Finished saving annotated video: {output_video_path}")
    if save_frames:
        print(f"Saved {saved_frame_count} annotated frames to: {Path(output_dir) / 'frames'}")
    # cv2.destroyAllWindows()
    print("Video processing complete.")


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on images or videos.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained YOLOv8 model (.pt file)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input image or video file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_predictions",
        help="Directory to save annotated images/videos."
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.35, # Adjusted default confidence
        help="Confidence threshold for detections."
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="For video input, save the output as an annotated video file."
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="For video input, save individual annotated frames as images."
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1, # Process every frame by default
        help="For video input, process every Nth frame (e.g., 5 to process 1 every 5 frames)."
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the YOLO model
    try:
        model = YOLO(str(model_path))
        print(f"Successfully loaded model from {model_path}")
        print(f"Model class names: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine if input is an image or video based on extension
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    input_ext = input_path.suffix.lower()

    if input_ext in image_extensions:
        process_image(model, input_path, output_dir, args.conf_threshold)
    elif input_ext in video_extensions:
        if not args.save_video and not args.save_frames:
            print("For video input, please specify --save_video or --save_frames to get output.")
            # Defaulting to save frames if nothing is specified for video
            args.save_frames = True
            print("Defaulting to --save_frames for video output.")

        process_video(
            model,
            input_path,
            output_dir,
            args.conf_threshold,
            save_as_video=args.save_video,
            save_frames=args.save_frames,
            frame_skip=args.frame_skip
        )
    else:
        print(f"Error: Unsupported file type for input: {input_path.name}")
        print(f"Supported image types: {image_extensions}")
        print(f"Supported video types: {video_extensions}")

if __name__ == "__main__":
    main()
