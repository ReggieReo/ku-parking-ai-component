from YOLOv8_Explainer import yolov8_heatmap, display_images
import os
from datetime import datetime

# Create output directory if it doesn't exist
os.makedirs("gran-cam-output", exist_ok=True)

model = yolov8_heatmap(
    weight="./mlartifacts/345266831538558357/71fa5455ea8e4bfda93abb0b28b24ad3/artifacts/weights/best.pt", 
        conf_threshold=0.4,  
        method = "EigenGradCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        ratio=0.02,
        # show_box=True,
        show_box=False,
        renormalize=False,
)

# Use an image with multiple car annotations
imagelist = model(
    img_path="dataset/images/train/SCR-20250509-tuiv_jpeg.rf.836092c97235f310261bb5e44c85f9a7.jpg"
    ) # image with multiple cars

# Try to display images, but continue even if it fails (when no cars detected)
try:
    display_images(imagelist)
    print("Successfully displayed images")
except Exception as e:
    print(f"Could not display images: {e}")
    print("This is expected when no cars are detected in the image")

# Save images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for i, img in enumerate(imagelist):
    if img is not None:  # Only try to save if the image is not None
        try:
            output_path = os.path.join("gran-cam-output", f"output_{timestamp}_{i}.jpg")
            img.save(output_path)
            print(f"Saved image to: {output_path}")
        except Exception as e:
            print(f"Error saving image {i}: {e}")
    else:
        print(f"Skipping image {i} as it is None (no cars detected)")
