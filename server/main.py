from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from PIL import Image
import io
import os
from ultralytics import YOLO
from contextlib import asynccontextmanager
import base64

# --- Configuration ---
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/sample.pt"
) # Default path, can be overridden by env var
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
# Define the target class name for counting, case-insensitive
TARGET_CLASS_FOR_COUNT = "car"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model file not found at {MODEL_PATH}. API will not function correctly.")
        model = None
    else:
        try:
            model = YOLO(MODEL_PATH)
            print(f"YOLO model loaded successfully from {MODEL_PATH}")
            if hasattr(model, 'names'):
                print(f"Model classes: {model.names}")
            else:
                print("Warning: Model does not have 'names' attribute. Class names might not be available.")
        except Exception as e:
            print(f"CRITICAL: Error loading YOLO model: {e}. API will not function correctly.")
            model = None
    
    if model is None:
        print("CRITICAL: YOLO Model could not be loaded. API will not function correctly.")
    else:
        print("Application startup complete. YOLO model is ready.")
    yield
    # Clean up resources if needed on shutdown (not strictly necessary for YOLO model here)
    print("Application shutdown.")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="YOLO Car Detection API",
    description="API for detecting cars in images using a YOLOv8 model.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Global variable for the model ---
# Model will be loaded during startup via the lifespan manager
model: Optional[YOLO] = None


# --- Pydantic Models for Request/Response Schemas (for documentation) ---
class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Minimum x-coordinate of the bounding box")
    y_min: float = Field(..., description="Minimum y-coordinate of the bounding box")
    x_max: float = Field(..., description="Maximum x-coordinate of the bounding box")
    y_max: float = Field(..., description="Maximum y-coordinate of the bounding box")

class Detection(BaseModel):
    class_name: str = Field(..., description="Name of the detected class (e.g., 'car')")
    confidence: float = Field(..., description="Confidence score of the detection (0.0 to 1.0)")
    bounding_box: BoundingBox = Field(..., description="Coordinates of the bounding box")

class DetectionResponse(BaseModel):
    detections: List[Detection] = Field(..., description="List of detected objects")
    image_filename: str = Field(..., description="Original filename of the processed image")
    object_count: int = Field(..., description=f"Number of '{TARGET_CLASS_FOR_COUNT}' objects detected in the image")
    annotated_image_base64: Optional[str] = Field(None, description="Base64 encoded string of the image with bounding boxes. Image is in JPEG format.")

# --- API Endpoints ---

@app.get("/health", summary="Health Check", tags=["General"])
async def health_check():
    """
    Endpoint to check the health and status of the API.
    Returns the model loading status.
    """
    if model is None:
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={"status": "error", "message": "YOLO model not loaded."}
        )
    return {"status": "ok", "message": "API is running and model is loaded."}

@app.post("/predict", response_model=DetectionResponse, summary="Detect Objects in Image", tags=["Detection"])
async def predict_objects(file: UploadFile = File(..., description="Image file to process for object detection.")):
    """
    Upload an image and get object detections.

    - **Input**: Image file (JPEG, PNG, etc.)
    - **Output**: JSON with detected objects, their class, confidence, bounding boxes,
                  count of specific objects (e.g., cars), and a base64 encoded image with detections.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model is not loaded. Cannot process requests.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read or open image file: {e}")
    finally:
        await file.close()

    try:
        # Perform inference
        results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False) # verbose=False to reduce console output
    except Exception as e:
        # This could happen if the model is loaded but encounters an issue during prediction
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")


    detections_list = []
    counted_objects = 0
    annotated_image_base64_str = None

    if results and len(results) > 0:
        result = results[0] # Assuming one image was processed
        
        # Get annotated image if there are results
        try:
            pil_annotated_image = result.plot(pil=True) # Returns a PIL Image object
            buffered = io.BytesIO()
            pil_annotated_image.save(buffered, format="JPEG")
            annotated_image_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not generate annotated image: {e}")
            # annotated_image_base64_str will remain None

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        model_class_names = getattr(model, 'names', None)
        if model_class_names is None:
            # Fallback if model.names is not available
            # Ensure class_ids is not empty before calling max
            max_class_id = 0
            if class_ids.size > 0: # Check if numpy array is not empty
                max_class_id = int(max(class_ids)) # Safely get max if class_ids has elements
            model_class_names = {i: f"class_{i}" for i in range(max_class_id + 1)}
            print("Warning: model.names not found, using generic class names.")


        for i in range(len(boxes)):
            class_id = class_ids[i]
            
            # Ensure class_id is a valid key for model_class_names
            if class_id in model_class_names:
                class_name = model_class_names[class_id]
            elif 0 <= class_id < len(model_class_names): # common case for list-like names
                 class_name = model_class_names[class_id]
            else:
                class_name = f"unknown_class_{class_id}"
                print(f"Warning: Detected unknown or out-of-bounds class_id: {class_id}")

            if class_name.lower() == TARGET_CLASS_FOR_COUNT.lower():
                counted_objects += 1

            detection = Detection(
                class_name=class_name,
                confidence=float(confidences[i]),
                bounding_box=BoundingBox(
                    x_min=float(boxes[i][0]),
                    y_min=float(boxes[i][1]),
                    x_max=float(boxes[i][2]),
                    y_max=float(boxes[i][3]),
                )
            )
            detections_list.append(detection)

    return DetectionResponse(
        detections=detections_list,
        image_filename=file.filename,
        object_count=counted_objects,
        annotated_image_base64=annotated_image_base64_str
    )
