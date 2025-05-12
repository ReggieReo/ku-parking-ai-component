from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple
from PIL import Image
import io
import os
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/best.pt"
) # Default path, can be overridden by env var
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))

# --- Initialize FastAPI App ---
app = FastAPI(
    title="YOLO Car Detection API",
    description="API for detecting cars in images using a YOLOv8 model.",
    version="1.0.0",
)

# --- Load YOLO Model ---
# Ensure the model path is correct and the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    # You might want to raise an exception or handle this more gracefully
    # For now, we'll let it fail loudly if the model isn't there when YOLO tries to load.
    model = None # Or handle startup failure
else:
    try:
        model = YOLO(MODEL_PATH)
        # Perform a dummy prediction to ensure model is loaded correctly (optional)
        # model.predict(Image.new('RGB', (640, 480)), verbose=False)
        print(f"YOLO model loaded successfully from {MODEL_PATH}")
        if hasattr(model, 'names'):
            print(f"Model classes: {model.names}")
        else:
            print("Warning: Model does not have 'names' attribute. Class names might not be available.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        model = None # Set model to None if loading fails

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

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    if model is None:
        print("CRITICAL: YOLO Model could not be loaded. API will not function correctly.")
    else:
        print("Application startup complete. YOLO model is ready.")

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
    Upload an image and get object detections (specifically cars, based on your model).

    - **Input**: Image file (JPEG, PNG, etc.)
    - **Output**: JSON with detected objects, their class, confidence, and bounding boxes.
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
    if results and len(results) > 0:
        # Ultralytics YOLOv8 returns a list of Results objects
        result = results[0] # Assuming one image was processed
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
        confidences = result.boxes.conf.cpu().numpy() # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int) # Class IDs

        # Ensure model.names is available and class_ids are valid
        model_class_names = getattr(model, 'names', None)
        if model_class_names is None:
            # Fallback if model.names is not available
            model_class_names = {i: f"class_{i}" for i in range(max(class_ids) + 1)}
            print("Warning: model.names not found, using generic class names.")


        for i in range(len(boxes)):
            class_id = class_ids[i]
            if class_id < len(model_class_names):
                class_name = model_class_names[class_id]
            else:
                class_name = f"unknown_class_{class_id}" # Handle unexpected class_id
                print(f"Warning: Detected unknown class_id: {class_id}")


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
        image_filename=file.filename
    )

# To run the app (save this as main.py):
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Then you can access the API docs at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
