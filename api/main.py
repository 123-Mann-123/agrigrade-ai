"""AgriGrade AI FastAPI Application with YOLOv8 inference."""

from pathlib import Path
from typing import Any, Dict

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

from api.grading_rules import evaluate_quality

app = FastAPI(
    title="AgriGrade AI API",
    description="Backend API for AgriGrade AI quality grading and defect assessment",
    version="1.0.0",
)

# CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model on startup
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "best.pt"
if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH = Path("weights/best.pt")

model = YOLO(str(WEIGHTS_PATH))


class QualityReport(BaseModel):
    grade: str
    defect_percentage: float
    shelf_life_days: str
    salvage_value: str
    recommended_action: str
    status_color: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/grade", response_model=QualityReport)
async def grade_fruit(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    allowed_content_types = {"image/jpeg", "image/png", "image/jpg"}

    if extension not in allowed_extensions and (file.content_type not in allowed_content_types):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are JPG, JPEG, and PNG.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Convert uploaded bytes to OpenCV image array
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded image file.",
        )

    h, w = img.shape[:2]
    total_image_area = h * w

    # Run model inference
    results = model(img)

    # Calculate real defect percentage based on detected Rotten bounding boxes
    rotten_mask = np.zeros((h, w), dtype=np.uint8)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, "")
            if str(cls_name).lower() == "rotten" or cls_id == 1:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                rotten_mask[y1:y2, x1:x2] = 1

    rotten_pixel_count = int(np.sum(rotten_mask))
    if total_image_area > 0:
        defect_percentage = round((rotten_pixel_count / total_image_area) * 100.0, 2)
    else:
        defect_percentage = 0.0

    # Evaluate quality using the grading rules engine
    report = evaluate_quality(defect_percentage)
    return QualityReport(**report)
