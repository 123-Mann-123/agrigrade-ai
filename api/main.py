"""AgriGrade AI FastAPI Application with YOLOv8 inference."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel

os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "0")
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
    category: str
    defect_percentage: float
    shelf_life: str
    salvage_rate: str
    ui_color: str
    commercial_action: str
    agronomic_prevention: str
    alternative_uses: List[str]
    # Backward compatibility aliases
    shelf_life_days: Optional[str] = None
    salvage_value: Optional[str] = None
    salvage_yield: Optional[str] = None
    recommended_action: Optional[str] = None
    status_color: Optional[str] = None
    # Detection telemetry
    has_rotten_detection: Optional[bool] = None
    max_rotten_conf: Optional[float] = None


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

    # 1. Inspect YOLO Detections and Determine Fruit & Rotten Areas
    boxes_list = results[0].boxes if len(results) > 0 and hasattr(results[0], "boxes") else []

    fruit_bbox = None
    largest_box_area = 0

    has_rotten_detection = False
    max_rotten_conf = 0.0
    rotten_box_area = 0

    rotten_mask = np.zeros((h, w), dtype=np.uint8)

    if boxes_list is not None and len(boxes_list) > 0:
        for box in boxes_list:
            cls_id = int(box.cls[0])
            class_name = str(model.names.get(cls_id, cls_id)).lower()
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            b_area = box_w * box_h

            # Track largest detected fruit object bounding box
            if b_area > largest_box_area:
                largest_box_area = b_area
                fruit_bbox = (x1, y1, x2, y2)

            is_rotten_class = (
                "rotten" in class_name
                or "defect" in class_name
                or "damaged" in class_name
                or cls_id == 1
            )

            if is_rotten_class:
                rotten_mask[y1:y2, x1:x2] = 1
                rotten_box_area += b_area

                if conf > 0.40:
                    has_rotten_detection = True
                    max_rotten_conf = max(max_rotten_conf, conf)

    # 2. Determine Fruit Area (avoid dividing by raw camera canvas size)
    if fruit_bbox is not None and largest_box_area > 0:
        fx1, fy1, fx2, fy2 = fruit_bbox
        fruit_crop = img[fy1:fy2, fx1:fx2]
        if fruit_crop.size > 0:
            # Color thresholding / background separation inside bounding box
            hsv_crop = cv2.cvtColor(fruit_crop, cv2.COLOR_BGR2HSV)
            sat = hsv_crop[:, :, 1]
            _, thresh = cv2.threshold(sat, 25, 255, cv2.THRESH_BINARY)
            fruit_pixel_count = int(np.sum(thresh > 0))
            # Use segmented count if significant, otherwise bounding box area
            fruit_area = fruit_pixel_count if fruit_pixel_count > 0.35 * largest_box_area else largest_box_area
        else:
            fruit_area = largest_box_area
    else:
        # Fallback segmentation on full canvas if no box was detected
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, thresh = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
        seg_count = int(np.sum(thresh > 0))
        fruit_area = seg_count if seg_count > 0.05 * total_image_area else total_image_area

    fruit_area = max(1, fruit_area)

    # 3. Calculate Real Defect Ratio Normalized by Fruit Area
    rotten_pixels = int(np.sum(rotten_mask))
    defect_percentage = min(100.0, (rotten_pixels / fruit_area) * 100.0)

    # Conservative bounding-box rot estimation
    rot_box_ratio = (rotten_box_area / fruit_area) * 100.0
    defect_percentage = max(defect_percentage, rot_box_ratio * 0.75)

    # If YOLO confirmed active rot/decay, produce must not be rated Grade A (Export Quality)
    if has_rotten_detection:
        # Minimum defect to reflect early-stage pathogen/fungal damage (landing in Grade B or higher)
        defect_percentage = max(defect_percentage, 8.5)

    defect_percentage = min(100.0, defect_percentage)
    defect_percentage = round(defect_percentage, 2)

    # 4. Evaluate quality using standardized rules engine
    report = evaluate_quality(defect_percentage)
    report["has_rotten_detection"] = has_rotten_detection
    report["max_rotten_conf"] = round(max_rotten_conf, 4)
    return QualityReport(**report)
