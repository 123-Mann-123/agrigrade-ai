"""AgriGrade AI FastAPI Application."""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

    # Simulated realistic defect percentage until YOLO model integration
    mock_defect_percentage = 13.8
    report = evaluate_quality(mock_defect_percentage)
    return QualityReport(**report)
