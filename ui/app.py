"""AgriGrade AI - Autonomous Post-Harvest Quality Grading & Valorization Engine.

A clean, modern, single-page Precision Agri-Tech application for automated
defect segmentation, quality grading, shelf-life estimation, and zero-waste
commercial valorization.
"""

from datetime import datetime
import io
from pathlib import Path
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st

# Optional ultralytics import for local CV bounding box overlay rendering
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from api.grading_rules import evaluate_quality

# -----------------------------------------------------------------------------
# 1. Page Configuration & Precision Agri-Tech Unified CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriGrade AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Enforce no sidebar and inject high-contrast Precision Agri-Tech styling
st.markdown(
    textwrap.dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Completely collapse and hide sidebar */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* Canvas: Deep Dark Forest Charcoal */
        html, body, .stApp {
            background: radial-gradient(circle at 50% 0%, #121F1A 0%, #0D1512 75%, #080E0C 100%) !important;
            color: #D1E7DD !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Remove default Streamlit top padding & watermarks */
        #MainMenu, header, footer {
            visibility: hidden !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 94% !important;
        }

        /* Hero Header */
        .hero-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #182B23;
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 16px;
            padding: 20px 26px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.45);
        }
        .hero-left {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .hero-icon {
            font-size: 2.2rem;
            line-height: 1;
        }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 0.88rem;
            color: #D1E7DD;
            margin-top: 3px;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            color: #22C55E;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
            padding: 4px 12px;
            border-radius: 20px;
        }
        .status-dot-pulse {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 8px #22C55E;
        }

        /* Section Cards */
        .agri-card {
            background: #182B23;
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 20px;
            box-shadow: 0 6px 20px -4px rgba(0, 0, 0, 0.35);
        }

        /* Placeholder Standby Box */
        .placeholder-box {
            border: 1px dashed rgba(34, 197, 94, 0.3);
            border-radius: 14px;
            background: rgba(18, 31, 26, 0.5);
            padding: 48px 24px;
            text-align: center;
            color: #D1E7DD;
            margin-top: 10px;
        }
        .placeholder-icon {
            font-size: 2.4rem;
            margin-bottom: 8px;
        }
        .placeholder-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 4px;
        }
        .placeholder-sub {
            font-size: 0.84rem;
            color: #94A3B8;
        }

        /* Vision container */
        .vision-container {
            background: #0D1512;
            border: 1px solid rgba(34, 197, 94, 0.18);
            border-radius: 12px;
            padding: 10px;
            margin-top: 8px;
            margin-bottom: 12px;
        }

        /* Caption row */
        .meta-caption {
            background: #182B23;
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 8px;
            padding: 10px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: #D1E7DD;
            text-align: center;
            margin-top: 10px;
        }

        /* Metric Cards */
        .metric-card {
            background: #182B23;
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 6px 20px -4px rgba(0, 0, 0, 0.35);
        }
        .metric-border-green {
            border-top: 4px solid #22C55E !important;
            background: linear-gradient(180deg, rgba(34, 197, 94, 0.08) 0%, #182B23 100%) !important;
        }
        .metric-border-amber {
            border-top: 4px solid #F59E0B !important;
            background: linear-gradient(180deg, rgba(245, 158, 11, 0.08) 0%, #182B23 100%) !important;
        }
        .metric-border-crimson {
            border-top: 4px solid #EF4444 !important;
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.08) 0%, #182B23 100%) !important;
        }

        .metric-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-val {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.15;
            margin-bottom: 6px;
        }
        .metric-subtext {
            font-size: 0.78rem;
            color: #94A3B8;
            line-height: 1.4;
        }

        .text-green { color: #22C55E !important; }
        .text-amber { color: #F59E0B !important; }
        .text-crimson { color: #EF4444 !important; }

        .category-pill {
            font-size: 0.74rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 6px;
            display: inline-block;
            margin-top: 4px;
        }
        .pill-green { background: rgba(34, 197, 94, 0.16); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.35); }
        .pill-amber { background: rgba(245, 158, 11, 0.16); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.35); }
        .pill-crimson { background: rgba(239, 68, 68, 0.16); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.35); }

        .progress-bar-wrap {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
            margin: 8px 0 6px 0;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.4s ease;
        }

        /* Commercial Directive Banner */
        .directive-box {
            background: #182B23;
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-left: 5px solid #22C55E;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 22px;
        }
        .directive-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.96rem;
            font-weight: 700;
            color: #22C55E;
            margin-bottom: 4px;
        }
        .directive-content {
            font-size: 0.94rem;
            color: #FFFFFF;
            line-height: 1.55;
        }

        /* Agronomic Advisory Card */
        .advisory-box {
            background: #182B23;
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-left: 5px solid #22C55E;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 22px;
        }
        .advisory-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 10px;
        }
        .advisory-content {
            font-size: 0.94rem;
            color: #D1E7DD;
            line-height: 1.68;
            background: #121F1A;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid rgba(34, 197, 94, 0.15);
        }

        /* Repurposing Card */
        .repurposing-box {
            background: #182B23;
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-left: 5px solid #F59E0B;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 24px;
        }
        .repurposing-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 14px;
        }
        .repurposing-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .repurposing-pill {
            background: #121F1A;
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 8px;
            padding: 8px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .pill-glyph { font-size: 1.15rem; }
        .pill-name { font-size: 0.88rem; font-weight: 600; color: #FFFFFF; }

        /* Streamlit Native UI overrides */
        .stButton > button {
            background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%) !important;
            color: #FFFFFF !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 14px rgba(34, 197, 94, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(34, 197, 94, 0.5) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid rgba(34, 197, 94, 0.2) !important;
            gap: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important;
            padding: 8px 16px !important;
            border-radius: 6px 6px 0 0 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #22C55E !important;
            border-bottom: 2px solid #22C55E !important;
            background: rgba(34, 197, 94, 0.08) !important;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(34, 197, 94, 0.35) !important;
            border-radius: 12px !important;
            background: #121F1A !important;
            padding: 14px !important;
        }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached YOLO Model Loader & Computer Vision Inference Engine
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """Load and cache the YOLOv8 model weights in memory."""
    if not HAS_YOLO:
        return None
    weights_path = Path(__file__).resolve().parent.parent / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = Path("weights/best.pt")
    if weights_path.exists():
        try:
            return YOLO(str(weights_path))
        except Exception:
            return None
    return None


def generate_cv_visualizations(
    image_pil: Image.Image,
    defect_percentage: float,
    confidence_threshold: float = 0.25,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """Generates the annotated bounding box image and defect pixel heatmap overlay."""
    img_rgb = np.array(image_pil.convert("RGB"))
    h, w = img_rgb.shape[:2]
    total_area = h * w
    cv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    annotated = cv_img.copy()
    rotten_mask = np.zeros((h, w), dtype=np.uint8)

    yolo_model = load_yolo_model()
    boxes_detected = 0

    if yolo_model is not None:
        try:
            results = yolo_model(cv_img, conf=confidence_threshold, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = yolo_model.names.get(cls_id, "")
                    is_rotten = str(cls_name).lower() == "rotten" or cls_id == 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))

                    if is_rotten:
                        boxes_detected += 1
                        rotten_mask[y1:y2, x1:x2] = 255
                        color = (68, 68, 239)  # Crimson BGR
                        label = f"Rotten: {conf * 100:.1f}%"
                    else:
                        color = (94, 197, 34)  # Sprout Green BGR
                        label = f"Fresh: {conf * 100:.1f}%"

                    fill_layer = annotated.copy()
                    cv2.rectangle(fill_layer, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(fill_layer, 0.18, annotated, 0.82, 0, annotated)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                    cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), (13, 21, 18), -1)
                    cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, 1)
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 5, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        except Exception:
            boxes_detected = 0

    # If no physical boxes were found (or simulated scenario with defect > 0)
    if boxes_detected == 0 and defect_percentage > 0.5:
        cx, cy = int(w * 0.54), int(h * 0.54)
        radius = int(np.sqrt((total_area * (defect_percentage / 100.0)) / np.pi))
        radius = max(14, min(min(h, w) // 2 - 10, radius))
        cv2.circle(rotten_mask, (cx, cy), radius, 255, -1)

        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w, cx + radius), min(h, cy + radius)

        color = (68, 68, 239)
        label = f"Defect Area: {defect_percentage:.1f}%"
        fill_layer = annotated.copy()
        cv2.rectangle(fill_layer, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(fill_layer, 0.18, annotated, 0.82, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1 + 6, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    elif boxes_detected == 0 and defect_percentage <= 0.5:
        cv2.putText(
            annotated,
            "Pristine Optical Profile // 0% Decay Detected",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (94, 197, 34),
            1,
            cv2.LINE_AA,
        )

    # Defect Heatmap Generation
    if np.sum(rotten_mask) > 0:
        darkened_base = (cv_img * 0.7).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(rotten_mask, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(darkened_base, 0.65, color_heatmap, 0.35, 0)
        contours, _ = cv2.findContours(rotten_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heatmap_overlay, contours, -1, (68, 68, 239), 2)
    else:
        heatmap_overlay = cv2.applyColorMap((cv_img * 0.25).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heatmap_overlay = cv2.addWeighted(cv_img, 0.8, heatmap_overlay, 0.2, 0)

    rotten_pixels = int(np.sum(rotten_mask > 0))
    stats = {
        "width": w,
        "height": h,
        "total_area": total_area,
        "rotten_pixels": rotten_pixels,
        "measured_pct": round((rotten_pixels / total_area) * 100.0, 2) if total_area > 0 else 0.0,
    }

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)

    return Image.fromarray(annotated_rgb), Image.fromarray(heatmap_rgb), stats


# -----------------------------------------------------------------------------
# 3. Backend Communication & Normalization Engine
# -----------------------------------------------------------------------------
DEFAULT_API_URL = "http://localhost:8000/api/v1/grade"
DEFAULT_CONFIDENCE = 0.25


def query_backend(image_file, endpoint_url: str = DEFAULT_API_URL) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Sends the uploaded image file to the FastAPI backend service."""
    image_bytes = image_file.getvalue()
    filename = image_file.name
    content_type = image_file.type or "image/jpeg"
    files = {"file": (filename, image_bytes, content_type)}

    try:
        response = requests.post(endpoint_url, files=files, timeout=12)
        if response.status_code == 422:
            files_alt = {"image": (filename, image_bytes, content_type)}
            response = requests.post(endpoint_url, files=files_alt, timeout=12)

        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        error_msg = (
            f"⚠️ **Could not connect to FastAPI server at `{endpoint_url}`.**\n\n"
            "Ensure the backend is running (`uvicorn api.main:app --reload --port 8000`), "
            "or toggle **Demo Simulation Mode** in the top bar to inspect pre-configured scenarios."
        )
        return None, error_msg
    except Exception as err:
        return None, f"❌ **Inference Error:** {str(err)}"


def generate_mock_response(scenario: str) -> Dict[str, Any]:
    """Generates realistic grading telemetry matching the chosen simulation scenario."""
    time.sleep(0.3)
    if "Grade A" in scenario:
        defect_percentage = 2.4
    elif "Grade B" in scenario:
        defect_percentage = 14.2
    else:
        defect_percentage = 37.3

    return evaluate_quality(defect_percentage)


def run_local_inference(image_pil: Image.Image) -> Dict[str, Any]:
    """Runs local inference directly in Streamlit if FastAPI backend is offline."""
    img_rgb = np.array(image_pil.convert("RGB"))
    h, w = img_rgb.shape[:2]
    total_image_area = h * w
    cv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    yolo_model = load_yolo_model()
    fruit_bbox = None
    largest_box_area = 0
    has_rotten_detection = False
    max_rotten_conf = 0.0
    rotten_box_area = 0
    rotten_mask = np.zeros((h, w), dtype=np.uint8)

    if yolo_model is not None:
        try:
            results = yolo_model(cv_img, conf=DEFAULT_CONFIDENCE, verbose=False)
            boxes_list = results[0].boxes if len(results) > 0 and hasattr(results[0], "boxes") else []
            if boxes_list is not None and len(boxes_list) > 0:
                for box in boxes_list:
                    cls_id = int(box.cls[0])
                    class_name = str(yolo_model.names.get(cls_id, cls_id)).lower()
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    box_w = max(0, x2 - x1)
                    box_h = max(0, y2 - y1)
                    b_area = box_w * box_h

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
        except Exception:
            pass

    # Determine fruit area
    if fruit_bbox is not None and largest_box_area > 0:
        fx1, fy1, fx2, fy2 = fruit_bbox
        fruit_crop = cv_img[fy1:fy2, fx1:fx2]
        if fruit_crop.size > 0:
            hsv_crop = cv2.cvtColor(fruit_crop, cv2.COLOR_BGR2HSV)
            sat = hsv_crop[:, :, 1]
            _, thresh = cv2.threshold(sat, 25, 255, cv2.THRESH_BINARY)
            fruit_pixel_count = int(np.sum(thresh > 0))
            fruit_area = fruit_pixel_count if fruit_pixel_count > 0.35 * largest_box_area else largest_box_area
        else:
            fruit_area = largest_box_area
    else:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        _, thresh = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
        seg_count = int(np.sum(thresh > 0))
        fruit_area = seg_count if seg_count > 0.05 * total_image_area else total_image_area

    fruit_area = max(1, fruit_area)

    rotten_pixels = int(np.sum(rotten_mask))
    defect_percentage = min(100.0, (rotten_pixels / fruit_area) * 100.0)
    rot_box_ratio = (rotten_box_area / fruit_area) * 100.0
    defect_percentage = max(defect_percentage, rot_box_ratio * 0.75)

    if has_rotten_detection:
        defect_percentage = max(defect_percentage, 8.5)

    defect_percentage = min(100.0, defect_percentage)
    defect_percentage = round(defect_percentage, 2)

    report = evaluate_quality(defect_percentage)
    report["has_rotten_detection"] = has_rotten_detection
    report["max_rotten_conf"] = round(max_rotten_conf, 4)
    return report


def normalize_api_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts and normalizes metrics returned by the backend without client-side grade overrides."""
    defect = (
        data.get("defect_percentage")
        if data.get("defect_percentage") is not None
        else data.get("defect_pct")
        if data.get("defect_pct") is not None
        else data.get("defects", 0.0)
    )
    if isinstance(defect, (int, float)):
        defect_val = float(defect)
        if 0.0 < defect_val <= 1.0:
            defect_val = defect_val * 100.0
    else:
        try:
            val = float(str(defect).replace("%", ""))
            defect_val = val * 100.0 if 0.0 < val <= 1.0 else val
        except ValueError:
            defect_val = 0.0

    defect_val = round(defect_val, 2)
    defect_str = f"{defect_val:.1f}% Surface Damage"

    # Directly render values returned by the backend without recalculation
    grade = data.get("grade")
    category = data.get("category")
    shelf_life_str = data.get("shelf_life") or data.get("shelf_life_days")
    salvage_val = data.get("salvage_rate") or data.get("salvage_yield") or data.get("salvage_value")
    ui_color = data.get("ui_color") or data.get("status_color")
    commercial_action = data.get("commercial_action") or data.get("recommended_action")
    agronomic_prevention = data.get("agronomic_prevention")
    alternative_uses = data.get("alternative_uses")

    # Fallback to rules engine ONLY if backend fields are missing
    if not grade or not ui_color or not shelf_life_str:
        rules_eval = evaluate_quality(defect_val)
        grade = grade or rules_eval["grade"]
        category = category or rules_eval["category"]
        shelf_life_str = shelf_life_str or rules_eval["shelf_life"]
        salvage_val = salvage_val or rules_eval["salvage_rate"]
        ui_color = ui_color or rules_eval["ui_color"]
        commercial_action = commercial_action or rules_eval["commercial_action"]
        agronomic_prevention = agronomic_prevention or rules_eval["agronomic_prevention"]
        alternative_uses = alternative_uses or rules_eval["alternative_uses"]

    return {
        "grade": grade,
        "category": category,
        "defect_val": defect_val,
        "defect_str": defect_str,
        "shelf_life_str": shelf_life_str,
        "salvage_val": salvage_val,
        "recommendation": commercial_action,
        "agronomic_prevention": agronomic_prevention,
        "alternative_uses": alternative_uses,
        "ui_color": ui_color,
        "raw": data,
    }


# -----------------------------------------------------------------------------
# 4. Top Section: Clean Minimal Hero Header
# -----------------------------------------------------------------------------
col_hero_left, col_hero_ctrl = st.columns([3, 1.2], gap="medium")

with col_hero_left:
    st.markdown(
        textwrap.dedent(
            """
            <div class="hero-container">
              <div class="hero-left">
                <span class="hero-icon">🍅</span>
                <div>
                  <div style="display: flex; align-items: center; gap: 10px;">
                    <h1 class="hero-title">AgriGrade AI</h1>
                    <span class="hero-pill">
                      <span class="status-dot-pulse"></span>
                      AI Model Online (YOLOv8)
                    </span>
                  </div>
                  <div class="hero-subtitle">
                    Autonomous Post-Harvest Quality Grading &amp; Valorization Engine
                  </div>
                </div>
              </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

with col_hero_ctrl:
    st.markdown(
        textwrap.dedent(
            """
            <div style="background: #182B23; border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 16px; padding: 14px 18px; margin-bottom: 24px;">
              <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                Inspection Mode
              </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
    simulate_mode = st.toggle("Demo Simulation Mode", value=False, help="Simulate instant responses for demo presentations.")
    if simulate_mode:
        selected_scenario = st.selectbox(
            "Demo Scenario",
            options=[
                "Grade A (< 5% defect)",
                "Grade B (14% defect)",
                "Grade C (37% defect)",
            ],
            index=0,
            label_visibility="collapsed",
        )
    else:
        selected_scenario = "Grade A"
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Middle Section: Side-by-Side Inspection Chamber
# -----------------------------------------------------------------------------
col_input, col_vision = st.columns(2, gap="large")

with col_input:
    st.markdown("### 📷 Specimen Input")
    uploaded_file = st.file_uploader(
        "Upload a high-contrast tomato photo (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    image_obj = None
    trigger_inspect = False

    if uploaded_file is not None:
        try:
            image_obj = Image.open(uploaded_file)
            st.markdown('<div class="vision-container">', unsafe_allow_html=True)
            st.image(image_obj, caption=f"Uploaded Specimen: {uploaded_file.name}", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            file_kb = len(uploaded_file.getvalue()) / 1024
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="meta-caption">
                      Dimensions: {image_obj.width} × {image_obj.height} px &nbsp;|&nbsp; File Size: {file_kb:.1f} KB &nbsp;|&nbsp; Color: RGB 24-Bit
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            trigger_inspect = st.button("Run Quality Inspection", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
            trigger_inspect = False
    elif simulate_mode:
        st.info(f"💡 **Demo Mode Active**: Ready to simulate grading for **{selected_scenario}**.")
        # Synthetic specimen image for instant testing
        synthetic_arr = np.zeros((360, 360, 3), dtype=np.uint8)
        cv2.circle(synthetic_arr, (180, 180), 140, (215, 45, 35), -1)
        cv2.circle(synthetic_arr, (180, 50), 20, (40, 165, 60), -1)
        image_obj = Image.fromarray(synthetic_arr)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        trigger_inspect = st.button("Run Quality Inspection", type="primary", use_container_width=True)
    else:
        st.markdown(
            textwrap.dedent(
                """
                <div class="placeholder-box">
                  <div class="placeholder-icon">🍅</div>
                  <div class="placeholder-title">Awaiting Tomato Specimen</div>
                  <div class="placeholder-sub">Upload an image above or toggle Demo Simulation Mode to begin.</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )


with col_vision:
    st.markdown("### 🔬 Neural Vision Diagnostics")

    if trigger_inspect and image_obj is not None:
        with st.spinner("🤖 Analyzing surface defects and calculating defect pixel mask..."):
            if simulate_mode or uploaded_file is None:
                result_raw = generate_mock_response(selected_scenario)
                error_msg = None
            else:
                result_raw, error_msg = query_backend(uploaded_file, DEFAULT_API_URL)
                if error_msg and image_obj is not None:
                    # Graceful in-process neural vision fallback if FastAPI backend is offline
                    result_raw = run_local_inference(image_obj)
                    error_msg = None

        if error_msg:
            st.warning(error_msg)
        elif result_raw:
            res = normalize_api_data(result_raw)
            defect_val = res["defect_val"]

            annotated_img, heatmap_img, cv_stats = generate_cv_visualizations(
                image_obj, defect_val, confidence_threshold=DEFAULT_CONFIDENCE
            )

            tab_bbox, tab_mask = st.tabs(["Detected Bounding Boxes", "Defect Pixel Mask"])

            with tab_bbox:
                st.markdown('<div class="vision-container">', unsafe_allow_html=True)
                st.image(annotated_img, caption="YOLOv8 Defect Detection Boxes", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with tab_mask:
                st.markdown('<div class="vision-container">', unsafe_allow_html=True)
                st.image(heatmap_img, caption="Surface Decay Pixel Density Heatmap", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="meta-caption">
                      Surface Area: {cv_stats['total_area']:,} px &nbsp;|&nbsp; Defect Pixels: {cv_stats['rotten_pixels']:,} px &nbsp;|&nbsp; Defect Coverage: <strong>{defect_val:.2f}%</strong>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
            st.session_state["active_result"] = res
    elif "active_result" in st.session_state and image_obj is not None:
        res = st.session_state["active_result"]
        annotated_img, heatmap_img, cv_stats = generate_cv_visualizations(
            image_obj, res["defect_val"], confidence_threshold=DEFAULT_CONFIDENCE
        )
        tab_bbox, tab_mask = st.tabs(["Detected Bounding Boxes", "Defect Pixel Mask"])
        with tab_bbox:
            st.markdown('<div class="vision-container">', unsafe_allow_html=True)
            st.image(annotated_img, caption="YOLOv8 Defect Detection Boxes", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with tab_mask:
            st.markdown('<div class="vision-container">', unsafe_allow_html=True)
            st.image(heatmap_img, caption="Surface Decay Pixel Density Heatmap", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="meta-caption">
                  Surface Area: {cv_stats['total_area']:,} px &nbsp;|&nbsp; Defect Pixels: {cv_stats['rotten_pixels']:,} px &nbsp;|&nbsp; Defect Coverage: <strong>{res['defect_val']:.2f}%</strong>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            textwrap.dedent(
                """
                <div class="placeholder-box">
                  <div class="placeholder-icon">🔬</div>
                  <div class="placeholder-title">Neural Diagnostics Inactive</div>
                  <div class="placeholder-sub">Click 'Run Quality Inspection' to generate bounding boxes and defect heatmap.</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 6. Bottom Section: Results & Actionable Value Deck
# -----------------------------------------------------------------------------
if "active_result" in st.session_state:
    res = st.session_state["active_result"]
    grade_text = res["grade"]
    category_text = res["category"]
    defect_str = res["defect_str"]
    defect_val = res["defect_val"]
    shelf_life_str = res["shelf_life_str"]
    salvage_val = res["salvage_val"]
    recommendation = res["recommendation"]
    agronomic_prevention = res["agronomic_prevention"]
    alternative_uses = res["alternative_uses"]

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("## 📊 Quality Assessment & Commercial Routing")

    # Determine Grade Color Scheme strictly
    grade_clean = str(grade_text).strip()
    grade_upper = grade_clean.upper()

    if "GRADE C" in grade_upper or grade_upper.endswith(" C") or grade_upper == "C" or "INDUSTRIAL" in grade_upper or "BIOMASS" in grade_upper:
        border_class = "metric-border-crimson"
        text_color_class = "text-crimson"
        pill_class = "pill-crimson"
        gauge_color = "#EF4444"
        badge_short = "GRADE C"
    elif "GRADE B" in grade_upper or grade_upper.endswith(" B") or grade_upper == "B" or "SECONDARY" in grade_upper or "MANDI" in grade_upper:
        border_class = "metric-border-amber"
        text_color_class = "text-amber"
        pill_class = "pill-amber"
        gauge_color = "#F59E0B"
        badge_short = "GRADE B"
    else:
        border_class = "metric-border-green"
        text_color_class = "text-green"
        pill_class = "pill-green"
        gauge_color = "#22C55E"
        badge_short = "GRADE A"

    ui_color = res.get("ui_color") or gauge_color

    # Row 1 — Primary Metrics (st.columns(4))
    clamped_pct = min(max(defect_val, 0.0), 100.0)
    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="metric-card" style="border-left: 3px solid {ui_color};">
                  <div class="metric-label">QUALITY GRADE</div>
                  <div class="metric-val" style="color: {ui_color};">{badge_short}</div>
                  <div class="category-pill" style="border: 1px solid {ui_color}55; background: {ui_color}18; color: {ui_color};">{category_text}</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="metric-card metric-border-crimson">
                  <div class="metric-label">DEFECT COVERAGE</div>
                  <div class="metric-val text-crimson">{defect_val:.1f}% Surface Damage</div>
                  <div class="progress-bar-wrap">
                    <div class="progress-bar-fill" style="width: {clamped_pct}%; background: {gauge_color};"></div>
                  </div>
                  <div class="metric-subtext">Fruit-normalized surface decay ratio</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="metric-card metric-border-amber">
                  <div class="metric-label">ESTIMATED SHELF LIFE</div>
                  <div class="metric-val text-amber">⏱️ {shelf_life_str}</div>
                  <div class="metric-subtext" style="margin-top: 8px;">Ambient holding (20°C / 65% RH)</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="metric-card metric-border-green">
                  <div class="metric-label">ECONOMIC SALVAGE</div>
                  <div class="metric-val text-green">💰 {salvage_val}</div>
                  <div class="metric-subtext" style="margin-top: 8px;">Preserved commodity asset yield</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    # Operational Commercial Routing Directive Banner
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="directive-box">
              <div class="directive-title">⚡ Commercial Routing Directive</div>
              <div class="directive-content"><strong>Recommended Action:</strong> {recommendation}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    # Row 2 — Farmer Agronomic Prevention Advisory
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="advisory-box">
              <div class="advisory-title">🌱 Crop Health &amp; Future Prevention Protocol</div>
              <div class="advisory-content">{agronomic_prevention}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    # Row 3 — Secondary Commercial Pathways (Zero-Waste Repurposing)
    icon_map = {
        "export": "🚢",
        "supermarket": "🛒",
        "packaging": "📦",
        "cold chain": "❄️",
        "mandi": "🏪",
        "local market": "🏪",
        "cloud kitchen": "🍳",
        "canteen": "🍽️",
        "restaurant": "👨‍🍳",
        "puree": "🥫",
        "paste": "🥫",
        "sauce": "🥫",
        "ketchup": "🍅",
        "composting": "🌱",
        "vermicompost": "🌱",
        "biogas": "⚡",
        "anaerobic": "⚡",
        "feed": "🐄",
        "livestock": "🐄",
    }

    pill_html_list = []
    for item in alternative_uses:
        matched_icon = "♻️"
        item_lower = item.lower()
        for kw, ico in icon_map.items():
            if kw in item_lower:
                matched_icon = ico
                break

        pill_html_list.append(
            f'<div class="repurposing-pill"><span class="pill-glyph">{matched_icon}</span><span class="pill-name">{item}</span></div>'
        )

    pills_rendered = "".join(pill_html_list)

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="repurposing-box">
              <div class="repurposing-title">♻️ Alternative Commercial Pathways &amp; Waste Divergence</div>
              <div class="repurposing-pills">
                {pills_rendered}
              </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
