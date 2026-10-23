import io
import time
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriGrade AI - Post-Harvest Quality & Freshness Grader",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, modern agricultural AI dashboard
st.markdown(
    """
    <style>
    /* Global font & background accents */
    .main {
        background-color: #fcfdfc;
    }
    
    /* Top title styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b6b58;
        margin-bottom: 1.5rem;
    }
    
    /* Card containers */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .metric-subtitle {
        font-size: 0.82rem;
        color: #495057;
    }
    
    /* Grade color badges */
    .grade-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .grade-a {
        background-color: #d8f3dc;
        color: #1b4332;
        border: 2px solid #52b788;
    }
    
    .grade-b {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeeba;
    }
    
    .grade-c {
        background-color: #ffe3e3;
        color: #c92a2a;
        border: 2px solid #ffa8a8;
    }
    
    /* Recommendation banner */
    .action-box {
        background: #f8f9fa;
        border-left: 5px solid #2d6a4f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    
    .action-box strong {
        color: #1b4332;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Configuration & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?w=400&auto=format&fit=crop&q=60",
        caption="AgriGrade AI Vision System",
        use_container_width=True,
    )
    st.title("🍅 System Settings")
    st.markdown(
        """
        **AgriGrade AI** uses computer vision and automated deep learning models
        to accurately grade tomato freshness, detect surface defects, and provide
        instant shelf-life & commercial disposition guidance.
        """
    )
    
    st.divider()
    
    st.subheader("🔌 Backend Connection")
    backend_url = st.text_input(
        "FastAPI Endpoint",
        value="http://localhost:8000/api/v1/grade",
        help="Target endpoint hosted by Aayush's backend service",
    )
    
    # Offline Demo Mode fallback toggle for convenience during hackathon demos
    simulate_mode = st.toggle(
        "Demo Simulation Mode",
        value=False,
        help="Simulate backend response if the local server is temporarily offline.",
    )

    st.divider()
    st.caption("AgriGrade AI • Post-Harvest Quality & Freshness Grader")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def query_backend(image_file, endpoint_url: str):
    """
    Sends the uploaded image file to the FastAPI backend service.
    Handles network exceptions, timeouts, and JSON parsing gracefully.
    """
    image_bytes = image_file.getvalue()
    filename = image_file.name
    content_type = image_file.type or "image/jpeg"

    # Multi-field fallback attempt: standard FastAPI expects UploadFile named 'file' or 'image'
    files = {"file": (filename, image_bytes, content_type)}

    try:
        response = requests.post(endpoint_url, files=files, timeout=12)
        
        # If the endpoint expected field name 'image' instead of 'file' (HTTP 422 Unprocessable Entity)
        if response.status_code == 422:
            files_alt = {"image": (filename, image_bytes, content_type)}
            response = requests.post(endpoint_url, files=files_alt, timeout=12)

        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        error_msg = (
            f"⚠️ **Could not connect to Aayush's backend server at `{endpoint_url}`.**\n\n"
            "Please ensure the FastAPI service is active. For example:\n"
            "```bash\nuvicorn main:app --reload --port 8000\n```\n"
            "*Tip: You can also enable **Demo Simulation Mode** in the sidebar to test UI flows.*"
        )
        return None, error_msg

    except requests.exceptions.Timeout:
        return None, "⏱️ **Backend Request Timed Out.** The model took too long to respond. Please try again."

    except requests.exceptions.HTTPError as http_err:
        return None, f"❌ **Server returned an error:** {http_err} (Status Code: {response.status_code})"

    except Exception as err:
        return None, f"❌ **Unexpected Error:** {str(err)}"


def generate_mock_response(image_file):
    """
    Simulates a plausible response for demo/testing purposes
    when the local FastAPI server is not actively running.
    """
    time.sleep(1.0)  # Simulate network and inference latency
    return {
        "grade": "Grade A",
        "defect_percentage": 3.8,
        "shelf_life_days": "7 - 10 days",
        "salvage_value": "$1.40 / kg (95% of Prime Retail Value)",
        "commercial_action": "Optimal for Export & Premium Supermarkets. No immediate salvage intervention required.",
        "confidence": 0.94,
        "details": {
            "freshness_index": "96.2%",
            "color_ripeness": "Firm Red (Stage 5)",
            "defect_type": "Minor surface blemish (< 4%)",
        },
    }


def normalize_api_data(data: dict):
    """
    Extracts and standardizes the required metrics from diverse potential
    backend response structures.
    """
    # Grade extraction
    grade = (
        data.get("grade")
        or data.get("quality_grade")
        or data.get("predicted_grade")
        or data.get("classification")
        or "Grade B"
    )

    # Defect percentage extraction
    defect = (
        data.get("defect_percentage")
        if data.get("defect_percentage") is not None
        else data.get("defect_pct")
        if data.get("defect_pct") is not None
        else data.get("defects", 0.0)
    )
    if isinstance(defect, (int, float)):
        # Normalize if returned as a decimal ratio (0.0 to 1.0)
        defect_val = defect * 100 if defect <= 1.0 and defect > 0 else float(defect)
        defect_str = f"{defect_val:.1f}%"
    else:
        defect_str = str(defect)
        try:
            defect_val = float(str(defect).replace("%", ""))
        except ValueError:
            defect_val = 0.0

    # Shelf life extraction
    shelf_life = (
        data.get("shelf_life")
        or data.get("shelf_life_days")
        or data.get("estimated_shelf_life")
        or data.get("remaining_shelf_life")
        or "4 - 6 days"
    )
    if isinstance(shelf_life, (int, float)):
        shelf_life_str = f"{shelf_life} days"
    else:
        shelf_life_str = str(shelf_life)

    # Commercial actions / Salvage value extraction
    salvage_val = (
        data.get("salvage_value")
        or data.get("estimated_value")
        or "Standard Market Rate"
    )
    recommendation = (
        data.get("commercial_action")
        or data.get("recommendation")
        or data.get("actions")
        or data.get("salvage_recommendation")
        or "Suitable for wholesale or regional fresh markets."
    )

    return {
        "grade": grade,
        "defect_val": defect_val,
        "defect_str": defect_str,
        "shelf_life_str": shelf_life_str,
        "salvage_val": salvage_val,
        "recommendation": recommendation,
        "raw": data,
    }


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">🍅 AgriGrade AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Post-Harvest Quality & Freshness Grader • Optical Inspection & Shelf-Life Assessment</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Upload a clear top or profile image of the tomato",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            # File metadata chips
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.caption(f"📏 Resolution: {image.width} × {image.height} px")
            with col_meta2:
                file_size_kb = len(uploaded_file.getvalue()) / 1024
                st.caption(f"💾 Size: {file_size_kb:.1f} KB")

            # Grade Quality Action Button
            grade_button = st.button("🔍 Grade Quality", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            grade_button = False
    else:
        st.info("👆 Please upload a tomato image (PNG, JPG, or JPEG) to start the grading process.")
        grade_button = False

with col_right:
    st.subheader("📊 Grading & Shelf Life Assessment")

    if uploaded_file is not None and grade_button:
        with st.spinner("🤖 Analyzing tomato freshness and surface defects with YOLO..."):
            if simulate_mode:
                result_data = generate_mock_response(uploaded_file)
                error_msg = None
            else:
                result_data, error_msg = query_backend(uploaded_file, backend_url)

        if error_msg:
            st.warning(error_msg)
        elif result_data:
            # Parse and normalize result
            res = normalize_api_data(result_data)
            grade_text = res["grade"]
            defect_val = res["defect_val"]
            defect_str = res["defect_str"]
            shelf_life_str = res["shelf_life_str"]
            salvage_val = res["salvage_val"]
            recommendation = res["recommendation"]

            # Choose badge color scheme
            grade_upper = str(grade_text).upper()
            if "A" in grade_upper or "FRESH" in grade_upper:
                badge_class = "grade-a"
                badge_icon = "🟢"
            elif "B" in grade_upper or "FAIR" in grade_upper:
                badge_class = "grade-b"
                badge_icon = "🟡"
            else:
                badge_class = "grade-c"
                badge_icon = "🔴"

            # Top Grade Announcement
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 1.2rem;">
                    <div class="grade-badge {badge_class}">
                        {badge_icon} Assigned Grade: {grade_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metric Cards Row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Quality Grade</div>
                        <div class="metric-value" style="color: #2d6a4f;">{grade_text}</div>
                        <div class="metric-subtitle">Standardized Grade</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Defect Area</div>
                        <div class="metric-value" style="color: #c92a2a;">{defect_str}</div>
                        <div class="metric-subtitle">YOLO Surface Inspection</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Est. Shelf Life</div>
                        <div class="metric-value" style="color: #1d3557;">{shelf_life_str}</div>
                        <div class="metric-subtitle">At Room Temp (20°C)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Defect severity progress bar
            st.write("**Defect Severity Gauge**")
            clamped_pct = min(max(defect_val / 100.0, 0.0), 1.0)
            st.progress(clamped_pct)

            # Salvage Value & Commercial Actions Box
            st.markdown(
                f"""
                <div class="action-box">
                    <strong>💡 Commercial Action & Salvage Recommendation:</strong><br>
                    {recommendation}
                    <hr style="margin: 0.6rem 0; border: none; border-top: 1px solid #d3d3d3;">
                    <strong>💰 Estimated Salvage Value:</strong> {salvage_val}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable Raw Diagnostic Response
            with st.expander("🔍 Detailed Model Output & Diagnostics"):
                st.json(result_data)

    elif uploaded_file is None:
        # Prompting placeholder
        st.markdown(
            """
            <div style="border: 2px dashed #ced4da; border-radius: 12px; padding: 3rem 1.5rem; text-align: center; color: #6c757d; margin-top: 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🍅</div>
                <h4 style="color: #495057;">Awaiting Tomato Image</h4>
                <p>Upload a photo on the left and click <b>Grade Quality</b> to view automated defect analysis, grade classification, and shelf-life predictions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Click the **'Grade Quality'** button on the left to analyze the uploaded tomato.")
