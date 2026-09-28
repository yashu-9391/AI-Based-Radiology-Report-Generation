
import os
import sys
import streamlit as st
from PIL import Image
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import RadiologyAIPipeline
from src.preprocessing.image_validator import validate_chest_xray
from src.utils.config_parser import load_config

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Preliminary Radiology Report Generator",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Medical UI styling & Popups
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .card { background-color: #F8FAFC; padding: 1.2rem; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem; }
    .disclaimer-box { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 1rem; color: #991B1B; font-weight: 500; margin-top: 1.5rem; border-radius: 4px; }
    .positive-badge { background-color: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .negative-badge { background-color: #F3F4F6; color: #4B5563; padding: 3px 8px; border-radius: 12px; font-weight: 500; font-size: 0.85rem; }
    .verified-badge { background-color: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.9rem; border: 1px solid #10B981; }
    .popup-modal {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.2rem 0;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.2);
    }
    .popup-title { color: #991B1B; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.5rem; }
    .popup-msg { color: #B91C1C; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; }
    .popup-subtext { color: #4B5563; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown("<div class='main-header'>AI-Based Preliminary Radiology Report Generation</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A Multi-Label Deep Learning and Single-Agent RAG Framework for Chest X-Ray Analysis</div>", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return RadiologyAIPipeline(config_path="configs/config.yaml")

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Error initializing pipeline: {e}")
    st.stop()

# Default parameters
model_path = "models/best_model.pth"
top_k = 3

# Main UI Layout: Two Columns
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader("Select a Chest X-Ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    is_xray_valid = False
    validation_error = ""

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Step 0: X-Ray Image Validation Check
            is_xray_valid, validation_error, metrics = validate_chest_xray(image)

            if not is_xray_valid:
                # Clear previous session results immediately
                st.session_state.pop("results", None)
                # Display explicit Modal Popup & Warning Toast
                st.markdown(f"""
                    <div class='popup-modal'>
                        <div class='popup-title'>⚠️ Invalid Image Selection</div>
                        <div class='popup-msg'>Please select only X-Ray images.</div>
                        <div class='popup-subtext'>The uploaded image is NOT a Chest X-Ray (Handwriting / paper / color content detected: P95 Channel Diff={metrics.get('p95_channel_diff', 0)}, P95 Saturation={metrics.get('p95_saturation', 0)}). Please upload a valid Chest X-Ray.</div>
                    </div>
                """, unsafe_allow_html=True)
                st.toast("🚨 Please select only X-Ray images!", icon="⚠️")
                st.error("❌ Non-X-Ray image detected. Please select only Chest X-Ray images.")
            else:
                st.image(image, caption="Uploaded Chest X-Ray Input (Validated ✅)", use_container_width=True)

        except Exception as e:
            st.error(f"Error reading image: {e}")
            is_xray_valid = False

        generate_btn = st.button(
            "🚀 Generate Preliminary Report",
            type="primary",
            use_container_width=True,
            disabled=not is_xray_valid
        )

        if generate_btn:
            if not is_xray_valid:
                st.warning("Please select only X-Ray images.")
            else:
                with st.spinner("Processing X-ray, predicting findings, retrieving IU-XRay reports, and generating preliminary report..."):
                    try:
                        results = pipeline.analyze_xray(image_input=image, model_path=model_path, top_k=top_k)
                        st.session_state["results"] = results
                    except ValueError as ve:
                        st.error(f"Validation Error: {ve}")

with col_right:
    st.subheader("2. Multi-Label Findings & Analysis")

    if "results" in st.session_state:
        results = st.session_state["results"]
        predictions = results["predictions"]
        retrieved_reports = results["retrieved_reports"]
        report_text = results["report"]
        verification = results.get("verification", {})

        # Multi-Label Predictions Display
        st.markdown("#### 📊 Predicted Findings & Probabilities")
        
        for item in predictions:
            finding = item["finding"]
            prob = item["probability"]
            percentage = item["percentage"]
            status = item["prediction"]

            c1, c2, c3 = st.columns([2, 2, 1])
            c1.write(f"**{finding}**")
            c2.progress(prob, text=f"{percentage}")
            if finding == "No Finding":
                if status == "Positive":
                    c3.markdown("<span class='verified-badge'>Normal (Clear)</span>", unsafe_allow_html=True)
                else:
                    c3.markdown("<span class='negative-badge'>Abnormalities Present</span>", unsafe_allow_html=True)
            else:
                if status == "Positive":
                    c3.markdown("<span class='positive-badge'>Positive</span>", unsafe_allow_html=True)
                else:
                    c3.markdown("<span class='negative-badge'>Negative</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Ground Truth Verification Section with IU-XRay Dataset
        st.markdown("#### 🔍 Verification with IU-XRay Reports Dataset")
        if verification and verification.get("verified"):
            sim_score = verification.get("similarity_percentage", "0%")
            status_text = verification.get("status", "Verified")
            matched_findings = verification.get("matched_findings", [])
            summary_text = verification.get("verification_summary", "")

            st.markdown(f"<span class='verified-badge'>✅ {status_text} ({sim_score})</span>", unsafe_allow_html=True)
            st.info(summary_text)

            if matched_findings:
                st.write("**Verified Finding Alignment:**")
                badges_html = " ".join([f"<span class='positive-badge'>Verified Match: {f}</span>" for f in matched_findings])
                st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.warning("Verification against IU-XRay dataset pending or unverified.")

        st.markdown("---")
        
        # Retrieved Reports Section
        st.markdown("#### 📚 Retrieved Reference Reports (IU X-Ray Corpus)")
        if retrieved_reports:
            for idx, rep in enumerate(retrieved_reports, 1):
                score = rep.get("similarity_score", 0.0)
                score_percent = f"{int(round(score * 100))}%" if score <= 1.0 else f"{score:.2f}"
                with st.expander(f"Reference Report #{idx} — Similarity: {score_percent}"):
                    st.write(f"**Report ID**: {rep.get('report_id', 'N/A')}")
                    st.write(f"**Findings**: {rep.get('findings', 'N/A')}")
                    st.write(f"**Impression**: {rep.get('impression', 'N/A')}")
        else:
            st.info("No matching reference reports retrieved.")

        st.markdown("---")

        # Generated Preliminary Report Section
        st.markdown("#### 📝 Generated Preliminary Radiology Report")
        st.text_area("Structured Report Output", report_text, height=280)


        # Download Button
        st.download_button(
            label="💾 Download Preliminary Report (.txt)",
            data=report_text,
            file_name="preliminary_radiology_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        # Disclaimer
        st.markdown(
            "<div class='disclaimer-box'>"
            "⚠️ <b>AI DISCLAIMER</b>: This is an AI-generated preliminary report intended strictly for research "
            "and educational purposes. It does NOT constitute a final medical diagnosis and must be reviewed by a qualified radiologist."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Please upload a chest X-ray image and click 'Generate Preliminary Report' to view predictions.")
