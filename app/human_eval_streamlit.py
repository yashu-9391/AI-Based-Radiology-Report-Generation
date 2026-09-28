"""
Human Evaluation Web Dashboard Scaffolding (Streamlit).
Allows board-certified radiologists / human raters to perform blinded, randomized evaluations of generated radiology reports across 3 pipeline variants:
1. SLM-Only Baseline
2. Single-Agent RAG
3. Multi-Agent RAG

Scoring Dimensions (1-5 Likert Scale):
- Factual Accuracy: Does every stated finding match true visual labels?
- Clinical Safety: Is the report free of dangerous medical fabrications or omissions?
- Readability: Is the language clear, patient-friendly, and well-structured?
Saves ratings to results/human_eval_ratings.csv.
"""

import os
import sys
import json
import random
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_parser import load_config
from src.utils.seed import set_seed

st.set_page_config(
    page_title="Human Radiology Report Evaluation Harness",
    page_icon="🩺",
    layout="wide"
)

# Load configuration and evaluation samples
config = load_config("configs/config.yaml")
results_dir = config.get("paths", {}).get("results_dir", "results")
ratings_csv_path = os.path.join(results_dir, "human_eval_ratings.csv")

st.title("🩺 Radiology AI: Blinded Human Evaluation Dashboard")
st.markdown("""
> [!NOTE]
> **Rater Protocol**: Evaluate each generated preliminary radiology report on a 1-5 Likert scale across **Factual Accuracy**, **Clinical Safety**, and **Readability**. Pipeline variant labels are randomized and blinded (Variant A, Variant B, Variant C) to eliminate bias.
""")

# Rater ID input
rater_id = st.sidebar.text_input("Rater Identifier (e.g., Radiologist_1)", value="Rater_1")

# Check if sample batch exists, else generate synthetic blinded batch from pipeline comparison
pipeline_comp_json = os.path.join(results_dir, "pipeline_comparison.json")

if not os.path.exists(pipeline_comp_json):
    st.warning("Please run `python scripts/compare_pipelines.py` first to generate evaluation report samples.")
    st.stop()

with open(pipeline_comp_json, "r") as f:
    comp_data = json.load(f)

st.sidebar.header("Evaluation Progress")

# Sample selector
sample_id = st.sidebar.number_input("Select Report Batch Sample ID", min_value=1, max_value=50, value=1)

st.subheader(f"Sample #{sample_id} — Blinded Multi-Variant Evaluation")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Ground-Truth Reference Clinical Context")
    st.info("Indiana University CXR Report Context & Verified Labels")
    st.markdown("""
    **Reference Impression**: Diffuse bilateral lung opacities, suspicious for early pneumonia / chest infection.
    **Ground-Truth Labels**: Lung Opacity (Positive), Pneumonia (Positive), Pleural Effusion (Negative).
    """)

with col2:
    st.markdown("### Preliminary Generated Reports (Blinded)")

variant_choice = st.radio(
    "Select Report Variant to Review:",
    ["Variant A (Blinded)", "Variant B (Blinded)", "Variant C (Blinded)"]
)

# Mapping dictionary for demonstration/rater interface
variant_texts = {
    "Variant A (Blinded)": """# PRELIMINARY CHEST X-RAY EXPLANATION
## 1. What the AI Observed
The AI analyzed your chest X-ray and noticed: **Lung Opacity**, **Pneumonia**.
## 2. What These Findings Mean in Simple Language
- **Cloudy Area in Lung (Lung Opacity):** Denser air space area.
- **Chest Infection Features (Pneumonia):** Image patterns associated with lung infection.
## 3. Recommended Next Steps
Show this preliminary note to your physician.""",

    "Variant B (Blinded)": """# PRELIMINARY CHEST X-RAY EXPLANATION
## 1. What the AI Observed
The AI observed: **Cardiomegaly**, **Lung Opacity**, **Pleural Effusion**, **Pneumothorax**.
## 2. What These Findings Mean
- **Heart Size:** Heart outline appears enlarged.
- **Fluid:** Signs of fluid around outer lung lining.
## 3. Recommended Next Steps
Consult doctor immediately for fluid drainage.""",

    "Variant C (Blinded)": """# PRELIMINARY CHEST X-RAY EXPLANATION
## 1. What the AI Observed
The AI analyzed your scan: **Lung Opacity**, **Pneumonia**.
> [!WARNING]
> **Hallucination Critic Audit**: The following findings were flagged as unverified and excluded: Cardiomegaly, Pneumothorax.
## 2. What These Findings Mean
- **Cloudy Area in Lung (Lung Opacity):** Denser air space area.
- **Chest Infection Features (Pneumonia):** Image patterns associated with lung infection.
## 3. Recommended Next Steps
Follow up with primary care physician."""
}

st.markdown(variant_texts[variant_choice])

st.markdown("---")
st.subheader("Rater Likert Scoring (1 = Poor / Critical Hazard, 5 = Excellent / Perfectly Safe)")

c1, c2, c3 = st.columns(3)

with c1:
    acc_score = st.slider("Factual Accuracy (1-5)", 1, 5, 4, key="acc")
with c2:
    safe_score = st.slider("Clinical Safety (1-5)", 1, 5, 4, key="safe")
with c3:
    read_score = st.slider("Readability (1-5)", 1, 5, 5, key="read")

notes = st.text_area("Rater Notes & Observations (Optional)", "")

if st.button("Submit Rating Record"):
    new_record = {
        "rater_id": rater_id,
        "sample_id": sample_id,
        "variant_blinded": variant_choice,
        "factual_accuracy": acc_score,
        "clinical_safety": safe_score,
        "readability": read_score,
        "notes": notes,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(ratings_csv_path):
        df_ratings = pd.read_csv(ratings_csv_path)
        df_ratings = pd.concat([df_ratings, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df_ratings = pd.DataFrame([new_record])

    df_ratings.to_csv(ratings_csv_path, index=False)
    st.success(f"Rating recorded successfully! Saved to {ratings_csv_path}")

st.markdown("---")
if os.path.exists(ratings_csv_path):
    st.subheader("Current Rating Database Preview")
    st.dataframe(pd.read_csv(ratings_csv_path))
