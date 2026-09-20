import pytest
import numpy as np
from PIL import Image
import os

from src.preprocessing.image_validator import validate_chest_xray
from src.verification.iu_xray_verifier import IUXRayReportVerifier
from src.pipeline import RadiologyAIPipeline

def test_validate_chest_xray_valid():
    # Create synthetic grayscale X-Ray-like image
    np_img = np.random.randint(40, 200, (224, 224), dtype=np.uint8)
    pil_xray = Image.fromarray(np_img).convert("RGB")
    
    is_valid, err_msg, metrics = validate_chest_xray(pil_xray)
    assert is_valid is True
    assert err_msg == ""
    assert metrics["mean_channel_diff"] == 0.0

def test_validate_chest_xray_invalid_color_photo():
    # Create a synthetic vibrant color image (like a photo or non-X-ray)
    np_img = np.zeros((224, 224, 3), dtype=np.uint8)
    np_img[:, :, 0] = 255  # Red channel
    np_img[:, :, 1] = 50   # Green channel
    np_img[:, :, 2] = 20   # Blue channel
    pil_color = Image.fromarray(np_img, mode="RGB")

    is_valid, err_msg, metrics = validate_chest_xray(pil_color)
    assert is_valid is False
    assert "Invalid image selected" in err_msg
    assert "select only Chest X-Ray images" in err_msg

def test_iu_xray_verifier():
    verifier = IUXRayReportVerifier()
    
    predictions_data = {
        "positive_findings": [{"finding": "Cardiomegaly", "probability": 0.85, "percentage": "85%"}]
    }
    retrieved_reports = [
        {
            "report_id": "REF_101",
            "findings": "The cardiac silhouette is moderately enlarged. Lungs are clear.",
            "impression": "Cardiomegaly.",
            "full_report": "FINDINGS: The cardiac silhouette is moderately enlarged. IMPRESSION: Cardiomegaly."
        }
    ]
    generated_report = "PRELIMINARY RADIOLOGY REPORT\nFINDINGS: Radiographic evidence of Cardiomegaly.\nIMPRESSION: Cardiomegaly."

    res = verifier.verify_report(generated_report, predictions_data, retrieved_reports)
    
    assert res["verified"] is True
    assert res["similarity_score"] > 0.5
    assert "Cardiomegaly" in res["matched_findings"]
    assert "VERIFIED" in res["status"]

def test_pipeline_validation_rejection():
    pipeline = RadiologyAIPipeline()
    
    # Non-X-ray color photo
    np_img = np.zeros((224, 224, 3), dtype=np.uint8)
    np_img[:, :, 0] = 200
    np_img[:, :, 1] = 100
    pil_color = Image.fromarray(np_img, mode="RGB")

    with pytest.raises(ValueError) as excinfo:
        pipeline.analyze_xray(pil_color)
    
    assert "Please select only Chest X-Ray images" in str(excinfo.value)
