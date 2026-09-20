import pytest
import numpy as np
from PIL import Image
from src.pipeline import RadiologyAIPipeline


def test_end_to_end_pipeline():
    pipeline = RadiologyAIPipeline("configs/config.yaml")
    # Synthetic valid X-ray grayscale texture
    np_img = np.random.randint(40, 200, (224, 224), dtype=np.uint8)
    dummy_img = Image.fromarray(np_img).convert("RGB")
    results = pipeline.analyze_xray(dummy_img, top_k=2)


    assert "predictions" in results
    assert "positive_findings" in results
    assert "retrieved_reports" in results
    assert "report" in results
    assert "PRELIMINARY RADIOLOGY REPORT" in results["report"]
    assert "AI DISCLAIMER" in results["report"]
