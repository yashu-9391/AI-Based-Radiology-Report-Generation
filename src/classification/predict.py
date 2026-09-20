import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Tuple
from src.classification.model import MultiLabelRadiologyClassifier

from src.preprocessing.image_preprocessing import preprocess_single_image
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_classifier(model_path: str = "models/best_model.pth", config: Dict[str, Any] = None) -> Tuple[MultiLabelRadiologyClassifier, torch.device]:
    """
    Loads pre-trained multi-label radiology classifier weights.
    Falls back to initializing architecture if checkpoint file does not exist yet.
    """
    config = config or {}
    device_setting = config.get("project", {}).get("device", "auto")
    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    arch = config.get("model", {}).get("architecture", "densenet121")
    num_classes = config.get("model", {}).get("num_classes", 14)

    model = MultiLabelRadiologyClassifier(
        architecture=arch,
        num_classes=num_classes,
        pretrained=False
    )

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded classifier weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint from {model_path}: {e}. Using initialized model weights.")
    else:
        logger.info(f"Checkpoint {model_path} not found. Running inference with baseline model.")

    model.to(device)
    model.eval()
    return model, device

def predict_image(
    image_input: Union[str, Image.Image, np.ndarray],
    config: Dict[str, Any],
    model_path: str = "models/best_model.pth"
) -> Dict[str, Any]:
    """
    Inference pipeline for a single chest X-ray image.
    Outputs structured predictions containing probabilities, percentage strings, and threshold status for all 14 observations.
    """
    labels = config.get("observations", [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
    ])
    threshold = config.get("classification", {}).get("threshold", 0.5)
    img_size = config.get("data", {}).get("image_size", 224)

    model, device = load_classifier(model_path=model_path, config=config)
    image_tensor = preprocess_single_image(image_input=image_input, image_size=img_size).to(device)

    with torch.no_grad():
        probabilities = model.predict_probabilities(image_tensor).squeeze(0).cpu().numpy()

    predictions = []
    positive_findings = []

    for idx, prob in enumerate(probabilities):
        finding_name = labels[idx] if idx < len(labels) else f"Finding_{idx}"
        prob_val = float(prob)
        percentage_str = f"{int(round(prob_val * 100))}%"
        is_positive = prob_val >= threshold

        status_str = "Positive" if is_positive else "Negative"

        pred_item = {
            "finding": finding_name,
            "probability": round(prob_val, 4),
            "percentage": percentage_str,
            "prediction": status_str,
            "is_positive": is_positive
        }
        predictions.append(pred_item)

        if is_positive and finding_name != "No Finding":
            positive_findings.append({
                "finding": finding_name,
                "probability": prob_val,
                "percentage": percentage_str
            })

    # Sort predictions by probability descending
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    positive_findings.sort(key=lambda x: x["probability"], reverse=True)

    image_identifier = image_input if isinstance(image_input, str) else "uploaded_image"

    return {
        "image": image_identifier,
        "threshold": threshold,
        "predictions": predictions,
        "positive_findings": positive_findings
    }
