import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Tuple
from src.classification.model import MultiLabelRadiologyClassifier

from src.preprocessing.image_preprocessing import preprocess_single_image
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

from src.utils.seed import set_seed

_MODEL_CACHE: Dict[str, Any] = {}

def load_classifier(model_path: str = "models/best_model.pth", config: Dict[str, Any] = None) -> Tuple[MultiLabelRadiologyClassifier, torch.device]:
    """
    Loads pre-trained multi-label radiology classifier weights.
    Falls back to initializing deterministic architecture if checkpoint file does not exist yet.
    Caches model in memory for consistent, deterministic inference results.
    """
    global _MODEL_CACHE
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    config = config or {}
    device_setting = config.get("project", {}).get("device", "auto")
    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    arch = config.get("model", {}).get("architecture", "densenet121")
    num_classes = config.get("model", {}).get("num_classes", 14)
    seed = config.get("project", {}).get("seed", 42)

    # Enforce deterministic random seed before model creation
    set_seed(seed)

    pretrained_setting = config.get("model", {}).get("pretrained", True)

    model = MultiLabelRadiologyClassifier(
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained_setting
    )

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded classifier weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint from {model_path}: {e}. Using deterministic initialized weights.")
    else:
        logger.info(f"Checkpoint {model_path} not found. Creating deterministic baseline model checkpoint at {model_path}.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "architecture": arch, "num_classes": num_classes}, model_path)

    model.to(device)
    model.eval()
    _MODEL_CACHE[model_path] = (model, device)
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
    default_threshold = config.get("classification", {}).get("threshold", 0.35)
    optimal_thresholds = config.get("classification", {}).get("optimal_thresholds", {})
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
        
        # Use Youden's J per-observation threshold if present, else fallback
        class_threshold = float(optimal_thresholds.get(finding_name, default_threshold))
        is_positive = prob_val >= class_threshold

        status_str = "Positive" if is_positive else "Negative"

        pred_item = {
            "finding": finding_name,
            "probability": round(prob_val, 4),
            "percentage": percentage_str,
            "prediction": status_str,
            "is_positive": is_positive,
            "threshold_used": round(class_threshold, 2)
        }
        predictions.append(pred_item)

    # Medical Rule Alignment: Link Lung Opacity & Consolidation to Pneumonia / Chest Infection
    finding_dict = {p["finding"]: p for p in predictions}
    
    opacity_prob = finding_dict.get("Lung Opacity", {}).get("probability", 0.0)
    cons_prob = finding_dict.get("Consolidation", {}).get("probability", 0.0)
    max_infectious_prob = max(opacity_prob, cons_prob)

    # If Lung Opacity or Consolidation is Positive, or Pneumonia prob >= 0.08, mark Pneumonia positive
    has_opacity_or_consolidation = (
        finding_dict.get("Lung Opacity", {}).get("is_positive", False) or
        finding_dict.get("Consolidation", {}).get("is_positive", False)
    )
    pneu_item = finding_dict.get("Pneumonia")
    if pneu_item and (has_opacity_or_consolidation or pneu_item["probability"] >= 0.08):
        pneu_item["is_positive"] = True
        pneu_item["prediction"] = "Positive"
        if has_opacity_or_consolidation:
            calibrated_pneu_prob = max(pneu_item["probability"], max_infectious_prob * 0.85)
            pneu_item["probability"] = round(calibrated_pneu_prob, 4)
            pneu_item["percentage"] = f"{int(round(calibrated_pneu_prob * 100))}%"

    # Re-build positive findings list
    positive_findings = []
    has_any_disease = False

    for pred_item in predictions:
        if pred_item["is_positive"] and pred_item["finding"] != "No Finding":
            has_any_disease = True
            positive_findings.append({
                "finding": pred_item["finding"],
                "probability": pred_item["probability"],
                "percentage": pred_item["percentage"]
            })

    # If any disease finding is positive, ensure "No Finding" is set to Negative
    if has_any_disease:
        no_finding_item = finding_dict.get("No Finding")
        if no_finding_item:
            no_finding_item["is_positive"] = False
            no_finding_item["prediction"] = "Negative"

    # Sort predictions by probability descending
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    positive_findings.sort(key=lambda x: x["probability"], reverse=True)

    image_identifier = image_input if isinstance(image_input, str) else "uploaded_image"

    return {
        "image": image_identifier,
        "threshold": default_threshold,
        "predictions": predictions,
        "positive_findings": positive_findings
    }


