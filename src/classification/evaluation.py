import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, confusion_matrix
from typing import Dict, Any, List

from src.classification.model import MultiLabelRadiologyClassifier
from src.preprocessing.image_preprocessing import get_transforms
from src.data.dataset_utils import get_dataloaders
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def evaluate_classifier(config: Dict[str, Any], model_path: str = "models/best_model.pth", dataset_type: str = "chexpert") -> Dict[str, Any]:
    """
    Evaluates multi-label radiology classifier performance across all 14 observations.
    Calculates AUROC, AUPRC, Precision, Recall, Sensitivity, Specificity, and F1-score.
    Saves plots to results/plots/ and metrics to results/metrics.json.
    """
    results_dir = config.get("paths", {}).get("results_dir", "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(model_path):
        logger.warning(f"Model checkpoint at {model_path} not found. Evaluation cannot complete.")
        unevaluated_res = {
            "status": "Not evaluated yet",
            "message": f"Model checkpoint '{model_path}' does not exist. Train model first.",
            "metrics": {}
        }
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(unevaluated_res, f, indent=4)
        return unevaluated_res

    # Device selection
    device_setting = config.get("project", {}).get("device", "auto")
    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data & Model
    img_size = config.get("data", {}).get("image_size", 224)
    val_transform = get_transforms(image_size=img_size, is_train=False)
    _, _, test_loader = get_dataloaders(config=config, val_transform=val_transform, dataset_type=dataset_type)

    if test_loader is None or len(test_loader) == 0:
        logger.warning("Test dataset not available for evaluation.")
        unevaluated_res = {
            "status": "Not evaluated yet",
            "message": f"Test split for dataset '{dataset_type}' not found or empty.",
            "metrics": {}
        }
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(unevaluated_res, f, indent=4)
        return unevaluated_res

    num_classes = config.get("model", {}).get("num_classes", 14)
    labels = config.get("observations", [f"Class_{i}" for i in range(num_classes)])

    arch = config.get("model", {}).get("architecture", "densenet121")
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiLabelRadiologyClassifier(architecture=arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["target"].cpu().numpy()
            probs = model.predict_probabilities(images).cpu().numpy()

            all_targets.append(targets)
            all_probs.append(probs)

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    threshold = config.get("classification", {}).get("threshold", 0.5)
    all_preds = (all_probs >= threshold).astype(int)

    per_class_metrics = []
    auroc_list = []
    auprc_list = []

    for i, label_name in enumerate(labels):
        y_true = all_targets[:, i]
        y_prob = all_probs[:, i]
        y_pred = all_preds[:, i]

        # AUROC & AUPRC calculation with safety fallback for single-class targets
        if len(np.unique(y_true)) > 1:
            try:
                auroc = roc_auc_score(y_true, y_prob)
            except Exception:
                auroc = 0.5
            try:
                auprc = average_precision_score(y_true, y_prob)
            except Exception:
                auprc = 0.0
        else:
            auroc = 0.5
            auprc = 0.0

        auroc_list.append(auroc)
        auprc_list.append(auprc)

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Calculate Sensitivity & Specificity
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class_metrics.append({
            "finding": label_name,
            "auroc": round(float(auroc), 4),
            "auprc": round(float(auprc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(f1), 4)
        })

    macro_auroc = float(np.mean(auroc_list))
    macro_auprc = float(np.mean(auprc_list))

    eval_summary = {
        "status": "Evaluated successfully",
        "dataset": dataset_type,
        "num_test_samples": len(all_targets),
        "macro_auroc": round(macro_auroc, 4),
        "macro_auprc": round(macro_auprc, 4),
        "per_class_metrics": per_class_metrics
    }

    # Save metrics JSON
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(eval_summary, f, indent=4)

    # Plot & save AUROC Bar Chart
    plt.figure(figsize=(10, 6))
    df_metrics = pd.DataFrame(per_class_metrics)
    plt.barh(df_metrics["finding"], df_metrics["auroc"], color="steelblue")
    plt.axvline(x=macro_auroc, color="red", linestyle="--", label=f"Macro AUROC: {macro_auroc:.2f}")
    plt.xlabel("AUROC Score")
    plt.title("Per-Finding Multi-Label AUROC Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "auroc_per_class.png"))
    plt.close()

    logger.info(f"Evaluation complete. Macro AUROC: {macro_auroc:.4f}. Saved metrics and plots.")
    return eval_summary
