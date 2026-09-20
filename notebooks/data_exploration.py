import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

def explore_dataset(chexpert_csv: str = "datasets/chexpert/train.csv") -> Dict[str, Any]:
    """
    Explores dataset statistics: total count, label distribution, missing values, and uncertain labels.
    """
    if not os.path.exists(chexpert_csv):
        return {
            "status": "Dataset not found",
            "message": f"Dataset CSV at {chexpert_csv} does not exist. Please place dataset files first."
        }

    df = pd.read_csv(chexpert_csv)
    total_samples = len(df)

    observations = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
    ]

    label_stats = {}
    for col in observations:
        if col in df.columns:
            pos_count = int((df[col] == 1.0).sum())
            neg_count = int((df[col] == 0.0).sum())
            unc_count = int((df[col] == -1.0).sum())
            null_count = int(df[col].isna().sum())
            label_stats[col] = {
                "positive": pos_count,
                "negative": neg_count,
                "uncertain": unc_count,
                "missing": null_count,
                "positive_percentage": f"{(pos_count / total_samples) * 100:.2f}%"
            }

    exploration_summary = {
        "dataset_path": chexpert_csv,
        "total_samples": total_samples,
        "label_statistics": label_stats
    }

    os.makedirs("results", exist_ok=True)
    with open("results/dataset_exploration.json", "w") as f:
        json.dump(exploration_summary, f, indent=4)

    print(f"Dataset exploration completed. Processed {total_samples} samples.")
    return exploration_summary

if __name__ == "__main__":
    explore_dataset()
