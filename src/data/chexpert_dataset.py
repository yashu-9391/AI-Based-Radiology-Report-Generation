import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional
from src.data.base_dataset import BaseDatasetLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class CheXpertDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for CheXpert-v1.0-small multi-label chest X-ray classification.
    """
    def __init__(
        self,
        root_dir: str,
        csv_filename: str = "train.csv",
        config: Optional[Dict[str, Any]] = None,
        transform=None,
        is_train: bool = True
    ):
        config = config or {}
        super().__init__(root_dir=root_dir, config=config, transform=transform, is_train=is_train)
        self.csv_path = os.path.join(self.root_dir, csv_filename)
        self.uncertain_strategy = self.config.get("uncertain_label_strategy", "u-ignore")
        
        if self.validate_dataset():
            self.load_dataset()

    def validate_dataset(self) -> bool:
        if not os.path.exists(self.root_dir):
            logger.warning(f"CheXpert root directory does not exist at: {self.root_dir}")
            return False
        if not os.path.exists(self.csv_path):
            # Check for subdirectories like CheXpert-v1.0-small
            for item in os.listdir(self.root_dir):
                sub = os.path.join(self.root_dir, item)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, os.path.basename(self.csv_path))):
                    self.root_dir = sub
                    self.csv_path = os.path.join(self.root_dir, os.path.basename(self.csv_path))
                    break
        if not os.path.exists(self.csv_path):
            logger.warning(f"CheXpert CSV file does not exist at: {self.csv_path}")
            return False
        return True

    def load_dataset(self) -> None:
        logger.info(f"Loading CheXpert dataset from {self.csv_path} with strategy '{self.uncertain_strategy}'...")
        df = pd.read_csv(self.csv_path)

        # Fill NaNs with 0 (Negative/Unmentioned)
        for col in self.labels_list:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0

        # Apply uncertain label strategy
        for col in self.labels_list:
            if col in df.columns:
                if self.uncertain_strategy == "u-zero" or self.uncertain_strategy == "u-ignore":
                    df[col] = df[col].replace(-1.0, 0.0)
                elif self.uncertain_strategy == "u-one":
                    df[col] = df[col].replace(-1.0, 1.0)

        for idx, row in df.iterrows():
            img_rel_path = str(row["Path"])
            # Handle full path vs relative path
            img_full_path = os.path.join(self.root_dir, "..", img_rel_path) if not os.path.isabs(img_rel_path) else img_rel_path
            if not os.path.exists(img_full_path):
                img_full_path = os.path.join(self.root_dir, os.path.basename(img_rel_path))

            targets = [float(row[col]) for col in self.labels_list]
            self.samples.append({
                "image_path": img_full_path,
                "target": targets,
                "patient_id": row.get("Path", f"sample_{idx}")
            })

        logger.info(f"Loaded {len(self.samples)} samples from CheXpert.")

    def __getitem__(self, idx: int):
        if not self.samples:
            raise IndexError("Dataset is empty or could not be loaded. Please ensure dataset files exist.")

        sample_info = self.samples[idx]
        img_path = sample_info["image_path"]

        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error opening image {img_path}: {e}")
                image = Image.new("RGB", (224, 224), color=0)
        else:
            # Fallback black image if path does not exist during placeholder execution
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(sample_info["target"], dtype=torch.float32)

        return {
            "image": image,
            "target": target,
            "image_path": img_path
        }
