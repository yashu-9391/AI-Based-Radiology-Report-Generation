import os
import pandas as pd
import torch
from PIL import Image
from typing import Dict, Any, Optional, List
from src.data.base_dataset import BaseDatasetLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class IUXRayDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for Indiana University (IU) Chest X-Ray images and reports.
    Supports reading reports for FAISS index retrieval and multi-label classification.
    """
    def __init__(
        self,
        root_dir: str,
        csv_filename: str = "indiana_reports.csv",
        config: Optional[Dict[str, Any]] = None,
        transform=None,
        is_train: bool = True
    ):
        config = config or {}
        super().__init__(root_dir=root_dir, config=config, transform=transform, is_train=is_train)
        self.csv_path = os.path.join(self.root_dir, csv_filename)
        self.reports: List[Dict[str, Any]] = []

        if self.validate_dataset():
            self.load_dataset()

    def validate_dataset(self) -> bool:
        if not os.path.exists(self.root_dir):
            logger.warning(f"IU X-Ray root directory does not exist at: {self.root_dir}")
            return False
        if not os.path.exists(self.csv_path):
            for item in os.listdir(self.root_dir):
                sub = os.path.join(self.root_dir, item)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, os.path.basename(self.csv_path))):
                    self.root_dir = sub
                    self.csv_path = os.path.join(self.root_dir, os.path.basename(self.csv_path))
                    break
        if not os.path.exists(self.csv_path):
            logger.warning(f"IU X-Ray report CSV does not exist at: {self.csv_path}")
            return False
        return True

    def load_dataset(self) -> None:
        logger.info(f"Loading IU X-Ray reports dataset from {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error(f"Failed to read IU X-Ray CSV file: {e}")
            return

        for idx, row in df.iterrows():
            uid = str(row.get("uid", idx))
            findings = str(row.get("findings", "")).strip()
            impression = str(row.get("impression", "")).strip()

            # Clean NaNs or "nan" strings
            if findings.lower() == "nan":
                findings = ""
            if impression.lower() == "nan":
                impression = ""

            full_report = f"FINDINGS: {findings} IMPRESSION: {impression}".strip()
            if full_report == "FINDINGS:  IMPRESSION:":
                continue

            report_item = {
                "report_id": uid,
                "findings": findings,
                "impression": impression,
                "full_report": full_report
            }
            self.reports.append(report_item)

            # Look up images associated with report
            image_dirs = [
                os.path.join(self.root_dir, "images", "images_normalized"),
                os.path.join(self.root_dir, "images"),
                os.path.join(self.root_dir, "..", "images")
            ]
            img_path = ""
            for d in image_dirs:
                candidate = os.path.join(d, f"{uid}.png")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                img_path = os.path.join(self.root_dir, "images", f"{uid}.png")

            # Dummy zero targets for 14 classes if explicit labels aren't in CSV
            targets = [0.0] * len(self.labels_list)

            self.samples.append({
                "image_path": img_path,
                "target": targets,
                "report_id": uid,
                "report": full_report
            })

        logger.info(f"Loaded {len(self.reports)} reports and {len(self.samples)} sample mappings from IU X-Ray.")

    def get_reports(self) -> List[Dict[str, Any]]:
        """
        Returns all extracted radiology reports for vector store indexing.
        """
        return self.reports

    def __getitem__(self, idx: int):
        if not self.samples:
            raise IndexError("IU X-Ray Dataset is empty or dataset files are not present.")

        sample_info = self.samples[idx]
        img_path = sample_info["image_path"]

        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error opening image {img_path}: {e}")
                image = Image.new("RGB", (224, 224), color=0)
        else:
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(sample_info["target"], dtype=torch.float32)

        return {
            "image": image,
            "target": target,
            "image_path": img_path,
            "report": sample_info.get("report", "")
        }
