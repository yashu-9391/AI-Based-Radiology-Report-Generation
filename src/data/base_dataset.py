import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset

class BaseDatasetLoader(Dataset, ABC):
    """
    Abstract base class for all chest X-ray datasets.
    Supports multi-label disease classification and validation.
    """
    def __init__(self, root_dir: str, config: Dict[str, Any], transform=None, is_train: bool = True):
        self.root_dir = root_dir
        self.config = config
        self.transform = transform
        self.is_train = is_train
        self.labels_list: List[str] = config.get("observations", [])
        self.samples: List[Dict[str, Any]] = []

    @abstractmethod
    def validate_dataset(self) -> bool:
        """
        Validates that the required files and directories exist.
        """
        pass

    @abstractmethod
    def load_dataset(self) -> None:
        """
        Loads metadata and samples.
        """
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Calculates positive instance counts per class label.
        """
        counts = {label: 0 for label in self.labels_list}
        for sample in self.samples:
            target = sample.get("target", [])
            for idx, val in enumerate(target):
                if idx < len(self.labels_list) and val == 1.0:
                    counts[self.labels_list[idx]] += 1
        return counts
