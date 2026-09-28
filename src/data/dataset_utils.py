import os
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from src.data.chexpert_dataset import CheXpertDatasetLoader
from src.data.iu_xray_dataset import IUXRayDatasetLoader
from src.preprocessing.image_preprocessing import get_transforms
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TransformedDatasetWrapper(Dataset):
    """
    Wrapper around PyTorch Dataset/Subset to apply image transformation on demand.
    Ensures images are converted from PIL.Image to PyTorch Tensors.
    """
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if self.transform is not None and "image" in item:
            img = item["image"]
            if isinstance(img, Image.Image):
                item["image"] = self.transform(img)
            elif not isinstance(img, torch.Tensor) and self.transform:
                item["image"] = self.transform(img)
        return item

def validate_dataset_exists(dataset_name: str, dataset_path: str) -> bool:
    """
    Checks if a dataset folder exists and contains necessary subfiles.
    """
    if not os.path.exists(dataset_path):
        logger.warning(f"Validation warning: '{dataset_name}' path '{dataset_path}' does not exist.")
        return False
    
    files = os.listdir(dataset_path)
    if not files or files == ["README.md"]:
        logger.warning(f"Validation warning: '{dataset_name}' path '{dataset_path}' is empty or contains only README.md.")
        return False
        
    return True

def get_dataloaders(
    config: Dict[str, Any],
    train_transform=None,
    val_transform=None,
    dataset_type: str = "chexpert"
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits with graceful fallback.
    """
    data_config = config.get("data", {})
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = data_config.get("num_workers", 0)
    image_size = data_config.get("image_size", 224)

    if train_transform is None:
        train_transform = get_transforms(image_size=image_size, is_train=True)
    if val_transform is None:
        val_transform = get_transforms(image_size=image_size, is_train=False)

    if dataset_type == "chexpert":
        path = data_config.get("chexpert_path", "datasets/chexpert")
        if not validate_dataset_exists("CheXpert", path):
            logger.warning("CheXpert dataset files not found. DataLoaders will return None until dataset is provided.")
            return None, None, None
        
        full_dataset = CheXpertDatasetLoader(
            root_dir=path,
            csv_filename="train.csv",
            config=config,
            transform=None,
            is_train=True
        )
    elif dataset_type == "iu_xray":
        path = data_config.get("iu_xray_path", "datasets/iu_xray")
        if not validate_dataset_exists("IU X-Ray", path):
            logger.warning("IU X-Ray dataset files not found. DataLoaders will return None until dataset is provided.")
            return None, None, None

        full_dataset = IUXRayDatasetLoader(
            root_dir=path,
            csv_filename="indiana_reports.csv",
            config=config,
            transform=None,
            is_train=True
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if len(full_dataset) == 0:
        logger.warning(f"Dataset {dataset_type} is empty.")
        return None, None, None

    total_len = len(full_dataset)
    train_split = data_config.get("train_split", 0.8)
    val_split = data_config.get("val_split", 0.1)

    train_size = int(total_len * train_split)
    val_size = int(total_len * val_split)
    test_size = total_len - train_size - val_size

    raw_train, raw_val, raw_test = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_dataset = TransformedDatasetWrapper(raw_train, train_transform)
    val_dataset = TransformedDatasetWrapper(raw_val, val_transform)
    test_dataset = TransformedDatasetWrapper(raw_test, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

