import pytest
import os
from src.utils.config_parser import load_config
from src.data.dataset_utils import validate_dataset_exists, get_dataloaders
from src.data.chexpert_dataset import CheXpertDatasetLoader

def test_config_loading():
    config = load_config("configs/config.yaml")
    assert "data" in config
    assert "model" in config
    assert "observations" in config
    assert len(config["observations"]) == 14

def test_dataset_validation():
    # Should safely return False for unpopulated path
    is_valid = validate_dataset_exists("NonExistent", "datasets/non_existent_folder")
    assert is_valid is False
