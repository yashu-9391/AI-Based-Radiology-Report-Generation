import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration file and resolves default device & paths.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure required directories exist
    paths = config.get("paths", {})
    for path_key, dir_path in paths.items():
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    return config
