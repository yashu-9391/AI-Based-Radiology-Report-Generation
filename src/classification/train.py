import os
import json
import time
import pandas as pd
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional

from src.classification.model import MultiLabelRadiologyClassifier
from src.classification.losses import get_loss_function
from src.preprocessing.image_preprocessing import get_transforms
from src.data.dataset_utils import get_dataloaders
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__)

def train_model(config: Dict[str, Any], dataset_type: str = "chexpert") -> Dict[str, Any]:
    """
    Complete PyTorch training pipeline for multi-label radiology classifier.
    Saves best_model.pth, latest_model.pth, and experiment metrics.
    """
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    device_setting = config.get("project", {}).get("device", "auto")
    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    # Transforms & DataLoaders
    img_size = config.get("data", {}).get("image_size", 224)
    train_transform = get_transforms(image_size=img_size, is_train=True)
    val_transform = get_transforms(image_size=img_size, is_train=False)

    train_loader, val_loader, test_loader = get_dataloaders(
        config=config,
        train_transform=train_transform,
        val_transform=val_transform,
        dataset_type=dataset_type
    )

    if train_loader is None or len(train_loader) == 0:
        logger.error("Dataset not found or empty. Please populate the dataset folder before training.")
        return {"status": "error", "message": "Dataset not found. Please place dataset files in datasets/ directory."}

    # Model instantiation
    arch = config.get("model", {}).get("architecture", "densenet121")
    num_classes = config.get("model", {}).get("num_classes", 14)
    pretrained = config.get("model", {}).get("pretrained", True)
    dropout = config.get("model", {}).get("dropout", 0.2)

    model = MultiLabelRadiologyClassifier(
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    ).to(device)

    # Loss & Optimizer
    criterion = get_loss_function(config)
    lr = config.get("training", {}).get("learning_rate", 1e-4)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_patience = config.get("training", {}).get("scheduler_patience", 2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=scheduler_patience)

    epochs = config.get("training", {}).get("epochs", 10)
    early_stopping_patience = config.get("training", {}).get("early_stopping_patience", 5)

    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    models_dir = config.get("paths", {}).get("models_dir", "models")
    results_dir = config.get("paths", {}).get("results_dir", "results")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    logger.info(f"Starting training for {epochs} epochs on {arch}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)

                logits = model(images)
                loss = criterion(logits, targets)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "lr": current_lr
        })

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0

            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config
            }

            best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
            best_model_path = os.path.join(models_dir, "best_model.pth")
            torch.save(checkpoint_payload, best_ckpt_path)
            torch.save(checkpoint_payload, best_model_path)
            logger.info(f" Saved new best model checkpoint to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Save Latest Checkpoint
        latest_ckpt_path = os.path.join(checkpoints_dir, "latest_model.pth")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, latest_ckpt_path)

    total_training_time = time.time() - start_time
    logger.info(f"Training finished in {total_training_time:.2f} seconds.")

    # Save training history CSV & metadata
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)

    metadata = {
        "architecture": arch,
        "epochs_completed": len(history),
        "best_val_loss": best_val_loss,
        "training_time_seconds": total_training_time,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(models_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(results_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return {"status": "success", "metadata": metadata}
