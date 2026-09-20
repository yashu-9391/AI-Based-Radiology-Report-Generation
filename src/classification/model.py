import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Tuple

class MultiLabelRadiologyClassifier(nn.Module):
    """
    Multi-label deep learning model for chest X-ray observation classification.
    Supports DenseNet121, EfficientNet-B0, and ResNet50 architectures.
    Outputs raw logits; apply torch.sigmoid(logits) for independent probabilities per class.
    """
    def __init__(
        self,
        architecture: str = "densenet121",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        elif self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        elif self.architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'. Supported: densenet121, resnet50, efficientnet_b0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (batch_size, num_classes).
        """
        return self.backbone(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning sigmoid probabilities (batch_size, num_classes).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
