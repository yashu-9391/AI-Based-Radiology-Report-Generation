import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross-Entropy with Logits to mitigate severe medical class imbalance.
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets)

class FocalLoss(nn.Module):
    """
    Multi-label Focal Loss for focusing on hard medical examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

def get_loss_function(config: Dict[str, Any], class_counts: Optional[Dict[str, int]] = None) -> nn.Module:
    """
    Loss function factory supporting standard BCE, Weighted BCE, and Focal Loss.
    """
    loss_type = config.get("training", {}).get("loss_function", "weighted_bce")
    
    if loss_type == "focal":
        gamma = config.get("training", {}).get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma)

    elif loss_type == "weighted_bce":
        if class_counts:
            total_samples = max(sum(class_counts.values()), 1)
            num_classes = len(class_counts)
            weights = []
            for label, count in class_counts.items():
                pos_weight = (total_samples - count) / max(count, 1)
                weights.append(pos_weight)
            pos_weight_tensor = torch.tensor(weights, dtype=torch.float32)
            return WeightedBCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            return nn.BCEWithLogitsLoss()

    return nn.BCEWithLogitsLoss()
