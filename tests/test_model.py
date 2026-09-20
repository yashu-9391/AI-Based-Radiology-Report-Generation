import pytest
import torch
from src.classification.model import MultiLabelRadiologyClassifier
from src.classification.losses import get_loss_function

def test_model_forward_pass():
    model = MultiLabelRadiologyClassifier(architecture="densenet121", num_classes=14, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 14)

    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (2, 14)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_loss_functions():
    config = {"training": {"loss_function": "focal", "focal_gamma": 2.0}}
    loss_fn = get_loss_function(config)
    dummy_logits = torch.randn(4, 14)
    dummy_targets = torch.randint(0, 2, (4, 14)).float()
    loss = loss_fn(dummy_logits, dummy_targets)
    assert loss.item() >= 0.0
