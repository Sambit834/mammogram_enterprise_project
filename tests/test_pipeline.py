import torch
import pytest
from app.model import MammogramClassifier

def test_resnet50_forward():
    model = MammogramClassifier(model_name='resnet50', pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1)

def test_vit_b_16_forward():
    model = MammogramClassifier(model_name='vit_b_16', pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1)
