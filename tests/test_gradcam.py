"""
test_gradcam.py
-----------------
Phase 10 tests: verify GradCAM1D produces correctly shaped, correctly
ranged heatmaps, and that hooks can be safely attached/removed without
leaking or crashing.

Run with:
    pytest tests/test_gradcam.py -v
"""

import numpy as np
import torch

from src.explainability.gradcam import GradCAM1D
from src.models.cnn import ECG1DCNN


def test_gradcam_output_shape_matches_input_length():
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    model.eval()
    gradcam = GradCAM1D(model, target_layer=model.conv_block3)

    input_tensor = torch.randn(1, 1, 180, requires_grad=True)
    cam, predicted_class = gradcam.generate(input_tensor)

    assert cam.shape == (180,)
    assert 0 <= predicted_class < 5
    gradcam.remove_hooks()


def test_gradcam_output_values_in_valid_range():
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    model.eval()
    gradcam = GradCAM1D(model, target_layer=model.conv_block3)

    input_tensor = torch.randn(1, 1, 180, requires_grad=True)
    cam, _ = gradcam.generate(input_tensor)

    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6  # small float tolerance
    gradcam.remove_hooks()


def test_gradcam_works_with_multi_channel_input():
    model = ECG1DCNN(in_channels=2, window_length=180, num_classes=5)
    model.eval()
    gradcam = GradCAM1D(model, target_layer=model.conv_block3)

    input_tensor = torch.randn(1, 2, 180, requires_grad=True)
    cam, predicted_class = gradcam.generate(input_tensor)

    assert cam.shape == (180,)
    gradcam.remove_hooks()


def test_gradcam_explains_specific_target_class():
    """Requesting an explicit target_class should use that class's
    logit for the backward pass, not necessarily the model's own
    top prediction."""
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    model.eval()
    gradcam = GradCAM1D(model, target_layer=model.conv_block3)

    input_tensor = torch.randn(1, 1, 180, requires_grad=True)
    cam, returned_class = gradcam.generate(input_tensor, target_class=2)

    assert returned_class == 2
    assert cam.shape == (180,)
    gradcam.remove_hooks()


def test_hooks_can_be_removed_without_error():
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    gradcam = GradCAM1D(model, target_layer=model.conv_block3)
    gradcam.remove_hooks()  # should not raise

    # Creating a second GradCAM1D on the same model after removing the
    # first's hooks should work cleanly (no leftover duplicate hooks).
    gradcam2 = GradCAM1D(model, target_layer=model.conv_block3)
    input_tensor = torch.randn(1, 1, 180, requires_grad=True)
    cam, _ = gradcam2.generate(input_tensor)
    assert cam.shape == (180,)
    gradcam2.remove_hooks()
