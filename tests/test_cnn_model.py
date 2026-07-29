"""
test_cnn_model.py
-------------------
Phase 6 tests: verify the ECG1DCNN produces correctly shaped outputs
across different configurations, using random dummy tensors (no real
data needed -- this tests the architecture's math, not data loading).

Run with:
    pytest tests/test_cnn_model.py -v
"""

import torch

from src.models.cnn import ECG1DCNN


def test_forward_pass_output_shape_single_channel():
    """Single-lead input should produce (batch, num_classes) logits."""
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    dummy_input = torch.randn(8, 1, 180)  # batch=8, 1 lead, 180 samples

    output = model(dummy_input)

    assert output.shape == (8, 5)


def test_forward_pass_output_shape_multi_channel():
    """Multi-lead input (e.g. MIT-BIH's 2 leads) should work identically."""
    model = ECG1DCNN(in_channels=2, window_length=180, num_classes=5)
    dummy_input = torch.randn(16, 2, 180)  # batch=16, 2 leads, 180 samples

    output = model(dummy_input)

    assert output.shape == (16, 5)


def test_model_has_trainable_parameters():
    """A freshly initialized model should have a nonzero, reasonable
    parameter count (sanity check against a completely empty/broken model)."""
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    param_count = model.count_parameters()

    assert param_count > 1000  # should be a real model, not a trivial one
    assert param_count < 10_000_000  # should NOT be absurdly oversized


def test_different_window_lengths_produce_valid_output():
    """The architecture should adapt correctly to different window sizes,
    as long as they're divisible by 8 (3 MaxPool(2) layers)."""
    for window_length in [128, 180, 256]:
        model = ECG1DCNN(in_channels=1, window_length=window_length, num_classes=5)
        dummy_input = torch.randn(4, 1, window_length)
        output = model(dummy_input)
        assert output.shape == (4, 5), f"Failed for window_length={window_length}"


def test_output_is_raw_logits_not_probabilities():
    """Output should NOT be softmax-normalized -- values can be negative
    or exceed 1, since we use raw logits with CrossEntropyLoss."""
    model = ECG1DCNN(in_channels=1, window_length=180, num_classes=5)
    dummy_input = torch.randn(4, 1, 180)

    output = model(dummy_input)

    # If it were softmax-ed, every row would sum to 1.0 and be non-negative.
    row_sums = output.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones(4), atol=1e-3)
