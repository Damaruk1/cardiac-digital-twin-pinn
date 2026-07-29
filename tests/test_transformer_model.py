"""
test_transformer_model.py
---------------------------
Phase 7 tests: verify ECGTransformer produces correctly shaped outputs,
that positional encoding actually changes the representation, and that
invalid configurations are rejected early.

Run with:
    pytest tests/test_transformer_model.py -v
"""

import torch

from src.models.transformer import ECGTransformer, PositionalEncoding


def test_forward_pass_output_shape_single_channel():
    model = ECGTransformer(in_channels=1, window_length=180, num_classes=5)
    dummy_input = torch.randn(8, 1, 180)

    output = model(dummy_input)

    assert output.shape == (8, 5)


def test_forward_pass_output_shape_multi_channel():
    model = ECGTransformer(in_channels=2, window_length=180, num_classes=5)
    dummy_input = torch.randn(16, 2, 180)

    output = model(dummy_input)

    assert output.shape == (16, 5)


def test_invalid_d_model_head_combination_raises():
    """d_model must be divisible by n_heads -- e.g. 65 isn't divisible by 4."""
    try:
        ECGTransformer(in_channels=1, window_length=180, d_model=65, n_heads=4)
        assert False, "Expected ValueError for incompatible d_model/n_heads"
    except ValueError:
        pass


def test_model_has_trainable_parameters():
    model = ECGTransformer(in_channels=1, window_length=180, num_classes=5)
    param_count = model.count_parameters()

    assert param_count > 1000
    assert param_count < 10_000_000


def test_positional_encoding_changes_representation():
    """Two identical embeddings at different positions should become
    different after positional encoding is added -- proving position
    information is actually injected."""
    pe = PositionalEncoding(d_model=16, max_len=50)
    identical_embeddings = torch.ones(1, 10, 16)  # same values at every timestep

    output = pe(identical_embeddings)

    # Position 0 and position 5 started identical; after PE they must differ.
    assert not torch.allclose(output[0, 0, :], output[0, 5, :])


def test_output_is_raw_logits_not_probabilities():
    model = ECGTransformer(in_channels=1, window_length=180, num_classes=5)
    dummy_input = torch.randn(4, 1, 180)

    output = model(dummy_input)
    row_sums = output.sum(dim=1)

    assert not torch.allclose(row_sums, torch.ones(4), atol=1e-3)
