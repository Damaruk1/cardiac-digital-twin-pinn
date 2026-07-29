"""
transformer.py
---------------
Transformer-based classifier for individual ECG beats, using
multi-head self-attention instead of convolution.

Architecture:
    Input (batch, window_length, in_channels)
      -> Linear projection to d_model
      -> + Positional encoding
      -> [Multi-Head Self-Attention -> Add&Norm -> FeedForward -> Add&Norm] x N
      -> Global average pool over time
      -> Dense (num_classes)

Why this shape:
    - Linear projection lifts each raw timestep (1-2 voltage values)
      into a higher-dimensional embedding space (d_model), giving
      attention more room to represent different kinds of patterns.
    - Positional encoding is REQUIRED -- self-attention has no built-in
      sense of sequence order, unlike a CNN's sliding kernel or an RNN's
      recurrence.
    - Multiple encoder layers let the model build increasingly abstract
      representations, each layer attending over the previous layer's output.
    - Global average pooling collapses the per-timestep outputs into one
      fixed-size vector per beat, ready for classification.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds fixed (non-learned) sinusoidal position information to each
    timestep's embedding, using the standard formulation from
    "Attention Is All You Need" (Vaswani et al.).

    Using sine/cosine functions of different frequencies means the
    model can learn to attend to RELATIVE positions (e.g. "80 samples
    earlier") via linear combinations of these encodings, not just
    absolute ones.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions: cosine

        # Registered as a buffer (not a learned parameter) so it moves
        # with the model to GPU/CPU but is never updated by the optimizer.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x with positional information added, same shape.
        """
        return x + self.pe[:, : x.size(1), :]


class ECGTransformer(nn.Module):
    """Transformer encoder for single-beat ECG classification."""

    def __init__(
        self,
        in_channels: int = 1,
        window_length: int = 180,
        num_classes: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_channels: Number of ECG leads (raw input feature dim).
            window_length: Number of timesteps per beat window.
            num_classes: Number of AAMI output classes.
            d_model: Embedding dimension used throughout the Transformer.
                     Must be divisible by n_heads.
            n_heads: Number of parallel attention heads.
            n_layers: Number of stacked Transformer encoder layers.
            dim_feedforward: Hidden size of the per-position feedforward
                              network inside each encoder layer.
            dropout: Dropout probability used throughout.
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.input_projection = nn.Linear(in_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=window_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # we use (batch, seq, feature) ordering throughout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, window_length) --
               same convention as ECG1DCNN, so both models can share the
               same BeatDataset/DataLoader without modification.

        Returns:
            Raw logits of shape (batch_size, num_classes).
        """
        x = x.transpose(1, 2)  # (batch, channels, length) -> (batch, length, channels)
        x = self.input_projection(x)  # -> (batch, length, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # -> (batch, length, d_model)
        x = x.mean(dim=1)  # global average pool over time -> (batch, d_model)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
