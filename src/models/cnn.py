"""
cnn.py
------
1D Convolutional Neural Network for classifying individual ECG beats
into the 5 AAMI superclasses (N, S, V, F, Q).

Architecture:
    Input (batch, channels, window_length)
      -> [Conv1D -> BatchNorm -> ReLU -> MaxPool] x 3
      -> Flatten
      -> Dense -> Dropout -> ReLU
      -> Dense (num_classes)

Why this shape:
    - Conv1D layers learn local waveform patterns (QRS shape, T-wave
      shape) regardless of their exact position in the window.
    - BatchNorm stabilizes training by normalizing activations between
      layers.
    - MaxPool shrinks the sequence length, building a progressively
      larger "receptive field" so deeper layers see broader context.
    - Dropout on the dense layer fights overfitting -- important given
      how few examples exist for the minority AAMI classes (S, V, F, Q).
"""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """1D CNN for single-beat ECG classification."""

    def __init__(
        self,
        in_channels: int = 1,
        window_length: int = 180,
        num_classes: int = 5,
        conv_channels: tuple = (16, 32, 64),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels: Number of ECG leads used as input channels
                         (1 for single-lead, 2 for MIT-BIH's MLII+V5).
            window_length: Number of samples in each beat window.
            num_classes: Number of AAMI output classes (5: N,S,V,F,Q).
            conv_channels: Output channel counts for each of the 3
                            conv blocks.
            kernel_size: Width of each convolutional filter, in samples.
            dropout: Dropout probability applied before the final layer.
        """
        super().__init__()

        c1, c2, c3 = conv_channels
        padding = kernel_size // 2  # "same" padding, keeps length stable before pooling

        self.conv_block1 = self._make_conv_block(in_channels, c1, kernel_size, padding)
        self.conv_block2 = self._make_conv_block(c1, c2, kernel_size, padding)
        self.conv_block3 = self._make_conv_block(c2, c3, kernel_size, padding)

        # After 3 MaxPool(2) layers, length shrinks by a factor of 2^3 = 8.
        flattened_length = window_length // 8
        flattened_size = c3 * flattened_length

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _make_conv_block(in_ch: int, out_ch: int, kernel_size: int, padding: int) -> nn.Sequential:
        """One Conv1D -> BatchNorm -> ReLU -> MaxPool block."""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, window_length).

        Returns:
            Raw logits of shape (batch_size, num_classes). Softmax is
            NOT applied here -- use nn.CrossEntropyLoss during training,
            which expects raw logits and applies softmax internally.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.flatten(start_dim=1)  # (batch, channels, length) -> (batch, channels*length)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
