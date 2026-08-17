"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) adapted for 1D
convolutional models, showing which timesteps of an ECG beat window
most influenced the model's prediction.

Works via forward/backward hooks on a target conv layer -- we capture
that layer's activations during the forward pass, and its gradients
during the backward pass, then combine them per the Grad-CAM formula.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """Computes Grad-CAM heatmaps for a 1D CNN's predictions."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The trained model (should be in eval mode by the caller).
            target_layer: The conv layer to explain -- typically the LAST
                          conv block, since it has the most semantically
                          meaningful (but still spatially/temporally
                          localized) features.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Computes the Grad-CAM heatmap for one input example.

        Args:
            input_tensor: Shape (1, channels, window_length) -- a SINGLE
                          example (batch size 1).
            target_class: Which class's prediction to explain. If None,
                          uses the model's own top predicted class.

        Returns:
            1D numpy array of length window_length, values in [0, 1],
            where higher = more important for the prediction.
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass from the target class's logit -- this populates
        # self.gradients via the hook registered above.
        score = logits[0, target_class]
        score.backward()

        # activations, gradients: shape (1, num_filters, reduced_length)
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, num_filters, 1) -- alpha_k
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, reduced_length)
        cam = F.relu(cam)

        # Upsample back to the original window length for overlay on the raw signal
        window_length = input_tensor.shape[2]
        cam = F.interpolate(cam, size=window_length, mode="linear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1] for consistent visualization
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)  # degenerate case: uniform activation, no useful signal

        return cam, target_class

    def remove_hooks(self):
        """Call this when done to avoid memory leaks from lingering hooks
        (important if creating many GradCAM1D instances in a loop)."""
        self._forward_handle.remove()
        self._backward_handle.remove()
