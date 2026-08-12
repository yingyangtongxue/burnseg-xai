"""
Bottleneck spatial attention gate for the burned-area 3D CNN autoencoder.

Invariants:
  - No spatial downsampling (no pooling, no stride > 1).
  - Output spatial dims (H, W, T) are identical to input dims.
  - The attention map is differentiable, so it can be supervised by the dNBR prior
    through xai_loss without breaking the gradient tape.
  - self._last_attn is always set after forward() so the trainer can read it.
"""

import torch
import torch.nn as nn


class SpatialAttentionModule(nn.Module):
    """
    Bottleneck spatial attention gate.

    Takes z of shape (B, C, T, H, W) and produces a scalar attention weight
    per spatial-temporal location (B, 1, T, H, W).  The gated output is
    z_attended = z * attention_weights.

    The attention map is stored in self._last_attn after every forward pass so
    that xai_loss() can compare it to the dNBR prior without a second forward.

    Architecture (no downsampling):
        Conv3d(C, C//2, kernel=1)  → ReLU
        Conv3d(C//2, 1, kernel=1)  → Sigmoid
    Both convolutions use kernel_size=1, so H, W, T are unchanged.

    Args:
        in_channels: number of input channels C (typically 8 for the bottleneck).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        mid = max(in_channels // 2, 1)
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Stores the last computed attention map; set in forward().
        # Initialised to None so downstream code can detect "not yet called".
        self._last_attn: torch.Tensor | None = None

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, C, T, H, W), bottleneck feature map.

        Returns:
            z_attended : (B, C, T, H, W), gated feature map.
            attn_map   : (B, 1, T, H, W), attention weights in [0, 1].

        Side-effect:
            self._last_attn is updated to attn_map (detach-free; gradients flow).
        """
        attn_map = self.gate(z)          # (B, 1, T, H, W)  values in [0, 1]
        self._last_attn = attn_map       # keep the live tensor (gradients intact)
        z_attended = z * attn_map        # element-wise gate; broadcast over C
        return z_attended, attn_map
