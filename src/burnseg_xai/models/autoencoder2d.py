import torch
import torch.nn as nn


class SpatialAttentionModule2D(nn.Module):
    """2D mirror of burnseg_xai.xai.attention.SpatialAttentionModule (kernel_size=1
    convs are already T-independent, so this is a trivial reshape, not an
    approximation)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        mid = max(in_channels // 2, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        attn_map = self.gate(z)
        return z * attn_map, attn_map


class Autoencoder2D(nn.Module):
    """
    Exact 2D-equivalent of burnseg_xai.models.autoencoder.Autoencoder for the
    T=1 case (every patch in this dataset is a single time step).

    Why this is exact, not an approximation:
    Conv3d(kernel_size=3, padding=1) applied to an input with T=1 zero-pads
    the temporal axis to T=3 and convolves with a 3-tap temporal kernel. Two
    of the three taps multiply against the zero-padding and contribute
    nothing; only the centre tap (kernel[:, :, 1, :, :]) ever touches real
    data. The result is therefore identical, term for term, to a Conv2d using
    only that centre slice. BatchNorm3d over (N, 1, H, W) reduces to
    BatchNorm2d over (N, H, W) since the temporal dimension is degenerate.
    See convert_3d_to_2d() / scripts/benchmark_inference_time_2d.py for the
    numerical proof (max abs diff vs. the original Conv3d model).

    Use only on checkpoints trained with the original Autoencoder where every
    sample has T=1 -- the conversion does not generalise to T>1 inputs.
    """

    def __init__(self, in_channels: int = 21) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.attention = SpatialAttentionModule2D(8)

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)  -- T axis already squeezed out.

        Returns:
            x_hat : (B, C, H, W)
            z     : (B, 8, H, W) raw bottleneck
        """
        z = self.encoder(x)
        z_attended, _ = self.attention(z)
        x_hat = self.decoder(z_attended)
        return x_hat, z


def convert_3d_to_2d(state_dict_3d: dict) -> dict:
    """
    Translates a state_dict from the 3D Autoencoder (Conv3d/BatchNorm3d,
    kernel (kOut, kIn, kT, kH, kW)) into one loadable by Autoencoder2D
    (Conv2d/BatchNorm2d, kernel (kOut, kIn, kH, kW)).

    For every conv weight, keeps only the centre temporal tap
    (index kT // 2) -- exact for T=1 inputs, see Autoencoder2D docstring.
    BatchNorm parameters (weight/bias/running_mean/running_var) are 1D
    per-channel tensors and are copied unchanged.
    """
    out = {}
    for key, val in state_dict_3d.items():
        if val.dim() == 5:  # Conv weight: (out, in, kT, kH, kW)
            kT = val.shape[2]
            out[key] = val[:, :, kT // 2, :, :].clone()
        else:
            out[key] = val.clone()
    return out
