import torch
import torch.nn as nn

from burnseg_xai.xai.attention import SpatialAttentionModule


class Autoencoder(nn.Module):
    """
    3D CNN autoencoder for spatiotemporal burned area anomaly detection.

    Architecture:
      Encoder:   21 -> 64 -> 32 -> 16 -> 8   (channel bottleneck: 2.6x compression)
      Attention: 8  -> 8   (learned spatial gate supervised by dNBR prior)
      Decoder:   8  -> 16 -> 32 -> 64 -> 21

    The channel bottleneck (8 << 21) forces the model to learn a compressed
    representation. Burned patches are anomalous and therefore harder to
    reconstruct, yielding higher reconstruction error.

    The spatial attention module sits between encoder and decoder.  It gates
    the bottleneck feature map so that the decoder attends to physically
    meaningful spatial regions (guided by the dNBR prior through xai_loss).

    Invariants (must not be violated):
      - No spatial downsampling: no pooling, no stride > 1
      - H and W are identical at input and output
      - forward() returns (x_hat, z) as a 2-tuple: reconstruction and the
        RAW (pre-attention) bottleneck latent
      - No sigmoid/softmax on the output layer
    """

    def __init__(self, in_channels: int = 21) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        # 8 = bottleneck channels; no downsampling inside SpatialAttentionModule
        self.attention = SpatialAttentionModule(8)

        self.decoder = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T, H, W)

        Returns:
            x_hat : (B, C, T, H, W), reconstruction (decoder uses attended z)
            z     : (B, 8, T, H, W), raw bottleneck (pre-attention) for GradCAM

        The decoder receives z_attended (gated by SpatialAttentionModule) so the
        model learns to focus on physically meaningful regions.  GradCAM and the
        trainer receive the raw z so that gradients w.r.t. the bottleneck reflect
        feature importance before the learned gate.

        Side-effect:
            self.attention._last_attn is set to the current attention map.
        """
        z = self.encoder(x)                        # (B, 8, T, H, W)  raw bottleneck
        z_attended, _ = self.attention(z)          # gate; _last_attn stored inside
        x_hat = self.decoder(z_attended)           # decoder uses attended features
        return x_hat, z                            # z = raw, for GradCAM
