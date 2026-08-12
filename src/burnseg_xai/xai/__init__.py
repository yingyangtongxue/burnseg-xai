"""
burnseg_xai.xai: training-time XAI regularization package.

Exports:
    SpatialAttentionModule  : bottleneck spatial gate supervised by dNBR prior
    compute_gradcam         : GradCAM-style map from bottleneck gradients
    xai_loss                : combined XAI regularization loss (grad + gradcam + attn)
"""

from .attention import SpatialAttentionModule
from .gradcam import compute_gradcam
from .losses import xai_loss

__all__ = ["SpatialAttentionModule", "compute_gradcam", "xai_loss"]
