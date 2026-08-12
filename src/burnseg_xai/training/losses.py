"""
DEPRECATED: this module is superseded by burnseg_xai.xai.losses.

The combined XAI loss (gradient saliency + GradCAM + spatial attention alignment)
lives in:

    from burnseg_xai.xai.losses import xai_loss

This file is kept to avoid breaking any external scripts that may import
get_loss() or rrr_loss() by name.  The canonical trainer (training/trainer.py)
does NOT import from here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss() -> nn.MSELoss:
    return nn.MSELoss()


def rrr_loss(saliency: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Single-term MSE RRR loss. Superseded by xai.losses.xai_loss."""
    return F.mse_loss(saliency, prior)
