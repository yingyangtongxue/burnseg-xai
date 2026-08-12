"""
GradCAM-style spatial importance map from the autoencoder bottleneck.

The implementation follows the GradCAM formula adapted for 3D CNNs:
  1. Compute gradients of the scalar loss w.r.t. every spatial-temporal
     location in the bottleneck tensor z  (shape B, C, T, H, W).
  2. Pool the gradients over the channel dimension to get per-location
     importance weights alpha  (shape B, 1, T, H, W).
  3. Weight the bottleneck activations by alpha and sum over channels
     → raw GradCAM map  (shape B, T, H, W).
  4. Apply ReLU (keep only positive contributions), average over time,
     and normalise per sample to [0, 1].

Invariants:
  - create_graph=True is used so the GradCAM term can backprop to model
    weights during training.
  - retain_graph=True is used so that loss_total.backward() can reuse the
    same computation graph after xai_loss returns.
  - NaN/Inf values are replaced by zero (safe fallback).
"""

import torch
import torch.nn.functional as F


def compute_gradcam(
    loss: torch.Tensor,
    z: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    GradCAM-style spatial importance map from bottleneck z.

    Steps
    -----
    1. grads  = ∂loss/∂z  with create_graph and retain_graph=True
    2. alpha  = grads.mean(dim=(2,3,4), keepdim=True)  : per-channel importance
    3. gcam   = ReLU( (alpha * z).sum(dim=1) )          : (B, T, H, W)
    4. gcam   = gcam.mean(dim=1)                         : (B, H, W)  temporal avg
    5. normalise each sample to [0, 1]
    6. NaN-safe: return zeros on NaN/Inf

    Args:
        loss         : scalar reconstruction loss (or any differentiable scalar).
        z            : bottleneck tensor (B, C, T, H, W), requires_grad must be
                       possible (the encoder must not have been called under
                       torch.no_grad()).
        create_graph : whether to build a higher-order graph for the gradient
                       computation.  Must be True during training so that the
                       GradCAM term propagates back to model weights.

    Returns:
        gradcam_map : (B, H, W) float32 tensor, values in [0, 1].
                      All zeros if the gradient is NaN/Inf.
    """
    # ------------------------------------------------------------------
    # 1. Compute ∂loss/∂z
    #    retain_graph=True is required: loss_total.backward() must be
    #    able to traverse the same graph after this call returns.
    # ------------------------------------------------------------------
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=z,
        create_graph=create_graph,   # required for training-time regularization
        retain_graph=True,           # required, graph reused by loss_total.backward()
        allow_unused=True,
    )[0]  # (B, C, T, H, W)

    if grads is None:
        return torch.zeros(z.shape[0], z.shape[3], z.shape[4], device=z.device)

    # ------------------------------------------------------------------
    # 2. NaN/Inf guard: compute on gradients before any further ops
    # ------------------------------------------------------------------
    grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 3. alpha: global average pool over spatial+temporal dims  (B, C, 1, 1, 1)
    # ------------------------------------------------------------------
    alpha = grads.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)

    # ------------------------------------------------------------------
    # 4. Weighted sum over channels, then ReLU  →  (B, T, H, W)
    # ------------------------------------------------------------------
    gcam = (alpha * z).sum(dim=1)   # (B, T, H, W)
    gcam = F.relu(gcam)

    # ------------------------------------------------------------------
    # 5. Temporal average  →  (B, H, W)
    # ------------------------------------------------------------------
    gcam = gcam.mean(dim=1)         # (B, H, W)

    # ------------------------------------------------------------------
    # 6. Per-sample normalisation to [0, 1]
    # ------------------------------------------------------------------
    gcam = torch.nan_to_num(gcam, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = gcam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    gcam = gcam / max_val

    # Final NaN guard after division
    if torch.isnan(gcam).any():
        return torch.zeros_like(gcam)

    return gcam   # (B, H, W)
