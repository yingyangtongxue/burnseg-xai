"""
Combined XAI regularization loss for training-time supervision.

Three complementary spatial alignment terms force the model to be correct
for the right physical reasons (dNBR prior):

  1. loss_grad    : gradient saliency RRR: ∂loss_recon/∂x vs dNBR prior
  2. loss_gradcam : GradCAM:               ∂loss_recon/∂z weighted map vs prior
  3. loss_attn    : attention gate:         attention_module._last_attn vs prior

All three terms use create_graph=True so they contribute real gradients to
model weights during backpropagation.

Invariants:
  - prior.detach() is used in every distance computation: the physical prior
    must never receive gradients.
  - Batches with no burn signal (prior.amax() < 1e-8) are skipped to avoid
    penalising noise; the corresponding term returns 0.0.
  - NaN in any intermediate map causes that term to return 0.0 safely.
  - Distance metric is configurable: "mse" (default) or "cosine".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradcam import compute_gradcam

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_map(m: torch.Tensor) -> torch.Tensor:
    """
    Normalise a (B, H, W) map to [0, 1] per sample.
    NaN/Inf values are replaced by zero first.
    """
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = m.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return m / max_val


def _distance(
    map_normalized: torch.Tensor,
    prior: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """
    Scalar distance between a (B, H, W) map and the (B, H, W) dNBR prior.

    Both tensors should already be normalised to [0, 1].
    prior must have been detached by the caller.

    Args:
        map_normalized : (B, H, W) XAI map, values in [0, 1].
        prior          : (B, H, W) dNBR prior, already detached, values in [0, 1].
        metric         : "mse" or "cosine".

    Returns:
        Scalar loss tensor.
    """
    if metric == "cosine":
        B = map_normalized.shape[0]
        s_flat = map_normalized.view(B, -1)
        p_flat = prior.view(B, -1)
        return (1.0 - F.cosine_similarity(s_flat, p_flat, dim=1)).mean()
    else:  # "mse" default
        return F.mse_loss(map_normalized, prior)


# ---------------------------------------------------------------------------
# Per-term helpers (each returns a scalar tensor)
# ---------------------------------------------------------------------------

def _grad_term(
    loss_recon: torch.Tensor,
    x: torch.Tensor,
    prior: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """
    RRR gradient saliency term: ∂loss_recon/∂x vs dNBR prior.

    Mirrors the existing _rrr_loss in trainer.py but expressed as a
    standalone function that is composed inside xai_loss.

    Returns 0.0 tensor on NaN saliency or missing burn signal.
    """
    if prior.amax() < 1e-8:
        return torch.tensor(0.0, device=x.device)

    grads = torch.autograd.grad(
        outputs=loss_recon,
        inputs=x,
        create_graph=True,   # required, allows d(loss_rrr)/d(theta)
        retain_graph=True,   # required, graph reused by downstream backward()
    )[0]  # (B, C, T, H, W)

    # Aggregate over channel and time → (B, H, W)
    sal = grads.abs().mean(dim=(1, 2))
    sal = torch.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(sal).any():
        return torch.tensor(0.0, device=x.device)

    sal = _normalize_map(sal)
    return _distance(sal, prior.detach(), metric)


def _gradcam_term(
    loss_recon: torch.Tensor,
    z: torch.Tensor,
    prior: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """
    GradCAM term: ∂loss_recon/∂z weighted activation map vs dNBR prior.

    Returns 0.0 tensor on NaN map or missing burn signal.
    """
    if prior.amax() < 1e-8:
        return torch.tensor(0.0, device=z.device)

    gcam = compute_gradcam(loss_recon, z, create_graph=True)  # (B, H, W)

    if torch.isnan(gcam).any():
        return torch.tensor(0.0, device=z.device)

    gcam = _normalize_map(gcam)
    return _distance(gcam, prior.detach(), metric)


def _attn_term(
    attention_module: nn.Module,
    prior: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """
    Attention alignment term: attention_module._last_attn vs dNBR prior.

    The attention map has shape (B, 1, T, H, W).  We average over the time
    dimension and squeeze the channel dim to get (B, H, W) before comparing
    to the (B, H, W) prior.

    Returns 0.0 tensor if _last_attn is None, NaN, or no burn signal.
    """
    attn = getattr(attention_module, "_last_attn", None)

    if attn is None:
        return torch.tensor(0.0, device=prior.device)

    if prior.amax() < 1e-8:
        return torch.tensor(0.0, device=attn.device)

    # (B, 1, T, H, W) → (B, H, W): squeeze channel, average over time
    attn_map = attn.squeeze(1).mean(dim=1)   # (B, H, W)
    attn_map = torch.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(attn_map).any():
        return torch.tensor(0.0, device=attn.device)

    attn_map = _normalize_map(attn_map)
    return _distance(attn_map, prior.detach(), metric)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def xai_loss(
    loss_recon: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    prior: torch.Tensor,
    attention_module: nn.Module,
    distance_metric: str = "mse",
    terms: tuple = ("grad", "gradcam", "attn"),
) -> tuple[torch.Tensor, dict]:
    """
    Compute the XAI regularization losses for the active terms and return their average.

    Three available terms:
      1. "grad"    : gradient saliency RRR (∂loss_recon/∂x vs prior)
      2. "gradcam" : GradCAM              (∂loss_recon/∂z weighted map vs prior)
      3. "attn"    : attention alignment  (attention_module._last_attn vs prior)

    All active terms use create_graph=True so they contribute actual gradients to
    model weights during backpropagation.
    All terms skip gracefully when the batch has no burn signal
    (prior.amax() < 1e-8), returning 0.0 for that component.

    The combined loss is divided by the NUMBER OF ACTIVE TERMS (not always 3)
    so that λ has the same effective scale regardless of how many terms are active.
    This makes single-term ablation runs directly comparable to three-term runs
    at the same λ value.

    Args:
        loss_recon       : scalar reconstruction loss (live tensor, graph intact).
        x                : model input (B, C, T, H, W), requires_grad=True.
        z                : raw bottleneck before attention (B, 8, T, H, W).
        prior            : dNBR prior (B, H, W), NOT yet detached here.
        attention_module : the SpatialAttentionModule instance; its _last_attn
                           attribute must already have been set by the most
                           recent forward pass.
        distance_metric  : "mse" (default) or "cosine".
        terms            : tuple of active term names from {"grad", "gradcam", "attn"}.
                           Default: all three. For ablation, pass e.g. ("grad",).

    Returns:
        combined_loss : scalar tensor = sum(active_terms) / len(active_terms).
        components    : dict with keys "grad", "gradcam", "attn" holding each
                        individual scalar tensor (or 0.0 float for inactive/skipped).
    """
    zero_x    = torch.tensor(0.0, device=x.device)
    zero_z    = torch.tensor(0.0, device=z.device)
    zero_p    = torch.tensor(0.0, device=prior.device)

    lg = _grad_term(loss_recon, x, prior, distance_metric)    if "grad"    in terms else zero_x
    lc = _gradcam_term(loss_recon, z, prior, distance_metric) if "gradcam" in terms else zero_z
    la = _attn_term(attention_module, prior, distance_metric)  if "attn"    in terms else zero_p

    n_active = max(len([t for t in ("grad", "gradcam", "attn") if t in terms]), 1)
    combined = (lg + lc + la) / n_active

    components = {
        "grad":    lg,
        "gradcam": lc,
        "attn":    la,
    }
    return combined, components
