"""Tests for xai_loss: term ablation, NaN safety, differentiability."""
import pytest
import torch
import torch.nn.functional as F

from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.xai.losses import xai_loss


def _xai_setup(seed: int = 0):
    """
    Forward pass through model to produce (loss_recon, x, z, prior, attn).
    Calling model.train() ensures BN tracks running stats for small batches.
    """
    torch.manual_seed(seed)
    model = Autoencoder(in_channels=21)
    model.train()
    x = torch.randn(2, 21, 1, 9, 9, requires_grad=True)
    x_hat, z = model(x)
    loss_recon = F.mse_loss(x_hat, x.detach())
    prior = torch.zeros(2, 9, 9)
    prior[0] = 0.4   # burn signal on first sample
    return loss_recon, x, z, prior, model.attention


def test_all_terms_keys_present():
    loss_recon, x, z, prior, attn = _xai_setup()
    _, components = xai_loss(loss_recon, x, z, prior, attn)
    assert set(components.keys()) == {"grad", "gradcam", "attn"}


def test_combined_loss_is_scalar():
    loss_recon, x, z, prior, attn = _xai_setup()
    combined, _ = xai_loss(loss_recon, x, z, prior, attn)
    assert combined.dim() == 0


def test_ablation_grad_only():
    loss_recon, x, z, prior, attn = _xai_setup()
    combined, components = xai_loss(loss_recon, x, z, prior, attn, terms=("grad",))
    assert components["gradcam"] == 0.0
    assert components["attn"] == 0.0
    assert combined.item() == components["grad"].item()


def test_ablation_attn_only():
    loss_recon, x, z, prior, attn = _xai_setup()
    combined, components = xai_loss(loss_recon, x, z, prior, attn, terms=("attn",))
    assert components["grad"] == 0.0
    assert components["gradcam"] == 0.0
    assert combined.item() == components["attn"].item()


def test_no_signal_prior_returns_zero():
    loss_recon, x, z, _, attn = _xai_setup()
    prior_zero = torch.zeros(2, 9, 9)
    combined, _ = xai_loss(loss_recon, x, z, prior_zero, attn)
    assert combined.item() == 0.0


def test_combined_loss_differentiable():
    loss_recon, x, z, prior, attn = _xai_setup()
    combined, _ = xai_loss(loss_recon, x, z, prior, attn)
    combined.backward(retain_graph=True)   # must not raise


def test_prior_not_modified():
    loss_recon, x, z, prior, attn = _xai_setup()
    prior_orig = prior.clone()
    xai_loss(loss_recon, x, z, prior, attn)
    assert torch.allclose(prior, prior_orig)


def test_cosine_distance_metric():
    loss_recon, x, z, prior, attn = _xai_setup()
    combined, _ = xai_loss(loss_recon, x, z, prior, attn, distance_metric="cosine")
    assert not torch.isnan(combined)
    assert combined.dim() == 0


def test_combined_loss_finite():
    loss_recon, x, z, prior, attn = _xai_setup()
    combined, _ = xai_loss(loss_recon, x, z, prior, attn)
    assert torch.isfinite(combined)
