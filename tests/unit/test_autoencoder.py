"""Tests for Autoencoder: shape invariants, forward tuple, spatial preservation."""
import pytest
import torch

from burnseg_xai.models.autoencoder import Autoencoder


def test_forward_returns_two_tuple(tiny_autoencoder, model_input):
    out = tiny_autoencoder(model_input)
    assert isinstance(out, tuple) and len(out) == 2


def test_xhat_shape_matches_input(tiny_autoencoder, model_input):
    x_hat, _ = tiny_autoencoder(model_input)
    assert x_hat.shape == model_input.shape


def test_bottleneck_channel_count(tiny_autoencoder, model_input):
    _, z = tiny_autoencoder(model_input)
    assert z.shape[1] == 8


def test_spatial_dims_preserved(tiny_autoencoder):
    """No pooling / stride: H and W must be identical at input and output."""
    x = torch.randn(1, 21, 1, 32, 32)
    x_hat, _ = tiny_autoencoder(x)
    assert x_hat.shape[-2:] == (32, 32)


def test_temporal_dim_preserved(tiny_autoencoder, model_input):
    x_hat, z = tiny_autoencoder(model_input)
    assert x_hat.shape[2] == model_input.shape[2]
    assert z.shape[2] == model_input.shape[2]


def test_output_channel_count(tiny_autoencoder, model_input):
    x_hat, _ = tiny_autoencoder(model_input)
    assert x_hat.shape[1] == 21


def test_attention_last_attn_set_after_forward(tiny_autoencoder, model_input):
    tiny_autoencoder(model_input)
    assert tiny_autoencoder.attention._last_attn is not None


def test_gradient_flows_from_loss_to_input():
    model = Autoencoder(in_channels=21)
    model.train()
    x = torch.randn(2, 21, 1, 9, 9, requires_grad=True)
    x_hat, _ = model(x)
    x_hat.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_custom_in_channels_builds_and_runs():
    model = Autoencoder(in_channels=3)
    x = torch.randn(1, 3, 1, 9, 9)
    x_hat, z = model(x)
    assert x_hat.shape[1] == 3
