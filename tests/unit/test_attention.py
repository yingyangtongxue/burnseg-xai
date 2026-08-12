"""Tests for SpatialAttentionModule."""
import pytest
import torch

from burnseg_xai.xai.attention import SpatialAttentionModule


@pytest.fixture
def attn():
    return SpatialAttentionModule(in_channels=8)


@pytest.fixture
def z():
    return torch.randn(2, 8, 1, 9, 9)


def test_attended_shape_matches_input(attn, z):
    z_att, _ = attn(z)
    assert z_att.shape == z.shape


def test_attn_map_shape(attn, z):
    _, attn_map = attn(z)
    assert attn_map.shape == (2, 1, 1, 9, 9)


def test_attn_map_in_zero_one(attn, z):
    _, attn_map = attn(z)
    assert attn_map.min().item() >= 0.0 - 1e-6
    assert attn_map.max().item() <= 1.0 + 1e-6


def test_last_attn_side_effect(attn, z):
    _, attn_map = attn(z)
    assert attn._last_attn is attn_map


def test_gradient_flows_through_attended(attn):
    z_req = torch.randn(2, 8, 1, 9, 9, requires_grad=True)
    z_att, _ = attn(z_req)
    z_att.sum().backward()
    assert z_req.grad is not None


def test_different_inputs_produce_different_maps(attn):
    z1 = torch.randn(1, 8, 1, 9, 9)
    z2 = torch.randn(1, 8, 1, 9, 9)
    _, m1 = attn(z1)
    _, m2 = attn(z2)
    assert not torch.allclose(m1, m2)
