"""Tests for compute_gradcam: shape, normalization, NaN safety."""
import pytest
import torch

from burnseg_xai.xai.gradcam import compute_gradcam


@pytest.fixture
def z_req():
    return torch.randn(2, 8, 1, 9, 9, requires_grad=True)


def test_output_shape(z_req):
    loss = z_req.mean()
    cam = compute_gradcam(loss, z_req, create_graph=False)
    assert cam.shape == (2, 9, 9)


def test_output_in_zero_one(z_req):
    loss = z_req.mean()
    cam = compute_gradcam(loss, z_req, create_graph=False)
    assert cam.min().item() >= 0.0 - 1e-6
    assert cam.max().item() <= 1.0 + 1e-6


def test_z_not_in_graph_returns_zeros():
    # z.requires_grad=True but is not connected to loss (allow_unused path).
    # grads[0] is None → compute_gradcam returns zeros_like(z mean).
    z = torch.randn(2, 8, 1, 9, 9, requires_grad=True)
    y = torch.randn(1, requires_grad=True)
    loss = y.mean()   # loss is computed from y, not from z
    cam = compute_gradcam(loss, z, create_graph=False)
    assert torch.all(cam == 0.0)


def test_no_nan_in_output(z_req):
    cam = compute_gradcam(z_req.mean(), z_req, create_graph=False)
    assert not torch.isnan(cam).any()


def test_create_graph_false_valid_shape():
    z = torch.randn(1, 8, 1, 9, 9, requires_grad=True)
    cam = compute_gradcam(z.mean(), z, create_graph=False)
    assert cam.shape == (1, 9, 9)


def test_batch_dimension_preserved():
    z = torch.randn(4, 8, 1, 9, 9, requires_grad=True)
    cam = compute_gradcam(z.mean(), z, create_graph=False)
    assert cam.shape[0] == 4
