"""Tests for set_seed determinism."""
import pytest
import torch

from burnseg_xai.utils.seed import set_seed


def test_same_seed_produces_same_tensor():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_different_seeds_produce_different_tensors():
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    b = torch.rand(5)
    assert not torch.allclose(a, b)


def test_set_seed_does_not_raise():
    set_seed(0)
    set_seed(42)
    set_seed(999_999)
