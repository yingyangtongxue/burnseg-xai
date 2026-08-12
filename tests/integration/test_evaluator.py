"""
Integration tests for evaluator functions.
Uses FakeDataLoader + tiny untrained Autoencoder: no GeoTIFFs, no MLflow.
"""
import math

import pytest
import torch

from burnseg_xai.evaluator import (
    evaluate,
    find_optimal_threshold,
    otsu_segmentation_metrics,
    proxy_auc,
    recon_separation,
    saliency_prior_cosine,
    segmentation_metrics,
)


def test_evaluate_returns_required_keys(tiny_autoencoder, fake_val_loader):
    result = evaluate(tiny_autoencoder, fake_val_loader, device="cpu")
    required = {"recon_error_mean", "recon_error_std",
                "recon_error_p50", "recon_error_p90",
                "recon_error_p95", "recon_error_p99",
                "saliency_cosine"}
    assert required <= result.keys()


def test_evaluate_all_finite(tiny_autoencoder, fake_val_loader):
    result = evaluate(tiny_autoencoder, fake_val_loader, device="cpu")
    for k, v in result.items():
        assert math.isfinite(v), f"{k} is not finite: {v}"


def test_proxy_auc_in_range(tiny_autoencoder, fake_val_loader):
    auc = proxy_auc(tiny_autoencoder, fake_val_loader, device="cpu")
    assert 0.0 <= auc <= 1.0


def test_proxy_auc_returns_half_on_no_signal(tiny_autoencoder):
    """All-zero dNBR → only one class → should return 0.5."""
    class _NoSignalDS(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, i):
            t = torch.rand(1, 9, 9, 22)
            t[0, :, :, 20] = 0.0   # dNBR = 0, all clean
            return t

    loader = torch.utils.data.DataLoader(_NoSignalDS(), batch_size=2)
    auc = proxy_auc(tiny_autoencoder, loader, device="cpu")
    assert auc == 0.5


def test_recon_separation_finite(tiny_autoencoder, fake_val_loader):
    sep = recon_separation(tiny_autoencoder, fake_val_loader, device="cpu")
    assert math.isfinite(sep)


def test_find_optimal_threshold_finite(tiny_autoencoder, fake_val_loader):
    thr = find_optimal_threshold(tiny_autoencoder, fake_val_loader, device="cpu")
    assert math.isfinite(thr)


def test_segmentation_metrics_keys(tiny_autoencoder, fake_val_loader):
    thr = find_optimal_threshold(tiny_autoencoder, fake_val_loader, device="cpu")
    metrics = segmentation_metrics(tiny_autoencoder, fake_val_loader, device="cpu", threshold=thr)
    for key in ("f1", "precision", "recall", "accuracy", "threshold"):
        assert key in metrics


def test_otsu_metrics_keys(tiny_autoencoder, fake_val_loader):
    result = otsu_segmentation_metrics(tiny_autoencoder, fake_val_loader, device="cpu")
    for key in ("otsu_f1", "otsu_iou", "otsu_threshold"):
        assert key in result


def test_otsu_threshold_positive(tiny_autoencoder, fake_val_loader):
    result = otsu_segmentation_metrics(tiny_autoencoder, fake_val_loader, device="cpu")
    assert result["otsu_threshold"] >= 0.0


def test_saliency_prior_cosine_zeros():
    """Edge case: all-zero vectors: should return 0.0 without crashing."""
    s = torch.zeros(2, 9, 9)
    p = torch.zeros(2, 9, 9)
    score = saliency_prior_cosine(s, p)
    assert math.isfinite(score)
