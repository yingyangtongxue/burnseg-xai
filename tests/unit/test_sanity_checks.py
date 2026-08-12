"""Tests for run_sanity_checks: pass/fail cases for all six checks."""
import pytest
import torch

from burnseg_xai.config.schema import ProjectConfig
from burnseg_xai.sanity_checks import run_sanity_checks


def _clean_cfg():
    cfg = ProjectConfig()
    cfg.in_channels = 21
    cfg.lambda_rrr = 0.0
    return cfg


def test_passes_with_clean_data(fake_dataset, mock_config):
    run_sanity_checks(fake_dataset, mock_config, n_samples=4)   # should not raise


def test_raises_on_nan(mock_config):
    class _NaNDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, i):
            t = torch.rand(1, 9, 9, 22)
            t[0, 0, 0, 20] = float("nan")
            return t

    with pytest.raises(ValueError, match="NaN"):
        run_sanity_checks(_NaNDataset(), mock_config, n_samples=2)


def test_raises_on_inf(mock_config):
    class _InfDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, i):
            t = torch.rand(1, 9, 9, 22)
            t[0, 0, 0, 5] = float("inf")
            return t

    with pytest.raises(ValueError, match="Inf"):
        run_sanity_checks(_InfDataset(), mock_config, n_samples=2)


def test_raises_on_wrong_channel_count(mock_config):
    class _BadChannels(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, i): return torch.rand(1, 9, 9, 10)   # 10 channels, not 22

    with pytest.raises(ValueError, match="22"):
        run_sanity_checks(_BadChannels(), mock_config, n_samples=2)


def test_raises_on_wrong_in_channels(fake_dataset):
    cfg = _clean_cfg()
    cfg.in_channels = 22   # must be 21
    with pytest.raises(ValueError, match="in_channels"):
        run_sanity_checks(fake_dataset, cfg, n_samples=2)


def test_raises_on_negative_lambda(fake_dataset):
    cfg = _clean_cfg()
    cfg.lambda_rrr = -0.1
    with pytest.raises(ValueError, match="lambda_rrr"):
        run_sanity_checks(fake_dataset, cfg, n_samples=2)
