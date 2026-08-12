"""
Shared fixtures for the burnseg-xai test suite.

All tests use CPU-only tiny tensors for speed. FakeDataset mimics
BurnedAreaDataset's public API without any file I/O.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from burnseg_xai.config.schema import ProjectConfig
from burnseg_xai.models.autoencoder import Autoencoder
from tests.helpers import FakeDataset  # noqa: F401: re-exported for fixtures below

# Tensor fixtures

@pytest.fixture
def raw_batch() -> torch.Tensor:
    """(B=2, T=1, H=9, W=9, C=22) float32: matches DataLoader output shape."""
    t = torch.rand(2, 1, 9, 9, 22)
    t[0, 0, :, :, 20] = 0.4   # dNBR signal on first sample
    return t


@pytest.fixture
def model_input() -> torch.Tensor:
    """(B=2, C=21, T=1, H=9, W=9) with requires_grad=True."""
    return torch.randn(2, 21, 1, 9, 9, requires_grad=True)


@pytest.fixture
def prior() -> torch.Tensor:
    """(B=2, H=9, W=9) values in [0, 1]; first sample has burn signal."""
    p = torch.zeros(2, 9, 9)
    p[0] = torch.rand(9, 9) * 0.5 + 0.1   # signal in [0.1, 0.6]
    return p


@pytest.fixture
def bottleneck() -> torch.Tensor:
    """(B=2, C=8, T=1, H=9, W=9) with requires_grad=True."""
    return torch.randn(2, 8, 1, 9, 9, requires_grad=True)


# Model fixtures

@pytest.fixture
def tiny_autoencoder() -> Autoencoder:
    model = Autoencoder(in_channels=21)
    model.eval()
    return model


@pytest.fixture
def tiny_optimizer(tiny_autoencoder: Autoencoder) -> torch.optim.Optimizer:
    return torch.optim.Adam(tiny_autoencoder.parameters(), lr=1e-3)


# Dataset / DataLoader fixtures

@pytest.fixture
def fake_dataset() -> FakeDataset:
    return FakeDataset(n=8)


@pytest.fixture
def fake_train_loader() -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        FakeDataset(n=8, seed=0), batch_size=2, shuffle=False
    )


@pytest.fixture
def fake_val_loader() -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        FakeDataset(n=4, seed=1), batch_size=2, shuffle=False
    )


@pytest.fixture
def make_fake_dataset():
    """Factory: make_fake_dataset(n=8, seed=0) -> FakeDataset."""
    def _factory(n: int = 8, seed: int = 0) -> FakeDataset:
        return FakeDataset(n=n, seed=seed)
    return _factory


# GeoTIFF factory (only for test_dataset.py)

@pytest.fixture
def make_geotiff(tmp_path: Path):
    """
    Writes a real GeoTIFF at {tmp_path}/{region}/patches/{region}_patch_{id:05d}.tif.
    Default shape: (22, 257, 257): the exact size BurnedAreaDataset accepts.
    Pass a different shape to test the skipping logic.
    """
    def _make(region: str, patch_id: int, shape: tuple = (22, 257, 257)) -> Path:
        path = tmp_path / region / "patches" / f"{region}_patch_{patch_id:05d}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.random.rand(*shape).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, shape[2], shape[1])
        with rasterio.open(
            path, "w", driver="GTiff",
            height=shape[1], width=shape[2], count=shape[0],
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)
        return path
    return _make


# Logger / Config fixtures

@pytest.fixture
def mock_logger() -> MagicMock:
    logger = MagicMock()
    logger.start_run.return_value = "test-run-id"
    return logger


@pytest.fixture
def mock_config(tmp_path: Path) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.dataset_root            = str(tmp_path / "dataset")
    cfg.output_root             = str(tmp_path / "output")
    cfg.checkpoint_dir          = str(tmp_path / "checkpoints")
    cfg.mlflow_tracking_uri     = str(tmp_path / "mlruns")
    cfg.mlflow_experiment       = "test_experiment"
    cfg.batch_size              = 2
    cfg.epochs                  = 2
    cfg.lr                      = 1e-3
    cfg.lambda_rrr              = 0.0
    cfg.device                  = "cpu"
    cfg.num_workers             = 0
    cfg.max_samples             = None
    cfg.in_channels             = 21
    cfg.seed                    = 42
    cfg.early_stopping_patience = 10
    cfg.temporal_length         = 1
    cfg.regions                 = ["karipuna", "kayapo"]
    return cfg
