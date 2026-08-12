"""
Integration tests for Trainer: train loop, checkpoint save/resume.
Uses FakeDataLoader (H=W=9) + mock logger: no GeoTIFFs, no MLflow, < 30 s.
"""
import math

import pytest
import torch

from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.training.trainer import Trainer


def _make_trainer(tmp_path, mock_logger, lambda_rrr=0.0, patience=10):
    model = Autoencoder(in_channels=21)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        logger=mock_logger,
        lambda_rrr=lambda_rrr,
        checkpoint_dir=str(tmp_path / "ckpts"),
        early_stopping_patience=patience,
    )


def test_train_returns_dict(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger)
    result = trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    assert isinstance(result, dict)


def test_train_loss_finite(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger)
    result = trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    assert math.isfinite(result["train_loss"])
    assert math.isfinite(result["val_loss"])


def test_checkpoint_created(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger)
    trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    assert (tmp_path / "ckpts" / "checkpoint_latest.pt").exists()


def test_checkpoint_has_required_keys(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger)
    trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    state = torch.load(tmp_path / "ckpts" / "checkpoint_latest.pt", map_location="cpu")
    required = {"epoch", "model_state_dict", "optimizer_state_dict",
                "scaler_state_dict", "best_val_loss", "es_counter", "mlflow_run_id"}
    assert required <= state.keys()


def test_checkpoint_epoch_value(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger)
    trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    state = torch.load(tmp_path / "ckpts" / "checkpoint_latest.pt", map_location="cpu")
    assert state["epoch"] == 2   # stored as next epoch (0-indexed + 1 after 2 epochs)


def test_load_checkpoint_returns_epoch(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer1 = _make_trainer(tmp_path, mock_logger)
    trainer1.train(fake_train_loader, fake_val_loader, epochs=2)

    trainer2 = _make_trainer(tmp_path, mock_logger)
    ckpt = str(tmp_path / "ckpts" / "checkpoint_latest.pt")
    epoch = trainer2.load_checkpoint(ckpt)
    assert epoch == 2


def test_load_checkpoint_restores_weights(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer1 = _make_trainer(tmp_path, mock_logger)
    trainer1.train(fake_train_loader, fake_val_loader, epochs=2)
    saved_weights = {k: v.clone() for k, v in trainer1.model.state_dict().items()}

    trainer2 = _make_trainer(tmp_path, mock_logger)
    trainer2.load_checkpoint(str(tmp_path / "ckpts" / "checkpoint_latest.pt"))
    for key, val in saved_weights.items():
        assert torch.allclose(val, trainer2.model.state_dict()[key])


def test_auto_resume_trains_one_more_epoch(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer1 = _make_trainer(tmp_path, mock_logger)
    trainer1.train(fake_train_loader, fake_val_loader, epochs=2)

    trainer2 = _make_trainer(tmp_path, mock_logger)
    # checkpoint_latest.pt in ckpts/ → auto-resume picks it up
    trainer2.train(fake_train_loader, fake_val_loader, epochs=3)
    state = torch.load(tmp_path / "ckpts" / "checkpoint_latest.pt", map_location="cpu")
    assert state["epoch"] == 3


def test_logger_log_metrics_called(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger)
    trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    assert mock_logger.log_metrics.call_count >= 2   # at least once per epoch


def test_rrr_loss_nonzero_when_enabled(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger, lambda_rrr=1.0)
    result = trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    assert result["train_rrr_loss"] > 0.0


def test_rrr_loss_zero_when_disabled(tmp_path, mock_logger, fake_train_loader, fake_val_loader):
    trainer = _make_trainer(tmp_path, mock_logger, lambda_rrr=0.0)
    result = trainer.train(fake_train_loader, fake_val_loader, epochs=2)
    assert result["train_rrr_loss"] == 0.0
