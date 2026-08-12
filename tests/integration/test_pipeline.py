"""
Integration smoke test for run_experiment.run().
Monkeypatches BurnedAreaDataset → FakeDataset, MLflowLogger → mock,
and all visualization functions → no-ops. Runs 1 epoch on CPU only.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from burnseg_xai.pipeline.run_experiment import run


@pytest.fixture
def pipeline_config(mock_config):
    """Config configured for a minimal, fast pipeline run."""
    mock_config.epochs = 1
    mock_config.max_samples = None  # uses save_master_split; n=8 → val non-empty
    mock_config.early_stopping_patience = 5
    return mock_config


@pytest.fixture
def _patch_pipeline(pipeline_config, monkeypatch):
    """Monkeypatches all external dependencies for the pipeline test."""
    from tests.helpers import FakeDataset

    monkeypatch.setattr(
        "burnseg_xai.pipeline.run_experiment.BurnedAreaDataset",
        lambda root_dir, temporal_length=1: FakeDataset(n=8),
    )

    # MLflowLogger → mock (no real mlflow writes)
    fake_logger = MagicMock()
    fake_logger.start_run.return_value = "test-run-id"
    monkeypatch.setattr(
        "burnseg_xai.pipeline.run_experiment.MLflowLogger",
        lambda cfg: fake_logger,
    )

    # mlflow.active_run() → mock (called directly in run_experiment.py)
    fake_run = MagicMock()
    fake_run.info.run_id = "test-run-id"
    monkeypatch.setattr(
        "burnseg_xai.pipeline.run_experiment.mlflow.active_run",
        lambda: fake_run,
    )

    # All visualization functions → no-ops
    for fn_name in (
        "make_saliency_figures",
        "make_comparison_figures",
        "log_figures_to_mlflow",
    ):
        monkeypatch.setattr(
            f"burnseg_xai.pipeline.run_experiment.{fn_name}",
            lambda *a, **kw: ([], []),
        )
    for fn_name in ("plot_training_curves", "plot_recon_error_distribution"):
        monkeypatch.setattr(
            f"burnseg_xai.pipeline.run_experiment.{fn_name}",
            lambda *a, **kw: MagicMock(),
        )
    monkeypatch.setattr(
        "burnseg_xai.pipeline.run_experiment._log_fig",
        lambda *a, **kw: None,
    )

    return fake_logger


def test_run_completes(pipeline_config, _patch_pipeline):
    result = run(pipeline_config, run_name="test_run")
    assert isinstance(result, dict)


def test_run_returns_loss_keys(pipeline_config, _patch_pipeline):
    result = run(pipeline_config, run_name="test_run")
    assert "train_loss" in result
    assert "val_loss" in result


def test_model_saved(pipeline_config, _patch_pipeline):
    run(pipeline_config, run_name="my_run")
    model_path = Path(pipeline_config.output_root) / "my_run_model_final.pt"
    assert model_path.exists()


def test_split_json_saved(pipeline_config, _patch_pipeline):
    run(pipeline_config, run_name="test_run")
    split_path = Path(pipeline_config.output_root) / "split_master.json"
    assert split_path.exists()


def test_log_params_called_with_lambda_rrr(pipeline_config, _patch_pipeline):
    run(pipeline_config, run_name="test_run")
    mock_logger = _patch_pipeline
    all_calls = mock_logger.log_params.call_args_list
    params_logged = {}
    for call in all_calls:
        params_logged.update(call[0][0] if call[0] else call[1].get("params", {}))
    assert "lambda_rrr" in params_logged


def test_no_mlruns_in_cwd(_patch_pipeline, pipeline_config):
    """Verify that no real mlruns/ directory is created in the current working dir."""
    run(pipeline_config, run_name="test_run")
    assert not Path("mlruns").exists()
