"""Tests for ProjectConfig and load_config."""
import pytest

from burnseg_xai.config.loader import load_config
from burnseg_xai.config.schema import ProjectConfig


def test_default_config_in_channels():
    assert ProjectConfig().in_channels == 21


def test_default_config_lambda_rrr():
    assert ProjectConfig().lambda_rrr == 0.0


def test_default_config_seed():
    assert ProjectConfig().seed == 43


def test_load_config_none_returns_defaults():
    cfg = load_config(None)
    assert isinstance(cfg, ProjectConfig)
    assert cfg.in_channels == 21


def test_load_config_flat_override(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("batch_size: 16\nlr: 0.01\n")
    cfg = load_config(str(yaml_path))
    assert cfg.batch_size == 16
    assert abs(cfg.lr - 0.01) < 1e-9


def test_load_config_nested_override(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("training:\n  batch_size: 8\n  epochs: 50\n")
    cfg = load_config(str(yaml_path))
    assert cfg.batch_size == 8
    assert cfg.epochs == 50


def test_load_config_unknown_key_does_not_raise(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("nonexistent_key: 999\n")
    cfg = load_config(str(yaml_path))
    assert cfg.in_channels == 21   # defaults unchanged


def test_load_config_partial_override_leaves_rest_intact(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("seed: 99\n")
    cfg = load_config(str(yaml_path))
    assert cfg.seed == 99
    assert cfg.in_channels == 21   # not overridden → default
