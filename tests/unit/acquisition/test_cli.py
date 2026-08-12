"""Tests for the acquisition CLI: argument parsing and validation."""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from burnseg_xai.acquisition.cli import (
    _download_all_parser,
    _download_parser,
    main_download,
    main_download_all,
)

# _download_parser

def test_download_parser_requires_aoi():
    with pytest.raises(SystemExit):
        _download_parser().parse_args(["--pre-date", "2024-01-01",
                                        "--post-date", "2024-02-01",
                                        "--dataset-root", "/tmp/ds",
                                        "--region", "kayapo"])


def test_download_parser_requires_pre_date():
    with pytest.raises(SystemExit):
        _download_parser().parse_args(["--aoi", "k.kml",
                                        "--post-date", "2024-02-01",
                                        "--dataset-root", "/tmp/ds",
                                        "--region", "kayapo"])


def test_download_parser_all_required_args():
    args = _download_parser().parse_args([
        "--aoi", "aoi/kayapo.kml",
        "--pre-date", "2024-08-17",
        "--post-date", "2024-10-18",
        "--dataset-root", "/data/dataset",
        "--region", "kayapo",
    ])
    assert args.aoi == "aoi/kayapo.kml"
    assert args.pre_date == "2024-08-17"
    assert args.post_date == "2024-10-18"
    assert args.dataset_root == "/data/dataset"
    assert args.region == "kayapo"


def test_download_parser_optional_defaults():
    args = _download_parser().parse_args([
        "--aoi", "a.kml", "--pre-date", "2024-01-01",
        "--post-date", "2024-02-01", "--dataset-root", "/d", "--region", "kayapo",
    ])
    assert args.collection == "LANDSAT/LC08/C02/T1_L2"
    assert args.buffer_days == 15
    assert args.workers == 4
    assert args.retries == 3
    assert args.timeout == 120
    assert args.project is None


# _download_all_parser

def test_download_all_parser_requires_regions_json():
    with pytest.raises(SystemExit):
        _download_all_parser().parse_args(["--dataset-root", "/tmp/ds"])


def test_download_all_parser_requires_dataset_root():
    with pytest.raises(SystemExit):
        _download_all_parser().parse_args(["--regions-json", "r.json"])


def test_download_all_parser_all_required():
    args = _download_all_parser().parse_args([
        "--regions-json", "regions.json",
        "--dataset-root", "/data/dataset",
    ])
    assert args.regions_json == "regions.json"
    assert args.dataset_root == "/data/dataset"


# main_download_all validation

def test_main_download_all_missing_json(tmp_path):
    with pytest.raises(SystemExit):
        main_download_all([
            "--regions-json", str(tmp_path / "nonexistent.json"),
            "--dataset-root", str(tmp_path),
        ])


def test_main_download_all_empty_list(tmp_path):
    json_path = tmp_path / "regions.json"
    json_path.write_text("[]")
    with pytest.raises(SystemExit):
        main_download_all([
            "--regions-json", str(json_path),
            "--dataset-root", str(tmp_path),
        ])


def test_main_download_all_calls_download_regions(tmp_path):
    regions = [{"aoi": "k.kml", "region": "kayapo",
                "pre_date": "2024-08-17", "post_date": "2024-10-18"}]
    json_path = tmp_path / "regions.json"
    json_path.write_text(json.dumps(regions))

    # download_regions is imported lazily inside main_download_all;
    # patch it at the source module rather than on cli.
    with patch("burnseg_xai.acquisition.gee_download.download_regions") as mock_fn:
        main_download_all([
            "--regions-json", str(json_path),
            "--dataset-root", str(tmp_path),
        ])
    mock_fn.assert_called_once()
    call_kwargs = mock_fn.call_args[1]
    assert call_kwargs["regions"] == regions
