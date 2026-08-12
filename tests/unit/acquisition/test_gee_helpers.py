"""Tests for pure-Python helpers in gee_download: _is_valid, _process_patch."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from burnseg_xai.acquisition.gee_download import _is_valid, _process_patch

# _is_valid

def test_is_valid_nonexistent_path(tmp_path):
    assert _is_valid(tmp_path / "missing.tif") is False


def test_is_valid_empty_file(tmp_path):
    p = tmp_path / "empty.tif"
    p.write_bytes(b"")
    assert _is_valid(p) is False


def test_is_valid_file_with_content(tmp_path):
    p = tmp_path / "data.tif"
    p.write_bytes(b"\x00" * 128)
    assert _is_valid(p) is True


# _process_patch

def _dummy_geom():
    return MagicMock()


def test_process_patch_success(tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()

    with patch("burnseg_xai.acquisition.gee_download._download_patch") as mock_dl:
        mock_dl.side_effect = lambda img, geom, out_path, timeout: out_path.write_bytes(b"x" * 64)
        pid, status = _process_patch(
            args=(0, _dummy_geom()),
            patches_dir=patches_dir,
            region="kayapo",
            patch_img=MagicMock(),
            max_retries=3,
            retry_delay=0,
            timeout=10,
        )

    assert pid == "patch_00000"
    assert status == "ok"
    assert (patches_dir / "kayapo_patch_00000.tif").exists()


def test_process_patch_skips_existing(tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    existing = patches_dir / "kayapo_patch_00001.tif"
    existing.write_bytes(b"x" * 64)

    with patch("burnseg_xai.acquisition.gee_download._download_patch") as mock_dl:
        pid, status = _process_patch(
            args=(1, _dummy_geom()),
            patches_dir=patches_dir,
            region="kayapo",
            patch_img=MagicMock(),
            max_retries=3,
            retry_delay=0,
            timeout=10,
        )
    mock_dl.assert_not_called()
    assert status == "skipped"


def test_process_patch_retries_on_failure(tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    call_count = {"n": 0}

    def _flaky(img, geom, out_path, timeout):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("transient error")
        out_path.write_bytes(b"x" * 64)

    with patch("burnseg_xai.acquisition.gee_download._download_patch", side_effect=_flaky):
        pid, status = _process_patch(
            args=(2, _dummy_geom()),
            patches_dir=patches_dir,
            region="kayapo",
            patch_img=MagicMock(),
            max_retries=3,
            retry_delay=0,
            timeout=10,
        )

    assert status == "ok"
    assert call_count["n"] == 2


def test_process_patch_exhausts_retries(tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()

    with patch("burnseg_xai.acquisition.gee_download._download_patch",
               side_effect=RuntimeError("always fails")):
        pid, status = _process_patch(
            args=(3, _dummy_geom()),
            patches_dir=patches_dir,
            region="kayapo",
            patch_img=MagicMock(),
            max_retries=2,
            retry_delay=0,
            timeout=10,
        )

    assert status.startswith("error")
    assert "always fails" in status
