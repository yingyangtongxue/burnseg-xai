"""
GEE integration tests: require live credentials.
Run manually:
    pytest tests/gee/ -m gee -v

These tests are NEVER run in CI (no GEE credentials available).
"""
from pathlib import Path

import numpy as np
import pytest
import rasterio

KEY_FILE = "./secrets/gee_service_account.json"
AOI_KML  = "./data/aoi/parna_chapada_dos_guimaraes.kml"
REGION   = "parna_chapada_dos_guimaraes"
PRE_DATE = "2019-07-17"
POST_DATE = "2019-08-18"


@pytest.mark.gee
def test_init_ee_with_real_key():
    from burnseg_xai.acquisition.gee_download import _init_ee
    _init_ee(key_file=KEY_FILE)   # should not raise


@pytest.mark.gee
def test_download_aoi_produces_patches(tmp_path):
    from burnseg_xai.acquisition.gee_download import download_aoi

    download_aoi(
        aoi_kml=AOI_KML,
        collection="LANDSAT/LC08/C02/T1_L2",
        pre_date=PRE_DATE,
        post_date=POST_DATE,
        dataset_root=tmp_path,
        region=REGION,
        buffer_days=15,
        max_workers=2,
        key_file=KEY_FILE,
    )

    patches_dir = tmp_path / REGION / "patches"
    patches = sorted(patches_dir.glob("*.tif"))
    assert len(patches) > 0, "No patches downloaded"


@pytest.mark.gee
def test_downloaded_patch_shape(tmp_path):
    from burnseg_xai.acquisition.gee_download import download_aoi

    download_aoi(
        aoi_kml=AOI_KML,
        collection="LANDSAT/LC08/C02/T1_L2",
        pre_date=PRE_DATE,
        post_date=POST_DATE,
        dataset_root=tmp_path,
        region=REGION,
        buffer_days=15,
        max_workers=2,
        key_file=KEY_FILE,
    )

    patches = sorted((tmp_path / REGION / "patches").glob("*.tif"))
    with rasterio.open(patches[0]) as src:
        data = src.read()
    assert data.shape[0] == 22
    assert data.shape[1] == 257
    assert data.shape[2] == 257


@pytest.mark.gee
def test_resume_skips_existing(tmp_path):
    from burnseg_xai.acquisition.gee_download import download_aoi

    kwargs = dict(
        aoi_kml=AOI_KML, collection="LANDSAT/LC08/C02/T1_L2",
        pre_date=PRE_DATE, post_date=POST_DATE,
        dataset_root=tmp_path, region=REGION,
        buffer_days=15, max_workers=2, key_file=KEY_FILE,
    )
    download_aoi(**kwargs)
    n_first = len(sorted((tmp_path / REGION / "patches").glob("*.tif")))

    # Second call: all patches exist → skipped (no re-download)
    import time
    ts_before = time.time()
    download_aoi(**kwargs)
    elapsed = time.time() - ts_before
    n_second = len(sorted((tmp_path / REGION / "patches").glob("*.tif")))

    assert n_second == n_first
    assert elapsed < 30, "Resume should be fast (< 30 s): patches not re-downloaded"


@pytest.mark.gee
def test_download_regions_single_entry(tmp_path):
    from burnseg_xai.acquisition.gee_download import download_regions

    regions = [{"aoi": AOI_KML, "region": REGION,
                "pre_date": PRE_DATE, "post_date": POST_DATE}]
    download_regions(regions=regions, dataset_root=tmp_path, key_file=KEY_FILE)
    assert (tmp_path / REGION / "patches").exists()
