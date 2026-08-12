"""
Integration tests for BurnedAreaDataset.
These are the ONLY tests that create real 257×257 GeoTIFF files.
"""
import pytest
import torch

from burnseg_xai.dataset import BurnedAreaDataset


@pytest.fixture
def dataset_root(make_geotiff, tmp_path):
    """Create 2 karipuna + 2 parna_chapada patches in a canonical layout."""
    make_geotiff("karipuna", 0)
    make_geotiff("karipuna", 1)
    make_geotiff("parna_chapada_dos_guimaraes", 0)
    make_geotiff("parna_chapada_dos_guimaraes", 1)
    return tmp_path


def test_discovers_all_patches(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    assert len(ds) == 4


def test_getitem_shape(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    sample = ds[0]
    assert sample.shape == (1, 257, 257, 22)


def test_getitem_dtype(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    assert ds[0].dtype == torch.float32


def test_wrong_size_patch_skipped(make_geotiff, tmp_path):
    make_geotiff("karipuna", 0)                   # correct 257×257
    make_geotiff("karipuna", 1, shape=(22, 100, 100))   # wrong size → skipped
    ds = BurnedAreaDataset(str(tmp_path))
    assert len(ds) == 1


def test_get_region_known(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    regions = {ds.get_region(i) for i in range(len(ds))}
    assert "karipuna" in regions
    assert "parna_chapada_dos_guimaraes" in regions


def test_get_region_unknown(make_geotiff, tmp_path):
    make_geotiff("unknown_area", 0)
    ds = BurnedAreaDataset(str(tmp_path))
    assert ds.get_region(0) == "unknown"


def test_get_biome_cerrado(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    biomes = {ds.get_biome(i) for i in range(len(ds))}
    assert "cerrado" in biomes


def test_get_biome_amazonia(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    biomes = {ds.get_biome(i) for i in range(len(ds))}
    assert "amazonia" in biomes


def test_region_counts(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root))
    counts = ds.region_counts()
    assert counts["karipuna"] == 2
    assert counts["parna_chapada_dos_guimaraes"] == 2


def test_no_patches_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        BurnedAreaDataset(str(tmp_path))


def test_patches_sorted_deterministically(dataset_root):
    ds1 = BurnedAreaDataset(str(dataset_root))
    ds2 = BurnedAreaDataset(str(dataset_root))
    assert ds1.samples == ds2.samples


def test_region_filter(dataset_root):
    ds = BurnedAreaDataset(str(dataset_root), region="karipuna")
    assert len(ds) == 2
    assert all("karipuna" in ds.get_region(i) for i in range(len(ds)))
