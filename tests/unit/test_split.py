"""Tests for all split functions."""
import json

import pytest

from burnseg_xai.split import (
    build_patch_manifest,
    create_cross_biome_split,
    create_loro_split,
    create_split,
    save_master_split,
    save_split,
)

# create_split

def test_create_split_sizes_sum_to_n():
    train, val, test = create_split(100)
    assert len(train) + len(val) + len(test) == 100


def test_create_split_no_overlap():
    train, val, test = create_split(100)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_create_split_deterministic():
    assert create_split(100, seed=7) == create_split(100, seed=7)


def test_create_split_seed_matters():
    train_a, _, _ = create_split(100, seed=1)
    train_b, _, _ = create_split(100, seed=2)
    assert train_a != train_b


def test_create_split_covers_all_indices():
    train, val, test = create_split(100)
    assert sorted(train + val + test) == list(range(100))


# create_loro_split

def test_loro_test_only_from_region(fake_dataset):
    _, _, test = create_loro_split(fake_dataset, test_region="karipuna")
    assert all(fake_dataset.get_region(i) == "karipuna" for i in test)


def test_loro_no_test_in_train_val(fake_dataset):
    train, val, test = create_loro_split(fake_dataset, test_region="karipuna")
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_loro_all_region_in_test(fake_dataset):
    _, _, test = create_loro_split(fake_dataset, test_region="karipuna")
    all_karipuna = [i for i in range(len(fake_dataset)) if fake_dataset.get_region(i) == "karipuna"]
    assert sorted(test) == sorted(all_karipuna)


def test_loro_deterministic(fake_dataset):
    a = create_loro_split(fake_dataset, test_region="kayapo", seed=42)
    b = create_loro_split(fake_dataset, test_region="kayapo", seed=42)
    assert a == b


# create_cross_biome_split

def test_cross_biome_test_only_cerrado(fake_dataset):
    _, _, test = create_cross_biome_split(fake_dataset, test_biome="cerrado")
    assert all(fake_dataset.get_biome(i) == "cerrado" for i in test)


def test_cross_biome_train_val_only_amazonia(fake_dataset):
    train, val, _ = create_cross_biome_split(fake_dataset, test_biome="cerrado")
    for i in train + val:
        assert fake_dataset.get_biome(i) == "amazonia"


def test_cross_biome_deterministic(fake_dataset):
    a = create_cross_biome_split(fake_dataset, seed=1)
    b = create_cross_biome_split(fake_dataset, seed=1)
    assert a == b


# save_master_split

def test_master_split_creates_json(fake_dataset, tmp_path):
    path = str(tmp_path / "split.json")
    save_master_split(fake_dataset, path, seed=1)
    assert (tmp_path / "split.json").exists()


def test_master_split_loads_on_second_call(fake_dataset, tmp_path):
    path = str(tmp_path / "split.json")
    first = save_master_split(fake_dataset, path, seed=1)
    second = save_master_split(fake_dataset, path, seed=99)   # seed ignored on reload
    assert first == second


def test_master_split_rebuilds_on_size_mismatch(make_fake_dataset, tmp_path):
    path = str(tmp_path / "split.json")
    ds_small = make_fake_dataset(n=8)
    save_master_split(ds_small, path, seed=1)
    ds_large = make_fake_dataset(n=12)
    save_master_split(ds_large, path, seed=1)
    with open(path) as f:
        data = json.load(f)
    assert data["n_total"] == 12


def test_master_split_sizes_correct(fake_dataset, tmp_path):
    path = str(tmp_path / "split.json")
    train, val, test = save_master_split(fake_dataset, path)
    n = len(fake_dataset)
    assert len(train) + len(val) + len(test) == n


# save_split

def test_save_split_roundtrip(tmp_path):
    path = str(tmp_path / "s.json")
    split = {"train": [0, 1, 2], "val": [3], "test": [4, 5]}
    save_split(split, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == split


def test_save_split_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "s.json")
    save_split({"train": [], "val": [], "test": []}, path)
    assert (tmp_path / "nested" / "dir" / "s.json").exists()


# build_patch_manifest

def test_manifest_length(fake_dataset):
    m = build_patch_manifest(fake_dataset)
    assert len(m) == len(fake_dataset)


def test_manifest_required_keys(fake_dataset):
    for entry in build_patch_manifest(fake_dataset):
        assert {"idx", "path", "region", "biome"} <= entry.keys()


def test_manifest_region_matches_dataset(fake_dataset):
    for entry in build_patch_manifest(fake_dataset):
        assert entry["region"] == fake_dataset.get_region(entry["idx"])
