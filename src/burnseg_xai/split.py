"""
split.py: deterministic train/val/test partitioning for all experiment types.

Split hierarchy
---------------
Global split (run_experiment.py)
    save_master_split()  -- saved to split_master.json; loaded on re-runs.
    Guarantees identical patch assignments across baseline, RRR, and ablation runs.

LORO splits (run_loro.py)
    create_loro_split()  -- test = all patches from one region; train/val from rest.
    Fully deterministic: region membership is fixed, val shuffle uses seed.

Cross-biome split (run_loro.py)
    create_cross_biome_split()  -- test = all cerrado patches; train/val = all amazonia.
    Tests generalisation to an unseen biome.
"""

import json
import os
import random
from datetime import date
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Patch manifest
# ---------------------------------------------------------------------------

def build_patch_manifest(dataset) -> List[Dict]:
    """
    Returns a list of {idx, path, region, biome} for every patch in dataset.
    Preserves dataset.samples ordering (always deterministically sorted).
    """
    manifest = []
    for i in range(len(dataset)):
        region = dataset.get_region(i)
        biome  = dataset.get_biome(i)
        manifest.append({
            "idx":    i,
            "path":   dataset.samples[i],
            "region": region,
            "biome":  biome,
        })
    return manifest


# ---------------------------------------------------------------------------
# Authoritative global split (load-or-create)
# ---------------------------------------------------------------------------

def save_master_split(
    dataset,
    path: str,
    seed: int = 43,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build (or load) the authoritative global split for baseline / RRR experiments.

    First call
        Shuffles all patch indices with ``seed``, partitions into train/val/test,
        and saves the full manifest + index lists to ``path`` (JSON).

    Subsequent calls
        Loads from ``path`` directly; the split is never regenerated.
        This guarantees identical patch assignments across all runs.

    If the saved file was built from a different number of patches (dataset
    size mismatch), it is rebuilt automatically.

    Parameters
    ----------
    dataset   : BurnedAreaDataset (must have .samples, .get_region, .get_biome)
    path      : absolute path to the JSON file
    seed      : random seed for the shuffle
    train_frac, val_frac : partition fractions; test gets the remainder

    Returns
    -------
    (train_idx, val_idx, test_idx), integer indices into ``dataset``
    """
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data.get("n_total") == len(dataset):
            print(
                f"[split] Loaded master split from {path}  "
                f"(train={len(data['train_idx'])}, "
                f"val={len(data['val_idx'])}, "
                f"test={len(data['test_idx'])})"
            )
            return data["train_idx"], data["val_idx"], data["test_idx"]
        print(
            f"[split] WARNING: saved split has {data.get('n_total')} patches "
            f"but current dataset has {len(dataset)}.  Rebuilding."
        )

    manifest = build_patch_manifest(dataset)
    n = len(manifest)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)
    train_idx = sorted(indices[:n_train])
    val_idx   = sorted(indices[n_train : n_train + n_val])
    test_idx  = sorted(indices[n_train + n_val :])

    data = {
        "created":    date.today().isoformat(),
        "seed":       seed,
        "train_frac": train_frac,
        "val_frac":   val_frac,
        "n_total":    n,
        "manifest":   manifest,
        "train_idx":  train_idx,
        "val_idx":    val_idx,
        "test_idx":   test_idx,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"[split] Master split created -> {path}  "
        f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Legacy / subset split (used when max_samples is active)
# ---------------------------------------------------------------------------

def create_split(
    dataset_size: int,
    seed: int = 43,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Simple random split.  Used only when dataset is a Subset (max_samples mode).
    For full-dataset experiments use save_master_split() instead.
    """
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(train_frac * dataset_size)
    n_val   = int(val_frac   * dataset_size)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


# ---------------------------------------------------------------------------
# LORO split
# ---------------------------------------------------------------------------

def create_loro_split(
    dataset,
    test_region: str,
    val_frac: float = 0.15,
    seed: int = 43,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Leave-One-Region-Out split.

    All patches whose region == ``test_region`` go to the test set.
    The remaining patches are shuffled (seeded) and partitioned into train/val.

    Fully deterministic: given the same dataset order, seed, and test_region
    the result is always identical.
    """
    test_idx:      List[int] = []
    train_val_idx: List[int] = []

    for i in range(len(dataset)):
        if dataset.get_region(i) == test_region:
            test_idx.append(i)
        else:
            train_val_idx.append(i)

    rng = random.Random(seed)
    rng.shuffle(train_val_idx)
    n_val     = max(1, int(len(train_val_idx) * val_frac))
    val_idx   = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Cross-biome split
# ---------------------------------------------------------------------------

def create_cross_biome_split(
    dataset,
    test_biome: str = "cerrado",
    val_frac: float = 0.15,
    seed: int = 43,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Cross-biome generalisation split.

    All patches from ``test_biome`` (e.g. parna_chapada_dos_guimaraes / Cerrado)
    go to the test set.  All remaining patches (e.g. all three Amazonian regions)
    are split into train / val.

    The model is trained exclusively on Amazonia and evaluated on Cerrado --
    two biomes with different fire regimes and vegetation structure.
    """
    test_idx:      List[int] = []
    train_val_idx: List[int] = []

    for i in range(len(dataset)):
        if dataset.get_biome(i) == test_biome:
            test_idx.append(i)
        else:
            train_val_idx.append(i)

    rng = random.Random(seed)
    rng.shuffle(train_val_idx)
    n_val     = max(1, int(len(train_val_idx) * val_frac))
    val_idx   = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def save_split(split: dict, path: str) -> None:
    """Save a simple {train, val, test} index dict to JSON."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
