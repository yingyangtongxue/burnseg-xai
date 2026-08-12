"""
Shared test helpers importable from any test file.

FakeDataset lives here (not in conftest.py) so test modules can import it
directly without relying on pytest's conftest loading order.
"""
from __future__ import annotations

from collections import Counter

import torch


class FakeDataset(torch.utils.data.Dataset):
    """
    Lightweight stub that mirrors BurnedAreaDataset's public API.
    Returns (T=1, H=9, W=9, C=22) float32 tensors: zero file I/O.
    Attributes: .samples, .get_region(), .get_biome(): needed by split functions.
    """

    KNOWN_REGIONS = ["karipuna", "kayapo", "parna_chapada_dos_guimaraes", "yanomami"]
    TOTAL_CHANNELS = 22
    DNBR_CHANNEL = 20
    PATCH_H = 9
    PATCH_W = 9

    _REGIONS_CYCLIC = [
        "karipuna", "karipuna",
        "kayapo", "kayapo",
        "parna_chapada_dos_guimaraes", "parna_chapada_dos_guimaraes",
        "yanomami", "yanomami",
    ]
    _BIOME_MAP = {
        "karipuna": "amazonia",
        "kayapo": "amazonia",
        "yanomami": "amazonia",
        "parna_chapada_dos_guimaraes": "cerrado",
    }

    def __init__(self, n: int = 8, seed: int = 0) -> None:
        self.n = n
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._data = torch.rand(n, 1, 9, 9, 22, generator=rng)
        # Even-index patches: dNBR=0.3 (burned proxy); odd: dNBR=0.0 (clean)
        for i in range(n):
            if i % 2 == 0:
                self._data[i, 0, :, :, 20] = 0.3
            else:
                self._data[i, 0, :, :, 20] = 0.0
        # build_patch_manifest accesses dataset.samples[i]
        self.samples = [
            f"fake_{self._region_for(i)}_patch_{i:05d}.tif"
            for i in range(n)
        ]

    def _region_for(self, i: int) -> str:
        return self._REGIONS_CYCLIC[i % len(self._REGIONS_CYCLIC)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        return self._data[i]

    def get_region(self, i: int) -> str:
        return self._region_for(i)

    def get_biome(self, i: int) -> str:
        return self._BIOME_MAP.get(self.get_region(i), "unknown")

    def region_counts(self) -> dict:
        return dict(Counter(self.get_region(i) for i in range(self.n)))

    def biome_counts(self) -> dict:
        return dict(Counter(self.get_biome(i) for i in range(self.n)))
