"""
BurnedAreaDataset: reads GeoTIFF patches from

    {root_dir}/{region}/patches/{region}_patch_{id:05d}.tif

Each TIF has shape (C=22, H, W), one temporal snapshot with 22 spectral
channels. The dataset returns raw (un-normalised) tensors of shape

    (T=1, H, W, C=22)

Normalisation and channel selection are performed in the Trainer to allow
extracting the raw dNBR (channel 20) for the RRR prior before any scaling.

Band/channel layout (fixed, must not change between experiments):
    0-19  : spectral indices (model input)
    20    : dNBR   (excluded from model input, used as RRR spatial prior)
    21    : dNDVI  (model input, index 20 in the 21-ch input tensor)
"""

import os
from typing import Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class BurnedAreaDataset(Dataset):

    TOTAL_CHANNELS: int = 22
    DNBR_CHANNEL:   int = 20

    # Known region names (used by get_region and LORO split)
    KNOWN_REGIONS = [
        "karipuna",
        "kayapo",
        "parna_chapada_dos_guimaraes",
        "yanomami",
    ]

    # Biome membership: parna_chapada_dos_guimaraes is Cerrado; all others are Amazonia
    BIOME_MAP = {
        "karipuna":                   "amazonia",
        "kayapo":                     "amazonia",
        "yanomami":                   "amazonia",
        "parna_chapada_dos_guimaraes": "cerrado",
    }

    def __init__(
        self,
        root_dir: str,
        temporal_length: int = 1,   # each TIF = 1 time step; kept for API compat
        region: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        root_dir        : dataset root, e.g. ./data
        temporal_length : kept for API compatibility; actual T is always 1 per TIF
        region          : if given, only load patches whose filename contains
                          this string (case-insensitive); used for LORO splits
        """
        self.root_dir        = root_dir
        self.temporal_length = temporal_length
        self.region          = region

        self.samples = self._discover_patches(root_dir, region)

        if len(self.samples) == 0:
            msg = f"No .tif patches found under '{root_dir}'"
            if region:
                msg += f" for region='{region}'"
            raise FileNotFoundError(msg)

    # ------------------------------------------------------------------

    # expected canonical patch size; edge patches with smaller dimensions are excluded
    PATCH_H: int = 257
    PATCH_W: int = 257

    def _discover_patches(self, root_dir: str, region: Optional[str]):
        """
        Walks root_dir looking for files matching:
            {root_dir}/{any_region}/patches/*.tif

        Only includes patches with the canonical shape (TOTAL_CHANNELS, PATCH_H, PATCH_W).
        Edge/border patches with smaller dimensions are silently skipped.
        Returns a deterministically sorted list of absolute paths.
        """
        found = []
        skipped = 0
        for entry in sorted(os.listdir(root_dir)):
            region_dir = os.path.join(root_dir, entry)
            if not os.path.isdir(region_dir):
                continue
            patches_dir = os.path.join(region_dir, "patches")
            if not os.path.isdir(patches_dir):
                continue
            for fname in sorted(os.listdir(patches_dir)):
                if not (fname.endswith(".tif") or fname.endswith(".tiff")):
                    continue
                if region is not None and region.lower() not in fname.lower():
                    continue
                path = os.path.join(patches_dir, fname)
                with rasterio.open(path) as src:
                    if src.count != self.TOTAL_CHANNELS or src.height != self.PATCH_H or src.width != self.PATCH_W:
                        skipped += 1
                        continue
                found.append(path)
        if skipped:
            print(f"[dataset] Skipped {skipped} edge patches (non-{self.PATCH_H}x{self.PATCH_W})")
        return found

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.samples[idx]

        with rasterio.open(path) as src:
            # (C, H, W): rasterio reads bands-first
            data = src.read().astype(np.float32)

        # (C, H, W) → (H, W, C) → (1, H, W, C)  [T=1]
        data = data.transpose(1, 2, 0)[np.newaxis, ...]

        return torch.tensor(data, dtype=torch.float32)

    # ------------------------------------------------------------------

    def get_region(self, idx: int) -> str:
        """
        Infers region from the filename. Returns 'unknown' if no known
        region name is found. Used by create_loro_split().
        """
        fname = os.path.basename(self.samples[idx]).lower()
        for r in self.KNOWN_REGIONS:
            if r in fname:
                return r
        return "unknown"

    def get_biome(self, idx: int) -> str:
        """Returns the biome ('amazonia' or 'cerrado') for patch at idx."""
        return self.BIOME_MAP.get(self.get_region(idx), "unknown")

    def region_counts(self) -> dict:
        """Returns patch count per region. Useful for imbalance analysis."""
        from collections import Counter
        return dict(Counter(self.get_region(i) for i in range(len(self))))

    def biome_counts(self) -> dict:
        """Returns patch count per biome."""
        from collections import Counter
        return dict(Counter(self.get_biome(i) for i in range(len(self))))
