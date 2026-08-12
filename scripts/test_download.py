"""
End-to-end smoke test for the one-step acquisition pipeline.

Downloads pre/post Landsat 8 composites for parna_chapada_dos_guimaraes
(smallest AOI, ~12 patches) and writes 22-channel patches directly to the
BurnedAreaDataset-compatible layout in ./outputs/smoke_test.

Expected output:
    ./outputs/smoke_test/
    └── parna_chapada_dos_guimaraes/
        ├── patches/    parna_chapada_dos_guimaraes_patch_00000.tif  (22ch, 257x257)
        ├── metadata/   dataset_metadata.json
        ├── _raw/       pre_fire_2019-07-17/ + post_fire_2019-08-18/  (download cache)
        └── progress.json

Run from repo root:
    python scripts/test_download.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Credentials: Earth Engine service account key.
# client_email is read directly from the JSON; no .env needed.
# ---------------------------------------------------------------------------
KEY_FILE = "./secrets/gee_service_account.json"
os.environ.pop("SERVICE_ACCOUNT_EMAIL", None)   # let _init_ee read from key JSON

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
AOI_KML      = "./data/aoi/parna_chapada_dos_guimaraes.kml"
REGION       = "parna_chapada_dos_guimaraes"     # must match KNOWN_REGIONS
PRE_DATE     = "2019-07-17"
POST_DATE    = "2019-08-18"
COLLECTION   = "LANDSAT/LC08/C02/T1_L2"
DATASET_ROOT = Path("./data/smoke_test")

# ---------------------------------------------------------------------------
# Make the package importable when running from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from burnseg_xai.acquisition.gee_download import download_aoi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=" * 60)
print("One-step download + build")
print(f"  Region       : {REGION}")
print(f"  Pre-fire     : {PRE_DATE}  (±15 days)")
print(f"  Post-fire    : {POST_DATE} (±15 days)")
print(f"  Collection   : {COLLECTION}")
print(f"  Dataset root : {DATASET_ROOT}")
print("=" * 60)

download_aoi(
    aoi_kml=AOI_KML,
    collection=COLLECTION,
    pre_date=PRE_DATE,
    post_date=POST_DATE,
    dataset_root=DATASET_ROOT,
    region=REGION,
    buffer_days=15,
    max_workers=2,
    max_retries=3,
    timeout=120,
    key_file=KEY_FILE,
)

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
import numpy as np
import rasterio

patches_dir = DATASET_ROOT / REGION / "patches"
patches = sorted(patches_dir.glob("*.tif"))

if not patches:
    print("FAIL: no patches found.")
    sys.exit(1)

with rasterio.open(patches[0]) as src:
    data = src.read().astype(np.float32)

print(f"\nSanity check: {patches[0].name}")
print(f"  Shape        : {data.shape}  (expected (22, 257, 257))")
print(f"  Min/Max      : {data.min():.4f} / {data.max():.4f}")
print(f"  NaN count    : {np.isnan(data).sum()}")
print(f"  dNBR  ch20   : mean = {data[20].mean():.4f}")
print(f"  dNDVI ch21   : mean = {data[21].mean():.4f}")

assert data.shape[0] == 22, "Expected 22 channels"
assert REGION in patches[0].name, f"Region '{REGION}' not in filename"

# Check progress.json
import json
progress = json.loads((DATASET_ROOT / REGION / "progress.json").read_text())
print(f"\nprogress.json : {progress['completed']}/{progress['total_patches']} completed")
print(f"  errors      : {len(progress['errors'])}")

print("\nAll checks passed.")
