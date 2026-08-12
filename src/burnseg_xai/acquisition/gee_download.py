"""
Google Earth Engine download pipeline. Delivers dataset-ready patches.

All 22 channels are computed server-side on GEE and downloaded as a single
GeoTIFF per patch. No local build step is needed.

Output layout
-------------
    {dataset_root}/{region}/
        patches/           {region}_patch_00000.tif ... (22 ch, 257x257, float32)
        metadata/          dataset_metadata.json
        progress.json      checkpoint, re-run the same command to resume

Channel layout (22 channels, matches BurnedAreaDataset)
--------------------------------------------------------
     0- 9  pre-fire  [SR_B2, B3, B4, B5, B6, B7, NDVI, NBR, NDMI, BAI]
    10-19  post-fire [same order]
    20     dNBR  = NBR_pre  - NBR_post   (RRR prior, not model input)
    21     dNDVI = NDVI_pre - NDVI_post  (model input)

Authentication (in order of priority)
--------------------------------------
    1. Service account: key_file with ``client_email``
       (or SERVICE_ACCOUNT_EMAIL env-var).
    2. OAuth user credentials (~/.config/earthengine/credentials): pass
       --project <gcp-project-id> or set GEE_PROJECT env-var.

Single-region usage (CLI: burnseg-download)
-------------------------------------------
    burnseg-download \\
        --aoi          aoi/kayapo.kml \\
        --pre-date     2024-08-17 \\
        --post-date    2024-10-18 \\
        --dataset-root ./data \\
        --region       kayapo \\
        --key-file     auth/api_key.json

Multi-region usage (CLI: burnseg-download-all)
----------------------------------------------
    burnseg-download-all \\
        --regions-json regions.json \\
        --dataset-root ./data \\
        --key-file     auth/api_key.json

    regions.json format:
    [
      {"aoi": "aoi/kayapo.kml",    "region": "kayapo",
       "pre_date": "2024-08-17",   "post_date": "2024-10-18"},
      {"aoi": "aoi/karipuna.kml",  "region": "karipuna",
       "pre_date": "2024-08-01",   "post_date": "2024-10-31"},
      ...
    ]

Resume: patches already on disk are skipped automatically (per region).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
SCALE = 30
PATCH_SIZE_PX = 256
_PATCH_SIZE_M = PATCH_SIZE_PX * SCALE

# Band names in the final 22-channel GeoTIFF (also written to metadata)
_CHANNEL_FEATURES = [
    "pre_SR_B2", "pre_SR_B3", "pre_SR_B4", "pre_SR_B5", "pre_SR_B6", "pre_SR_B7",
    "pre_NDVI",  "pre_NBR",   "pre_NDMI",  "pre_BAI",
    "post_SR_B2","post_SR_B3","post_SR_B4","post_SR_B5","post_SR_B6","post_SR_B7",
    "post_NDVI", "post_NBR",  "post_NDMI", "post_BAI",
    "dNBR",
    "dNDVI",
]


# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------

def _init_ee(key_file: str = "auth/api_key.json", project: str | None = None) -> None:
    import json as _json

    import ee
    from dotenv import load_dotenv

    load_dotenv()
    key_path = Path(key_file)

    service_account = os.environ.get("SERVICE_ACCOUNT_EMAIL", "")
    if not service_account and key_path.exists():
        try:
            service_account = _json.loads(key_path.read_text()).get("client_email", "")
        except Exception:
            pass

    if service_account and key_path.exists():
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials, project=project)
        print(f"GEE initialised, service account: {service_account}")
    else:
        if project is None:
            raise ValueError(
                "GEE requires a Cloud Project when using OAuth credentials.\n"
                "Pass --project <gcp-project-id> or set GEE_PROJECT env-var.\n"
                "Register at: https://code.earthengine.google.com/register"
            )
        ee.Initialize(project=project)
        print(f"GEE initialised, OAuth credentials, project: {project}")


# ---------------------------------------------------------------------------
# GEE image helpers
# ---------------------------------------------------------------------------

def _scale_factors(image):
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical, None, True).addBands(thermal, None, True)


def _apply_cloud_mask(image):
    qa    = image.select("QA_PIXEL")
    clear = qa.bitwiseAnd(1 << 3).eq(0)           # bit 3: cloud
    clear = clear.And(qa.bitwiseAnd(1 << 4).eq(0)) # bit 4: cloud shadow
    return image.updateMask(clear)


def get_composite(collection: str, aoi, date_str: str, buffer_days: int = 15):
    """Median composite of *collection* clipped to *aoi* around *date_str* ± buffer."""
    import ee
    start = ee.Date(date_str).advance(-buffer_days, "day")
    end   = ee.Date(date_str).advance( buffer_days, "day")
    return (
        ee.ImageCollection(collection)
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(_scale_factors)
        .map(_apply_cloud_mask)
        .median()
        .select(BANDS)
        .clip(aoi)
    )


def _add_indices(img, prefix: str):
    """
    Compute NDVI, NBR, NDMI, BAI server-side and return a 10-band image.
    Band order: SR_B2, B3, B4, B5, B6, B7, NDVI, NBR, NDMI, BAI
    All bands renamed with *prefix* to avoid name collisions when stacking.
    """
    import ee

    nir   = img.select("SR_B5")
    red   = img.select("SR_B4")
    swir1 = img.select("SR_B6")
    swir2 = img.select("SR_B7")
    eps   = ee.Image(1e-6)

    ndvi = nir.subtract(red).divide(nir.add(red).add(eps)).rename("NDVI")
    nbr  = nir.subtract(swir2).divide(nir.add(swir2).add(eps)).rename("NBR")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1).add(eps)).rename("NDMI")
    bai  = ee.Image(1.0).divide(
        ee.Image(0.1).subtract(red).pow(2)
        .add(ee.Image(0.06).subtract(nir).pow(2))
        .add(eps)
    ).rename("BAI")

    ten_band = img.addBands([ndvi, nbr, ndmi, bai])  # 10 bands

    new_names = [f"{prefix}{b}" for b in
                 ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                  "NDVI", "NBR", "NDMI", "BAI"]]
    return ten_band.rename(new_names)


def build_22ch_image(pre_img, post_img):
    """
    Build a 22-band GEE image entirely server-side.
    Band order matches _CHANNEL_FEATURES / BurnedAreaDataset.
    """
    pre  = _add_indices(pre_img,  prefix="pre_")   # bands pre_SR_B2 … pre_BAI
    post = _add_indices(post_img, prefix="post_")  # bands post_SR_B2 … post_BAI

    dnbr  = pre.select("pre_NBR").subtract(post.select("post_NBR")).rename("dNBR")
    dndvi = pre.select("pre_NDVI").subtract(post.select("post_NDVI")).rename("dNDVI")

    return pre.addBands(post).addBands(dnbr).addBands(dndvi)  # 22 bands


def subdivide_aoi(aoi) -> list:
    """Tile the bounding box of *aoi* into PATCH_SIZE_PX × PATCH_SIZE_PX cells."""
    import ee
    bounds = aoi.bounds().getInfo()["coordinates"][0]
    xmin = min(p[0] for p in bounds)
    ymin = min(p[1] for p in bounds)
    xmax = max(p[0] for p in bounds)
    ymax = max(p[1] for p in bounds)

    step = _PATCH_SIZE_M / 111_320
    patches = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            patches.append(
                ee.Geometry.Rectangle(
                    [x, y, min(x + step, xmax), min(y + step, ymax)]
                )
            )
            y += step
        x += step
    return patches


# ---------------------------------------------------------------------------
# Per-patch download
# ---------------------------------------------------------------------------

def _is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _download_patch(
    patch_img,
    geom,
    out_path: Path,
    timeout: int,
) -> None:
    """Download a 22-band GeoTIFF for one patch. Atomic: writes via .tmp file."""
    tmp = out_path.with_suffix(".tmp")
    url = patch_img.clip(geom).getDownloadURL({
        "scale": SCALE,
        "region": geom,
        "format": "GeoTIFF",
        "bands": _CHANNEL_FEATURES,
    })
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out_path)


def _process_patch(
    args: tuple,
    patches_dir: Path,
    region: str,
    patch_img,
    max_retries: int,
    retry_delay: int,
    timeout: int,
) -> tuple[str, str]:
    """Download one 22-channel patch with retries. Returns (patch_id, status)."""
    i, geom = args
    pid      = f"patch_{i:05d}"
    out_path = patches_dir / f"{region}_{pid}.tif"

    if _is_valid(out_path):
        return pid, "skipped"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            _download_patch(patch_img, geom, out_path, timeout)
            return pid, "ok"
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    return pid, f"error: {last_err}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def download_aoi(
    aoi_kml: str,
    collection: str,
    pre_date: str,
    post_date: str,
    dataset_root: Path,
    region: str,
    buffer_days: int = 15,
    max_workers: int = 4,
    max_retries: int = 3,
    retry_delay: int = 10,
    timeout: int = 120,
    key_file: str = "auth/api_key.json",
    project: str | None = None,
) -> None:
    """
    Download dataset-ready 22-channel patches for one AOI.

    All indices are computed server-side on GEE; only the final patches are
    transferred. Re-running resumes automatically: patches already on disk
    are skipped.

    Output: {dataset_root}/{region}/patches/{region}_patch_{i:05d}.tif
    """
    import geemap
    from tqdm import tqdm

    project = project or os.environ.get("GEE_PROJECT")
    _init_ee(key_file, project=project)
    aoi = geemap.kml_to_ee(aoi_kml)

    # Directories
    region_dir    = dataset_root / region
    patches_dir   = region_dir / "patches"
    meta_dir      = region_dir / "metadata"
    progress_path = region_dir / "progress.json"
    patches_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    # Build 22-band server-side image and patch grid
    print("Building server-side 22-channel image and patch grid…")
    pre_img   = get_composite(collection, aoi, pre_date,  buffer_days)
    post_img  = get_composite(collection, aoi, post_date, buffer_days)
    patch_img = build_22ch_image(pre_img, post_img)
    patches   = subdivide_aoi(aoi)
    total     = len(patches)

    done_before = sum(
        1 for i in range(total)
        if _is_valid(patches_dir / f"{region}_patch_{i:05d}.tif")
    )
    print(f"Total patches : {total}")
    print(f"Already done  : {done_before}  (will be skipped)")
    print(f"Remaining     : {total - done_before}\n")

    # Checkpoint
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {
            "region": region, "collection": collection,
            "pre_date": pre_date, "post_date": post_date,
            "total_patches": total, "completed": done_before,
            "errors": [], "started_at": datetime.now().isoformat(), "updated_at": "",
        }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    errors: list[str] = []
    work_args = list(enumerate(patches))

    with tqdm(
        total=total, initial=done_before,
        desc=f"[{region}]", unit="patch", dynamic_ncols=True,
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_patch,
                    args, patches_dir, region,
                    patch_img, max_retries, retry_delay, timeout,
                ): args[0]
                for args in work_args
            }
            try:
                for future in as_completed(futures):
                    pid, status = future.result()
                    if status == "skipped":
                        pass
                    elif status.startswith("error"):
                        errors.append(f"{pid}: {status}")
                        pbar.set_postfix_str(f"{pid} FAIL", refresh=False)
                        pbar.update(1)
                    else:
                        pbar.set_postfix_str(f"{pid} ok", refresh=False)
                        pbar.update(1)
            except KeyboardInterrupt:
                print("\nInterrupted, cancelling pending workers...")
                pool.shutdown(wait=False, cancel_futures=True)

    completed = sum(
        1 for i in range(total)
        if _is_valid(patches_dir / f"{region}_patch_{i:05d}.tif")
    )

    # Metadata
    meta = {
        "region": region, "collection": collection,
        "pre_date": pre_date, "post_date": post_date,
        "buffer_days": buffer_days, "resolution_m": SCALE,
        "patch_size_px": PATCH_SIZE_PX, "n_channels": 22,
        "features": _CHANNEL_FEATURES,
    }
    (meta_dir / "dataset_metadata.json").write_text(
        json.dumps(meta, indent=4), encoding="utf-8"
    )

    # Update checkpoint
    progress.update({
        "completed": completed, "errors": errors,
        "updated_at": datetime.now().isoformat(),
    })
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    if errors:
        print(f"\n{len(errors)} error(s), see {progress_path}")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print(f"\nComplete: {completed}/{total} patches in {patches_dir}")


# ---------------------------------------------------------------------------
# Multi-region entry point
# ---------------------------------------------------------------------------

def download_regions(
    regions: list[dict],
    dataset_root: Path,
    collection: str = "LANDSAT/LC08/C02/T1_L2",
    buffer_days: int = 15,
    max_workers: int = 4,
    max_retries: int = 3,
    retry_delay: int = 10,
    timeout: int = 120,
    key_file: str = "auth/api_key.json",
    project: str | None = None,
) -> None:
    """
    Download dataset-ready 22-channel patches for multiple AOIs sequentially.

    Each entry in *regions* must have the keys:
        aoi       - path to the KML file
        region    - region name matching BurnedAreaDataset.KNOWN_REGIONS
        pre_date  - YYYY-MM-DD centre date of the pre-fire composite
        post_date - YYYY-MM-DD centre date of the post-fire composite

    Optional per-entry override: ``collection`` (overrides the top-level default).

    Example::

        download_regions(
            regions=[
                {"aoi": "aoi/kayapo.kml", "region": "kayapo",
                 "pre_date": "2024-08-17", "post_date": "2024-10-18"},
                {"aoi": "aoi/karipuna.kml", "region": "karipuna",
                 "pre_date": "2024-08-01", "post_date": "2024-10-31"},
            ],
            dataset_root=Path("./data"),
            key_file="auth/api_key.json",
        )
    """
    n = len(regions)
    for idx, entry in enumerate(regions, 1):
        region = entry["region"]
        print(f"\n{'=' * 60}")
        print(f"Region {idx}/{n}: {region}")
        print(f"{'=' * 60}")
        download_aoi(
            aoi_kml=entry["aoi"],
            collection=entry.get("collection", collection),
            pre_date=entry["pre_date"],
            post_date=entry["post_date"],
            dataset_root=dataset_root,
            region=region,
            buffer_days=buffer_days,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            key_file=key_file,
            project=project,
        )
