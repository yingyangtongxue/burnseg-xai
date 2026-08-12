"""
CLI entry points for the acquisition pipeline.

    burnseg-download      : download + build 22-channel patches for one AOI
    burnseg-download-all  : download 22-channel patches for multiple AOIs from a JSON config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# burnseg-download  (single AOI)
# ---------------------------------------------------------------------------

def _download_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="burnseg-download",
        description=(
            "Download Landsat composites from GEE and build 22-channel patches "
            "ready for BurnedAreaDataset, no separate build step needed. "
            "Re-running the same command resumes from where it stopped."
        ),
    )
    p.add_argument("--aoi", required=True, metavar="KML",
                   help="Path to AOI KML file (geemap-compatible).")
    p.add_argument("--pre-date", required=True, metavar="YYYY-MM-DD",
                   help="Centre date of the pre-fire composite.")
    p.add_argument("--post-date", required=True, metavar="YYYY-MM-DD",
                   help="Centre date of the post-fire composite.")
    p.add_argument("--dataset-root", required=True, metavar="DIR",
                   help="Dataset root directory, e.g. ./data. "
                        "Output goes to {dataset-root}/{region}/patches/.")
    p.add_argument("--region", required=True, metavar="NAME",
                   help="Region name matching BurnedAreaDataset.KNOWN_REGIONS, "
                        "e.g. kayapo or parna_chapada_dos_guimaraes.")
    p.add_argument("--collection", default="LANDSAT/LC08/C02/T1_L2",
                   help="GEE ImageCollection ID (default: Landsat 8 C2 L2).")
    p.add_argument("--buffer-days", type=int, default=15, metavar="N",
                   help="Days ± around each date for the median composite (default: 15).")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Parallel download threads (default: 4).")
    p.add_argument("--retries", type=int, default=3, metavar="N",
                   help="Per-patch retry attempts on HTTP errors (default: 3).")
    p.add_argument("--timeout", type=int, default=120, metavar="SEC",
                   help="HTTP request timeout in seconds (default: 120).")
    p.add_argument("--key-file", default="auth/api_key.json", metavar="JSON",
                   help="GEE service-account JSON key (default: auth/api_key.json).")
    p.add_argument("--project", default=None, metavar="PROJECT_ID",
                   help="GCP project ID (required for OAuth auth only; ignored "
                        "when a service account key is used). "
                        "Can also be set via GEE_PROJECT env-var.")
    return p


def main_download(argv: list[str] | None = None) -> None:
    args = _download_parser().parse_args(argv)
    from .gee_download import download_aoi

    download_aoi(
        aoi_kml=args.aoi,
        collection=args.collection,
        pre_date=args.pre_date,
        post_date=args.post_date,
        dataset_root=Path(args.dataset_root),
        region=args.region,
        buffer_days=args.buffer_days,
        max_workers=args.workers,
        max_retries=args.retries,
        timeout=args.timeout,
        key_file=args.key_file,
        project=args.project,
    )


# ---------------------------------------------------------------------------
# burnseg-download-all  (multiple AOIs from JSON config)
# ---------------------------------------------------------------------------

def _download_all_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="burnseg-download-all",
        description=(
            "Download 22-channel patches for multiple AOIs defined in a JSON config "
            "file. Regions are processed sequentially; each resumes automatically "
            "if patches already exist on disk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSON config format (regions.json):
  [
    {"aoi": "aoi/kayapo.kml",    "region": "kayapo",
     "pre_date": "2024-08-17",   "post_date": "2024-10-18"},
    {"aoi": "aoi/karipuna.kml",  "region": "karipuna",
     "pre_date": "2024-08-01",   "post_date": "2024-10-31"},
    {"aoi": "aoi/yanomami.kml",  "region": "yanomami",
     "pre_date": "2024-02-29",   "post_date": "2024-03-31"},
    {"aoi": "aoi/chapada.kml",   "region": "parna_chapada_dos_guimaraes",
     "pre_date": "2019-07-17",   "post_date": "2019-08-18"}
  ]

Each entry may also include an optional "collection" key to override the
default GEE ImageCollection for that specific region.
""",
    )
    p.add_argument("--regions-json", required=True, metavar="JSON",
                   help="Path to JSON file listing regions to download.")
    p.add_argument("--dataset-root", required=True, metavar="DIR",
                   help="Dataset root directory; each region goes to "
                        "{dataset-root}/{region}/patches/.")
    p.add_argument("--collection", default="LANDSAT/LC08/C02/T1_L2",
                   help="Default GEE ImageCollection ID (default: Landsat 8 C2 L2). "
                        "Per-region 'collection' keys in the JSON take precedence.")
    p.add_argument("--buffer-days", type=int, default=15, metavar="N",
                   help="Days ± around each date for the median composite (default: 15).")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Parallel download threads per region (default: 4).")
    p.add_argument("--retries", type=int, default=3, metavar="N",
                   help="Per-patch retry attempts on HTTP errors (default: 3).")
    p.add_argument("--timeout", type=int, default=120, metavar="SEC",
                   help="HTTP request timeout in seconds (default: 120).")
    p.add_argument("--key-file", default="auth/api_key.json", metavar="JSON",
                   help="GEE service-account JSON key (default: auth/api_key.json).")
    p.add_argument("--project", default=None, metavar="PROJECT_ID",
                   help="GCP project ID for OAuth auth. Can also be set via GEE_PROJECT.")
    return p


def main_download_all(argv: list[str] | None = None) -> None:
    args = _download_all_parser().parse_args(argv)
    from .gee_download import download_regions

    regions_path = Path(args.regions_json)
    if not regions_path.exists():
        print(f"Error: regions JSON not found: {regions_path}", file=sys.stderr)
        sys.exit(1)

    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    if not isinstance(regions, list) or not regions:
        print("Error: regions JSON must be a non-empty list.", file=sys.stderr)
        sys.exit(1)

    download_regions(
        regions=regions,
        dataset_root=Path(args.dataset_root),
        collection=args.collection,
        buffer_days=args.buffer_days,
        max_workers=args.workers,
        max_retries=args.retries,
        timeout=args.timeout,
        key_file=args.key_file,
        project=args.project,
    )


# ---------------------------------------------------------------------------
# Allow running as `python -m burnseg_xai.acquisition.cli`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        main_download(sys.argv[2:])
    elif cmd == "download-all":
        main_download_all(sys.argv[2:])
    else:
        print("Usage: python -m burnseg_xai.acquisition.cli {download|download-all} [options]")
        sys.exit(1)
