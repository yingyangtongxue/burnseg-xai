"""
Acquisition sub-package: GEE download pipeline (server-side 22-channel patches).

    burnseg-download      → download one AOI from GEE, produces dataset-ready patches
    burnseg-download-all  → download multiple AOIs from a JSON config
"""

from .gee_download import download_aoi, download_regions

__all__ = [
    "download_aoi",
    "download_regions",
]
