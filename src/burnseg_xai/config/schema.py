from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProjectConfig:

    # --- Paths ---
    dataset_root: str = "./data"
    kml_dir: str = "./data/aoi"
    output_root: str = "./outputs"

    # --- AOIs ---
    regions: List[str] = field(default_factory=lambda: [
        "karipuna",
        "kayapo",
        "parna_chapada_dos_guimaraes",
        "yanomami",
    ])
    kml_files: Dict[str, str] = field(default_factory=lambda: {
        "kayapo": "kayapo_focos_qmd_inpe_2024-08-01_2024-11-03_01.248323.kml",
        "parna_chapada_dos_guimaraes": "focos_chapada.kml",
        "yanomami": "yanomami_focos_2024-02-29_2024-03-31.kml",
        "karipuna": "karipuna_focos_qmd_inpe_2024-08-01_2024-10-31_29.304993.kml",
    })

    # --- Reproducibility ---
    seed: int = 43

    # --- Training ---
    batch_size: int = 2           # memory constraint: batch_size <= 2
    epochs: int = 300
    lr: float = 1e-4
    device: str = "cuda"
    early_stopping_patience: int = 20

    # --- Data ---
    temporal_length: int = 1
    num_workers: int = 0          # 0 = main process only (avoids Windows multiprocessing bugs)
    max_samples: Optional[int] = None  # cap dataset size; None = full dataset

    # --- Model ---
    in_channels: int = 21         # channels 0-19 + 21 (dNDVI); channel 20 (dNBR) excluded

    # --- RRR regularization ---
    lambda_rrr: float = 0.0
    rrr_distance_metric: str = "mse"
    xai_terms: tuple = ("grad", "gradcam", "attn")

    # --- Checkpointing ---
    checkpoint_dir: str = "./outputs/checkpoints"

    # --- MLflow ---
    mlflow_experiment: str = "burned_area"
    mlflow_tracking_uri: str = "./outputs/mlruns"
