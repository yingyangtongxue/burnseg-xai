import torch
from tqdm import tqdm

from burnseg_xai.config.schema import ProjectConfig


def run_sanity_checks(dataset, cfg: ProjectConfig, n_samples: int = 10) -> None:
    """
    Assert dataset integrity and config/data consistency before training.
    Raises ValueError on the first failed check.
    """
    n = min(n_samples, len(dataset))

    bar = tqdm(range(n), desc="Sanity checks", leave=False)
    for i in bar:
        x = dataset[i]
        bar.set_postfix(shape=str(tuple(x.shape)))

        if torch.isnan(x).any():
            raise ValueError(f"NaN detected in sample {i}")
        if torch.isinf(x).any():
            raise ValueError(f"Inf detected in sample {i}")
        if x.shape[0] == 0:
            raise ValueError(f"Empty temporal sequence at sample {i}")

    # Dataset returns 22-channel raw tensors; model uses 21 (channel 20 excluded)
    actual_ch = dataset[0].shape[-1]
    if actual_ch != 22:
        raise ValueError(
            f"Expected 22 channels in dataset, got {actual_ch}. "
            "Verify dataset_root points to the correct .tif files."
        )

    if cfg.in_channels != 21:
        raise ValueError(
            f"cfg.in_channels must be 21, got {cfg.in_channels}"
        )

    if cfg.lambda_rrr < 0:
        raise ValueError(f"cfg.lambda_rrr must be >= 0, got {cfg.lambda_rrr}")

    print(f"Sanity checks OK: {n} samples, shape {tuple(dataset[0].shape)}")
