#!/usr/bin/env python3
"""
Regenerates comparison figures for section 10.2 of the report.
Uses the fixed make_comparison_figures (legend outside).

Usage:
    python scripts/regen_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from burnseg_xai.config.loader import load_config
from burnseg_xai.dataset import BurnedAreaDataset
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.evaluator import _prepare_batch
from burnseg_xai.visualization import make_comparison_figures


CONFIG    = Path(__file__).parent.parent / "configs" / "config.yaml"
BASE_DIR  = Path(__file__).parent.parent / "report_assets"

RUNS = [
    {
        "label":      "baseline",
        "checkpoint": Path("./outputs/checkpoints/baseline_seed43_2026-05-01/checkpoint_best.pt"),
        "out_dir":    BASE_DIR / "baseline" / "comparison",
    },
    {
        "label":      "rrr_cosine",
        "checkpoint": Path("./outputs/checkpoints/rrr_l0.1_cosine_seed43_2026-04-30/checkpoint_best.pt"),
        "out_dir":    BASE_DIR / "rrr_cosine" / "comparison",
    },
]

# Patches used in section 10.2
TARGET_PATCHES = {"karipuna_patch_00011", "karipuna_patch_00015", "karipuna_patch_00022"}


def find_pixel_threshold(model, ds, target_indices, device, dnbr_threshold=0.1):
    """Find pixel threshold maximising F1 on the target patches."""
    all_err, all_lbl = [], []
    model.eval()
    for idx in target_indices:
        raw = ds[idx]
        dnbr_raw = raw[:, :, :, 20].max(dim=0).values.numpy()
        mask = (dnbr_raw > dnbr_threshold)
        x, _ = _prepare_batch(raw.unsqueeze(0), device)
        with torch.no_grad():
            x_hat, _ = model(x)
            err = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))[0]
        all_err.append(err.cpu().numpy().flatten())
        all_lbl.append(mask.flatten())

    all_err = np.concatenate(all_err)
    all_lbl = np.concatenate(all_lbl)
    best_f1, best_thr = 0.0, float(np.percentile(all_err, 50))
    for thr in np.percentile(all_err, np.linspace(5, 95, 60)):
        pred = all_err >= thr
        tp = (pred & all_lbl).sum()
        fp = (pred & ~all_lbl).sum()
        fn = (~pred & all_lbl).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"  threshold -> {best_thr:.6f}  F1={best_f1:.3f}")
    return best_thr


def main():
    cfg    = load_config(str(CONFIG))
    device = "cpu"
    ds     = BurnedAreaDataset(cfg.dataset_root, cfg.temporal_length)

    # Build index of target patches
    target_indices = {
        Path(s).stem: i
        for i, s in enumerate(ds.samples)
        if Path(s).stem in TARGET_PATCHES
    }
    print(f"Found target patches: {list(target_indices.keys())}")

    for run in RUNS:
        print(f"\n=== {run['label']} ===")
        if not run["checkpoint"].exists():
            print(f"  SKIP: checkpoint not found: {run['checkpoint']}")
            continue

        model = Autoencoder(in_channels=cfg.in_channels).to(device)
        ckpt  = torch.load(run["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()

        indices = list(target_indices.values())
        threshold = find_pixel_threshold(model, ds, indices, device)

        # Build a minimal DataLoader for just those patches
        from torch.utils.data import DataLoader, Subset
        subset = Subset(ds, indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

        figs, names = make_comparison_figures(
            model, loader, device,
            pixel_threshold=threshold,
            n_samples=len(indices),
            dnbr_threshold=0.1,
        )

        run["out_dir"].mkdir(parents=True, exist_ok=True)
        for fig, name in zip(figs, names):
            out = run["out_dir"] / f"{name}.png"
            fig.savefig(str(out), dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
