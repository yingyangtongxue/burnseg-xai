"""
integrated_gradients.py: post-hoc attribution visualization using Integrated Gradients.

Generates per-channel attribution figures for post-hoc analysis.
NOT used during training: evaluation/visualization only.

Usage:
    python scripts/integrated_gradients.py \
        --checkpoint ./outputs/checkpoints/<run>/checkpoint_best.pt \
        --config configs/config.yaml \
        --n_samples 8 \
        --output_dir ./outputs/ig_figures/

Requires: captum (pip install captum)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from burnseg_xai.config.loader import load_config
from burnseg_xai.dataset import BurnedAreaDataset
from burnseg_xai.evaluator import _prepare_batch
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.split import save_master_split
from burnseg_xai.utils.seed import set_seed

CHANNEL_NAMES = [
    "B2_pre",  "B3_pre",  "B4_pre",  "B5_pre",  "B6_pre",  "B7_pre",
    "NDVI_pre","NBR_pre", "NDMI_pre","BAI_pre",
    "B2_post", "B3_post", "B4_post", "B5_post", "B6_post", "B7_post",
    "NDVI_post","NBR_post","NDMI_post","BAI_post",
    "dNDVI",
]


def _forward_recon_loss(model, x):
    x_hat, _ = model(x)
    return F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2, 3, 4), keepdim=False)


def compute_ig(model, x, baseline, n_steps=50):
    """
    Integrated Gradients attribution for reconstruction loss.

    Parameters
    ----------
    x        : (1, C, T, H, W) input tensor, requires_grad=False
    baseline : (1, C, T, H, W) reference input (zeros)
    n_steps  : number of interpolation steps

    Returns
    -------
    ig : (C,) mean absolute attribution per channel, normalized to [0, 1]
    """
    model.eval()
    alphas = torch.linspace(0, 1, n_steps, device=x.device)
    grads = []

    for alpha in alphas:
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        loss = _forward_recon_loss(model, interp).mean()
        grad = torch.autograd.grad(loss, interp, create_graph=False)[0]
        grads.append(grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)               # (1, C, T, H, W)
    ig_raw = ((x - baseline) * avg_grads).abs()              # element-wise product
    ig_per_channel = ig_raw.mean(dim=(0, 2, 3, 4)).cpu()     # (C,)

    ig_norm = ig_per_channel / ig_per_channel.sum().clamp(min=1e-8)
    return ig_norm.numpy()


def make_ig_figure(ig_scores, title="Integrated Gradients: channel importance"):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#d62728" if "post" in n or n == "dNDVI" else "#1f77b4"
              for n in CHANNEL_NAMES]
    bars = ax.bar(range(len(CHANNEL_NAMES)), ig_scores, color=colors)
    ax.set_xticks(range(len(CHANNEL_NAMES)))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized attribution")
    ax.set_title(title)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#d62728"),
            plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        ],
        labels=["post-fire / delta", "pre-fire"],
        loc="upper right",
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--n_samples",  type=int, default=8)
    parser.add_argument("--n_steps",    type=int, default=50)
    parser.add_argument("--output_dir", default="ig_figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = cfg.device

    dataset = BurnedAreaDataset(cfg.dataset_root, cfg.temporal_length)
    split_path = os.path.join(cfg.output_root, "split_master.json")
    _, _, test_idx = save_master_split(dataset, split_path, seed=cfg.seed)

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = Autoencoder(in_channels=cfg.in_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    all_ig = []
    count = 0
    for batch in test_loader:
        if count >= args.n_samples:
            break
        x, prior = _prepare_batch(batch, device)
        if prior.amax() < 0.1:
            continue

        baseline = torch.zeros_like(x)
        ig = compute_ig(model, x, baseline, n_steps=args.n_steps)
        all_ig.append(ig)

        fig = make_ig_figure(ig, title=f"IG attributions: sample {count + 1}")
        path = os.path.join(args.output_dir, f"ig_sample_{count + 1:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")
        count += 1

    if all_ig:
        mean_ig = np.mean(all_ig, axis=0)
        fig = make_ig_figure(mean_ig, title=f"IG attributions: mean over {count} burned patches")
        path = os.path.join(args.output_dir, "ig_mean.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  mean figure saved -> {path}")
        print("\nTop-5 channels by mean attribution:")
        top5 = np.argsort(mean_ig)[::-1][:5]
        for i in top5:
            print(f"  {CHANNEL_NAMES[i]:15s} {mean_ig[i]:.4f}")


if __name__ == "__main__":
    main()
