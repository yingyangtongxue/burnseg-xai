"""
loro_fold_viz.py: generate saliency figures for a LORO fold's test patches.

Loads the hold-out region checkpoint and generates saliency maps for the top N
test patches ranked by dNBR signal. Useful for qualitative LORO analysis.

Usage:
    python scripts/loro_fold_viz.py \
        --checkpoint <path/checkpoint_best.pt> \
        --region yanomami \
        --config configs/config_runpod_eval.yaml \
        --n_patches 20 \
        --output_dir <out_dir>
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from burnseg_xai.config.loader import load_config
from burnseg_xai.dataset import BurnedAreaDataset
from burnseg_xai.evaluator import _prepare_batch
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.split import save_master_split
from burnseg_xai.utils.seed import set_seed
from burnseg_xai.visualization import _compute_saliency, _get_patch_name
from torch.utils.data import Subset


DNBR_CHANNEL = 20   # raw channel index before model normalization


def _dnbr_max_for_patch(dataset, idx: int) -> float:
    """Quick scan: get dNBR max for a patch without GPU."""
    raw = dataset[idx]          # (T, H, W, C) or similar tensor
    x = raw[0] if isinstance(raw, (list, tuple)) else raw
    # x shape: (T, H, W, C=22)
    dnbr = x[..., DNBR_CHANNEL].numpy()
    dnbr_relu = np.clip(dnbr, 0, None)
    return float(dnbr_relu.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--region",       required=True, help="Test region (e.g. yanomami)")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--n_patches",    type=int, default=20)
    parser.add_argument("--scan_top_k",   type=int, default=60,
                        help="After fast dNBR scan, run saliency only on top K candidates.")
    parser.add_argument("--output_dir",   default="loro_viz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = cfg.device

    dataset = BurnedAreaDataset(cfg.dataset_root, cfg.temporal_length, region=args.region)
    print(f"Region '{args.region}' patches: {len(dataset)}")

    # --- Pass 1: fast dNBR scan (CPU, no GPU needed) ---
    print("Pass 1: scanning dNBR for all patches (CPU only)...")
    dnbr_scores = []
    for i in range(len(dataset)):
        score = _dnbr_max_for_patch(dataset, i)
        dnbr_scores.append((score, i))
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(dataset)}] scanned...")

    dnbr_scores.sort(reverse=True)
    top_k_idx = [idx for _, idx in dnbr_scores[:args.scan_top_k]]
    print(f"Top {args.scan_top_k} by dNBR: indices {top_k_idx[:5]}... (max dNBR: {dnbr_scores[0][0]:.3f})")

    # --- Pass 2: saliency on top-K candidates ---
    print(f"Pass 2: saliency computation on {args.scan_top_k} candidates (GPU)...")
    model = Autoencoder(in_channels=cfg.in_channels).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from torch.utils.data import Subset
    sub_dataset = Subset(dataset, top_k_idx)
    loader = DataLoader(sub_dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for batch_idx, batch in enumerate(loader):
        global_idx = top_k_idx[batch_idx]
        x, prior = _prepare_batch(batch, device)
        patch_name = _get_patch_name(dataset, global_idx)

        sal_np, err_np = _compute_saliency(model, x)
        prior_np = prior.squeeze().cpu().numpy()

        sal_flat   = sal_np.flatten().astype(float)
        prior_flat = prior_np.flatten().astype(float)
        denom = np.linalg.norm(sal_flat) * np.linalg.norm(prior_flat)
        cos = float(np.dot(sal_flat, prior_flat) / denom) if denom > 1e-8 else 0.0

        dnbr_max   = float(prior_np.max())
        recon_mean = float(err_np.mean())

        results.append({
            "idx": global_idx, "patch": patch_name, "cos": cos,
            "dnbr_max": dnbr_max, "recon_mean": recon_mean,
            "prior_np": prior_np, "sal_np": sal_np, "err_np": err_np,
        })
        print(f"  [{batch_idx+1}/{args.scan_top_k}] {patch_name}  cos={cos:.3f}  dnbr={dnbr_max:.3f}")

    # Rank: burned patches (dnbr_max > 0.1) first, then by cosine
    results.sort(key=lambda r: (r["dnbr_max"] > 0.1, r["cos"]), reverse=True)

    top = results[:args.n_patches]

    print(f"\nTop {len(top)} patches by cosine (burned-proxy first):")
    for rank, r in enumerate(top, 1):
        print(f"  #{rank}: {r['patch']}  cos={r['cos']:.3f}  dnbr_max={r['dnbr_max']:.3f}")

    for rank, r in enumerate(top, 1):
        prior_np = r["prior_np"]
        sal_np   = r["sal_np"]
        err_np   = r["err_np"]
        pname    = r["patch"]
        cos      = r["cos"]

        vmin_err = float(np.percentile(err_np, 1))
        vmax_err = float(np.percentile(err_np, 99))
        overlay  = sal_np * prior_np

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(prior_np, cmap="Reds",    vmin=0, vmax=1)
        axes[0].set_title("dNBR prior\n(normalized)",              fontsize=9)
        axes[1].imshow(sal_np,   cmap="hot",     vmin=0, vmax=1)
        axes[1].set_title("Saliency |grad L/grad x|\n(normalized)", fontsize=9)
        axes[2].imshow(overlay,  cmap="YlOrRd",  vmin=0, vmax=max(overlay.max(), 1e-6))
        axes[2].set_title(f"Overlay saliency x dNBR\ncos={cos:.3f}", fontsize=9)
        axes[3].imshow(err_np,   cmap="viridis", vmin=vmin_err, vmax=vmax_err)
        axes[3].set_title("Recon error\n(pixel)",                  fontsize=9)
        for ax in axes:
            ax.axis("off")

        best_flag = "  BEST" if rank == 1 else ""
        fig.suptitle(
            f"LORO leave-{args.region} | Rank #{rank}{best_flag}: {pname}",
            fontsize=10, fontweight="bold" if rank == 1 else "normal",
        )
        fig.tight_layout()

        fname = f"rank{rank:02d}_{pname}.png"
        fig.savefig(os.path.join(args.output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    ranking = [
        {"rank": i + 1, "patch": top[i]["patch"], "cos": top[i]["cos"],
         "dnbr_max": top[i]["dnbr_max"], "recon_mean": top[i]["recon_mean"]}
        for i in range(len(top))
    ]
    with open(os.path.join(args.output_dir, "ranking.json"), "w") as f:
        json.dump(ranking, f, indent=2)

    best = top[0]
    print(f"\nBest patch: {best['patch']}  cos={best['cos']:.3f}  dnbr_max={best['dnbr_max']:.3f}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
