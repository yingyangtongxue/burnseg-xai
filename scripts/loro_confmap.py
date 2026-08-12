#!/usr/bin/env python3
"""
Generates pixel-level confusion map figure for a single LORO patch.

5 panels (same style as make_comparison_figures in visualization.py):
  1. dNBR raw (continuous, RdYlGn_r)
  2. dNBR binary mask (proxy GT, dNBR > threshold)
  3. Reconstruction error (inferno)
  4. Binary burn prediction (recon > pixel_threshold)
  5. Confusion map (TP=green, FP=yellow, FN=red, TN=gray) + % per class

Threshold is computed by scanning percentiles on the test-region patches
and maximising pixel F1: consistent with the approach used in evaluator.py.

Usage:
    python scripts/loro_confmap.py \
        --config configs/config_runpod_eval.yaml \
        --checkpoint /workspace/experimento_queimadas/checkpoints/loro_yanomami_l0.1_mse_seed43_2026-05-05/checkpoint_best.pt \
        --patch_name yanomami_patch_00797 \
        --region yanomami \
        --fold_label "LORO leave-yanomami" \
        --cos 0.5 \
        --output_dir /workspace/experimento_queimadas/loro_confmaps/
"""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from burnseg_xai.config.loader import load_config
from burnseg_xai.dataset import BurnedAreaDataset
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.evaluator import _prepare_batch


# colour constants (same as visualization.py make_comparison_figures)
COLOR_TP = (0.10, 0.75, 0.10)   # green
COLOR_FP = (1.00, 0.85, 0.00)   # yellow
COLOR_FN = (0.95, 0.10, 0.10)   # red
COLOR_TN = (0.85, 0.85, 0.85)   # light gray


def find_optimal_threshold(model, patches, device, dnbr_threshold):
    """Scan test-region patches to find pixel-threshold that maximises F1."""
    all_err, all_lbl = [], []
    model.eval()

    for raw in patches:
        raw_t = raw.unsqueeze(0)
        # dNBR: channel 20, shape (T,H,W,C) → max over T → (H,W)
        dnbr_raw = raw[:, :, :, 20].max(dim=0).values.numpy()
        mask = (dnbr_raw > dnbr_threshold)

        x, _ = _prepare_batch(raw_t, device)
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

    print(f"  threshold scan -> best_thr={best_thr:.6f}  best_F1={best_f1:.3f}")
    return best_thr


def make_confmap_figure(model, raw, patch_name, threshold, device,
                        dnbr_threshold, fold_label, cos):
    """Return a 5-panel matplotlib Figure."""
    # dNBR ground-truth proxy
    dnbr_raw = raw[:, :, :, 20].max(dim=0).values.numpy()   # (H, W)
    mask_np  = (dnbr_raw > dnbr_threshold).astype(float)

    # Forward pass for error + saliency
    x, _ = _prepare_batch(raw.unsqueeze(0), device)
    x = x.requires_grad_(True)
    x_hat, _ = model(x)
    loss = F.mse_loss(x_hat, x)
    grads = torch.autograd.grad(loss, x, create_graph=False)[0]

    sal_np  = grads.abs().mean(dim=(1, 2))[0].detach().cpu().numpy()
    err_np  = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))[0].detach().cpu().numpy()

    pred_np = (err_np >= threshold).astype(float)

    H, W = dnbr_raw.shape
    n_total = H * W

    tp = (mask_np == 1) & (pred_np == 1)
    fp = (mask_np == 0) & (pred_np == 1)
    fn = (mask_np == 1) & (pred_np == 0)
    tn = (mask_np == 0) & (pred_np == 0)

    # Confusion RGBA
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[tn] = [*COLOR_TN, 0.50]
    rgba[tp] = [*COLOR_TP, 0.90]
    rgba[fp] = [*COLOR_FP, 0.90]
    rgba[fn] = [*COLOR_FN, 0.90]

    f1_val    = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8)
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    recall    = tp.sum() / (tp.sum() + fn.sum() + 1e-8)

    fig = plt.figure(figsize=(21, 4.5))
    gs  = gridspec.GridSpec(1, 5, wspace=0.08)

    # 1: dNBR raw
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(dnbr_raw, cmap="RdYlGn_r", vmin=-0.5, vmax=1.0)
    ax0.set_title("dNBR\n(raw, contínuo)", fontsize=8)
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 2: dNBR binary mask
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
    ax1.set_title(f"Mascara dNBR\n(dNBR > {dnbr_threshold}, proxy GT)", fontsize=8)
    ax1.axis("off")

    # 3: Reconstruction error
    ax2 = fig.add_subplot(gs[2])
    err_vmin = float(np.percentile(err_np, 1))
    err_vmax = float(np.percentile(err_np, 99)) + 1e-8
    im2 = ax2.imshow(err_np, cmap="inferno", vmin=err_vmin, vmax=err_vmax)
    ax2.set_title("Erro de reconstrução\n(saída do modelo)", fontsize=8)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 4: Binary prediction
    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(pred_np, cmap="Reds", vmin=0, vmax=1)
    ax3.set_title("Predicao binaria\n(erro >= threshold)", fontsize=8)
    ax3.axis("off")

    # 5: Confusion map (no legend inside: placed below figure)
    ax4 = fig.add_subplot(gs[4])
    ax4.imshow(rgba)
    ax4.set_title(
        f"Mapa de Confusao\nF1={f1_val:.3f}  P={precision:.3f}  R={recall:.3f}",
        fontsize=8,
    )
    ax4.axis("off")

    # Legend below the entire figure, outside all axes
    legend_elements = [
        Patch(facecolor=COLOR_TP,
              label=f"TP  {tp.sum()}  ({100*tp.sum()/n_total:.1f}%)"),
        Patch(facecolor=COLOR_FP,
              label=f"FP  {fp.sum()}  ({100*fp.sum()/n_total:.1f}%)"),
        Patch(facecolor=COLOR_FN,
              label=f"FN  {fn.sum()}  ({100*fn.sum()/n_total:.1f}%)"),
        Patch(facecolor=COLOR_TN,
              label=f"TN  {tn.sum()}  ({100*tn.sum()/n_total:.1f}%)"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center",
               bbox_to_anchor=(0.88, -0.12),   # centred under panel 5
               ncol=2, fontsize=7.5, framealpha=0.92,
               title="Mapa de confusao", title_fontsize=7)

    cos_str = f"  |  cos={cos:.3f}" if cos is not None else ""
    fig.suptitle(
        f"{fold_label}  |  {patch_name}{cos_str}",
        fontsize=9, y=1.01,
    )
    fig.subplots_adjust(bottom=0.18)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True)
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--patch_name",    required=True,
                        help="e.g. yanomami_patch_00797")
    parser.add_argument("--region",        required=True,
                        help="held-out region prefix, e.g. yanomami")
    parser.add_argument("--fold_label",    default="LORO fold")
    parser.add_argument("--cos",           type=float, default=None)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--dnbr_threshold",type=float, default=0.1)
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = args.device

    ds = BurnedAreaDataset(cfg.dataset_root, cfg.temporal_length)

    # Identify target patch index
    target_idx = next(
        (i for i, s in enumerate(ds.samples)
         if Path(s).stem == args.patch_name),
        None,
    )
    if target_idx is None:
        print(f"ERROR: patch '{args.patch_name}' not found in dataset")
        sys.exit(1)

    # Load model
    model = Autoencoder(in_channels=cfg.in_channels).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # Collect a capped sample of test-region patches to compute threshold
    import random
    random.seed(43)
    region_indices = [
        i for i, s in enumerate(ds.samples)
        if Path(s).stem.startswith(args.region)
    ]
    if not region_indices:
        print(f"WARNING: no patches found for region '{args.region}', "
              "using full dataset for threshold scan")
        region_indices = list(range(len(ds)))
    sample_size = min(150, len(region_indices))
    sampled = random.sample(region_indices, sample_size)
    region_patches = [ds[i] for i in sampled]

    print(f"Computing threshold from {sample_size} '{args.region}' patches "
          f"(of {len(region_indices)} total)...")
    threshold = find_optimal_threshold(model, region_patches, device, args.dnbr_threshold)

    # Generate figure
    raw = ds[target_idx]
    fig = make_confmap_figure(
        model, raw, args.patch_name, threshold, device,
        dnbr_threshold=args.dnbr_threshold,
        fold_label=args.fold_label,
        cos=args.cos,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir) / f"{args.patch_name}_confmap.png"
    fig.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
