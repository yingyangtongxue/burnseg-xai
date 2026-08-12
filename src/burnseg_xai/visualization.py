"""
Visualization utilities -- saliency maps, training curves, and reconstruction
error distribution figures saved as MLflow artifacts.

All figures use a non-interactive Agg backend so they work on headless
servers / remote machines without a display.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from burnseg_xai.evaluator import _prepare_batch

# Internal helpers

def _compute_saliency(model, x: torch.Tensor):
    """
    Forward pass + gradient-based saliency for a single (already-prepared) sample.

    Returns
    -------
    saliency : (H, W) float32 numpy array, normalized to [0, 1]
    recon_err: (H, W) float32 numpy array, absolute MSE per pixel
    """
    x = x.requires_grad_(True)
    x_hat, _ = model(x)
    loss = F.mse_loss(x_hat, x)

    grads = torch.autograd.grad(loss, x, create_graph=False)[0]  # (1,C,T,H,W)

    sal = grads.abs().mean(dim=(1, 2))[0]               # (H, W)
    sal = torch.nan_to_num(sal, nan=0.0, posinf=0.0)
    sal = (sal / sal.amax().clamp(min=1e-8)).detach().cpu().numpy().astype(np.float32)

    # Absolute per-pixel reconstruction error (not normalized, so scale is meaningful)
    err = (x_hat - x).pow(2).mean(dim=(1, 2))[0]        # (H, W)
    err = err.detach().cpu().numpy().astype(np.float32)

    return sal, err


def _log_fig(fig: plt.Figure, name: str, artifact_path: str = "") -> None:
    """Save figure to a temp file and log it as an MLflow artifact."""
    import mlflow
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def _get_patch_name(ds, global_idx: int) -> str:
    """
    Return the patch stem (e.g. 'karipuna_patch_00042') for the sample at
    global_idx in the dataloader's dataset. Works with both Subset and
    BurnedAreaDataset directly.
    """
    try:
        if hasattr(ds, "indices"):          # torch.utils.data.Subset
            path = ds.dataset.samples[ds.indices[global_idx]]
        else:                               # BurnedAreaDataset directly
            path = ds.samples[global_idx]
        return os.path.splitext(os.path.basename(path))[0]
    except Exception:
        return f"sample_{global_idx:05d}"


# Saliency map figures

def make_saliency_figures(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int = 6,
    dnbr_threshold: float = 0.15,
) -> Tuple[List[plt.Figure], List[str]]:
    """
    Creates publication-quality saliency figures, one per sample.

    Only patches with peak dNBR > dnbr_threshold are selected so that the
    figures show the model attending to real burn events, not noise.

    Each figure has 4 panels:
      Col 1 -- dNBR prior (temporal max, normalized to [0,1])
      Col 2 -- Saliency map  |dL_recon/dx|  (normalized to [0,1])
      Col 3 -- Saliency overlay on dNBR  (element-wise product)
      Col 4 -- Reconstruction error per pixel  (per-sample colorscale)

    Returns
    -------
    figures : list of matplotlib Figure objects
    names   : list of patch stem strings (e.g. 'karipuna_patch_00042')
    """
    model.eval()
    ds = dataloader.dataset
    candidates = []

    # First pass: collect candidates with meaningful dNBR signal
    global_idx = 0
    for batch in dataloader:
        for b_idx in range(batch.size(0)):
            single = batch[b_idx : b_idx + 1]
            x, prior = _prepare_batch(single, device)
            if prior[0].amax().item() >= dnbr_threshold:
                prior_np = prior[0].detach().cpu().numpy()
                sal_np, err_np = _compute_saliency(model, x)
                patch_name = _get_patch_name(ds, global_idx)
                candidates.append((prior_np, sal_np, err_np, patch_name))
            global_idx += 1
            if len(candidates) >= n_samples * 4:
                break
        if len(candidates) >= n_samples * 4:
            break

    # Fallback: any patches if none passed the threshold
    if not candidates:
        global_idx = 0
        for batch in dataloader:
            for b_idx in range(batch.size(0)):
                single = batch[b_idx : b_idx + 1]
                x, prior = _prepare_batch(single, device)
                prior_np = prior[0].detach().cpu().numpy()
                sal_np, err_np = _compute_saliency(model, x)
                patch_name = _get_patch_name(ds, global_idx)
                candidates.append((prior_np, sal_np, err_np, patch_name))
                global_idx += 1
                if len(candidates) >= n_samples:
                    break
            if len(candidates) >= n_samples:
                break

    # Sort by dNBR signal strength (strongest burn first)
    candidates.sort(key=lambda c: float(c[0].max()), reverse=True)
    selected = candidates[:n_samples]

    figures: List[plt.Figure] = []
    names: List[str] = []

    for prior_np, sal_np, err_np, patch_name in selected:
        overlay_np = sal_np * prior_np
        # Per-sample colorscale so low-error patches are not black
        err_vmax = float(np.percentile(err_np, 99)) + 1e-8

        fig = plt.figure(figsize=(17, 4))
        gs = gridspec.GridSpec(1, 4, wspace=0.08)

        # 1 -- dNBR prior
        ax0 = fig.add_subplot(gs[0])
        im0 = ax0.imshow(prior_np, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax0.set_title("dNBR prior\n(temporal max, norm.)", fontsize=8)
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # 2 -- Saliency map
        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(sal_np, cmap="hot", vmin=0, vmax=1)
        ax1.set_title("Saliency\n|dL/dx| (norm.)", fontsize=8)
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 3 -- Overlay: saliency * dNBR (alignment map)
        ax2 = fig.add_subplot(gs[2])
        im2 = ax2.imshow(overlay_np, cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_title("Alignment overlay\n(saliency x dNBR)", fontsize=8)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 4 -- Absolute reconstruction error (per-sample scale)
        ax3 = fig.add_subplot(gs[3])
        err_vmin = float(np.percentile(err_np, 1))
        im3 = ax3.imshow(err_np, cmap="inferno", vmin=err_vmin, vmax=err_vmax)
        ax3.set_title("Recon error\n(abs. MSE per pixel)", fontsize=8)
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Cosine similarity for this sample
        s_flat = sal_np.flatten()
        p_flat = prior_np.flatten()
        cos = float(np.dot(s_flat, p_flat) / (
            np.linalg.norm(s_flat) * np.linalg.norm(p_flat) + 1e-8
        ))
        fig.suptitle(
            f"{patch_name}  --  saliency/dNBR cosine = {cos:.3f}",
            fontsize=9,
            y=1.01,
        )
        figures.append(fig)
        names.append(patch_name)

    return figures, names


def log_figures_to_mlflow(
    figures: List[plt.Figure],
    prefix: str = "saliency",
    names: Optional[List[str]] = None,
) -> None:
    """Saves each figure as PNG and logs to the active MLflow run.

    Parameters
    ----------
    figures : list of Figure objects
    prefix  : MLflow artifact sub-path (e.g. 'saliency', 'comparison')
    names   : optional list of stem names for the output files; when provided,
              files are saved as ``{name}.png`` instead of ``{prefix}_{i:03d}.png``
    """
    import mlflow
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, prefix)
        os.makedirs(subdir, exist_ok=True)
        for i, fig in enumerate(figures):
            if names and i < len(names):
                fname = f"{names[i]}.png"
            else:
                fname = f"{os.path.basename(prefix)}_{i:03d}.png"
            fpath = os.path.join(subdir, fname)
            fig.savefig(fpath, dpi=130, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(fpath, artifact_path=prefix)


# Training curve figure

def plot_training_curves(metrics_per_epoch: List[Dict], run_name: str = "") -> plt.Figure:
    """
    Produces a 2-panel training curve figure:
      Top    -- train_loss and val_loss over epochs
      Bottom -- val_saliency_cosine over epochs

    Parameters
    ----------
    metrics_per_epoch : list of dicts, one per epoch, each containing at least
        "epoch", "train_loss", "val_loss", "val_saliency_cosine"
    run_name : used as the figure title
    """
    epochs = [m["epoch"] for m in metrics_per_epoch]
    train_loss = [m.get("train_loss", float("nan")) for m in metrics_per_epoch]
    val_loss   = [m.get("val_loss",   float("nan")) for m in metrics_per_epoch]
    cos        = [m.get("val_saliency_cosine", float("nan")) for m in metrics_per_epoch]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    ax1.plot(epochs, train_loss, label="train loss", color="#2563eb", linewidth=1.5)
    ax1.plot(epochs, val_loss,   label="val loss",   color="#dc2626", linewidth=1.5)
    ax1.set_ylabel("MSE loss", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, cos, label="val saliency cosine", color="#16a34a", linewidth=1.5)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Cosine similarity", fontsize=9)
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Training curves -- {run_name}" if run_name else "Training curves", fontsize=10)
    return fig


# Reconstruction error distribution

def plot_recon_error_distribution(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
    run_name: str = "",
) -> plt.Figure:
    """
    Plots per-patch reconstruction error distribution, split into two groups:
      - Patches where mean dNBR > dnbr_threshold  (proxy for burned)
      - Patches where mean dNBR <= dnbr_threshold (proxy for non-burned)

    The gap between distributions is the anomaly detection signal.
    """
    model.eval()
    burned_errors = []
    clean_errors  = []

    for batch in dataloader:
        raw = batch.to(device)
        dnbr_mean = raw[..., 20].mean(dim=(1, 2, 3)).cpu().numpy()  # (B,)

        x, _ = _prepare_batch(batch, device)
        with torch.no_grad():
            x_hat, _ = model(x)
            per_sample = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2, 3, 4))

        for j in range(len(dnbr_mean)):
            err = per_sample[j].item()
            if dnbr_mean[j] > dnbr_threshold:
                burned_errors.append(err)
            else:
                clean_errors.append(err)

    fig, ax = plt.subplots(figsize=(8, 4))

    all_errors = burned_errors + clean_errors
    if not all_errors:
        ax.set_title("No data", fontsize=9)
        return fig

    # Clip to 99th percentile to prevent outliers from collapsing the visible range
    p99 = float(np.percentile(all_errors, 99))
    bins = np.linspace(0, p99 * 1.05 + 1e-8, 50)

    if clean_errors:
        ax.hist(clean_errors,  bins=bins, alpha=0.65, color="#3b82f6",
                label=f"Non-burned proxy  (n={len(clean_errors)})", density=True)
    if burned_errors:
        ax.hist(burned_errors, bins=bins, alpha=0.65, color="#ef4444",
                label=f"Burned proxy  (n={len(burned_errors)})", density=True)

    ax.set_xlabel("Per-patch reconstruction error (MSE)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    sep = ""
    if burned_errors and clean_errors:
        gap = np.mean(burned_errors) - np.mean(clean_errors)
        sep = f"  |  mean gap = {gap:+.4f}"

    ax.set_title(
        f"Reconstruction error distribution -- {run_name}{sep}",
        fontsize=9,
    )
    return fig


# dNBR mask vs model prediction comparison

def make_comparison_figures(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    pixel_threshold: float,
    n_samples: int = 6,
    dnbr_threshold: float = 0.1,
    dnbr_vmin: float = -0.5,
    dnbr_vmax: float = 1.0,
) -> Tuple[List[plt.Figure], List[str]]:
    """
    Comparison figures: dNBR mask (USGS proxy GT) vs model prediction.

    Each figure has 5 panels:
      Col 1 -- Raw dNBR (continuous, USGS colorscale)
      Col 2 -- dNBR binary mask  (dNBR > 0.10, reference)
      Col 3 -- Reconstruction error map (continuous, model output)
      Col 4 -- Binary burn prediction  (recon error > pixel_threshold)
      Col 5 -- Confusion map  (TP=green, FP=yellow, FN=red, TN=light gray)

    Only patches with at least one burned pixel in the dNBR mask are selected
    so the figures show meaningful burn events.

    Parameters
    ----------
    pixel_threshold : the recon-error threshold chosen on the val set.

    Returns
    -------
    figures : list of matplotlib Figure objects
    names   : list of patch stem strings (e.g. 'karipuna_patch_00042')
    """
    model.eval()
    ds = dataloader.dataset
    candidates = []
    global_idx = 0

    for batch in dataloader:
        raw = batch.to(device)
        # Raw dNBR before any normalisation (B, T, H, W) → peak (B, H, W)
        dnbr_raw  = raw[..., 20].max(dim=1).values        # (B, H, W)
        dnbr_mask = (dnbr_raw > dnbr_threshold).cpu().numpy()  # bool (B, H, W)

        x, _ = _prepare_batch(batch, device)
        with torch.no_grad():
            x_hat, _ = model(x)
            err = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))  # (B, H, W)

        for b_idx in range(batch.size(0)):
            mask_b = dnbr_mask[b_idx]
            if mask_b.sum() > 0:
                patch_name = _get_patch_name(ds, global_idx)
                candidates.append({
                    "dnbr_raw":   dnbr_raw[b_idx].cpu().numpy(),
                    "dnbr_mask":  mask_b,
                    "err":        err[b_idx].cpu().numpy(),
                    "n_burned":   int(mask_b.sum()),
                    "patch_name": patch_name,
                })
            global_idx += 1
            if len(candidates) >= n_samples * 4:
                break
        if len(candidates) >= n_samples * 4:
            break

    # Sort by burn extent (most burned first)
    candidates.sort(key=lambda c: c["n_burned"], reverse=True)
    selected = candidates[:n_samples]

    if not selected:
        return [], []

    figures: List[plt.Figure] = []
    names: List[str] = []

    for c in selected:
        dnbr_np    = c["dnbr_raw"]
        mask_np    = c["dnbr_mask"].astype(float)
        err_np     = c["err"]
        pred_np    = (err_np >= pixel_threshold).astype(float)
        patch_name = c["patch_name"]
        # Per-patch colorscale: 99th percentile avoids single bright pixels squashing contrast
        err_vmax = float(np.percentile(err_np, 99)) + 1e-8

        # Confusion map: TP/FP/FN/TN as RGBA image
        H, W = dnbr_np.shape
        confusion_rgba = np.zeros((H, W, 4), dtype=np.float32)
        tp = (mask_np == 1) & (pred_np == 1)
        fp = (mask_np == 0) & (pred_np == 1)
        fn = (mask_np == 1) & (pred_np == 0)
        tn = (mask_np == 0) & (pred_np == 0)
        confusion_rgba[tn]  = [0.85, 0.85, 0.85, 0.50]   # TN: light gray
        confusion_rgba[tp]  = [0.10, 0.75, 0.10, 0.90]   # TP: green
        confusion_rgba[fp]  = [1.00, 0.85, 0.00, 0.90]   # FP: yellow
        confusion_rgba[fn]  = [0.95, 0.10, 0.10, 0.90]   # FN: red

        f1_val = (2 * tp.sum()) / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8)

        fig = plt.figure(figsize=(21, 4))
        gs  = gridspec.GridSpec(1, 5, wspace=0.08)

        # 1 -- Raw dNBR
        ax0 = fig.add_subplot(gs[0])
        im0 = ax0.imshow(dnbr_np, cmap="RdYlGn_r", vmin=dnbr_vmin, vmax=dnbr_vmax)
        ax0.set_title("dNBR\n(raw, continuous)", fontsize=8)
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # 2 -- dNBR binary mask (reference)
        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
        ax1.set_title(f"dNBR mask\n(dNBR > {dnbr_threshold}, USGS)", fontsize=8)
        ax1.axis("off")

        # 3 -- Reconstruction error
        ax2 = fig.add_subplot(gs[2])
        err_vmin = float(np.percentile(err_np, 1))
        im2 = ax2.imshow(err_np, cmap="inferno", vmin=err_vmin, vmax=err_vmax)
        ax2.set_title("Recon error\n(model output)", fontsize=8)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 4 -- Binary burn prediction
        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(pred_np, cmap="Reds", vmin=0, vmax=1)
        ax3.set_title("Burn prediction\n(recon > threshold)", fontsize=8)
        ax3.axis("off")

        # 5 -- Confusion map
        ax4 = fig.add_subplot(gs[4])
        ax4.imshow(confusion_rgba)
        n_total = dnbr_np.size
        ax4.set_title(
            f"Confusion map\nF1={f1_val:.3f}",
            fontsize=8,
        )
        ax4.axis("off")

        # Legend below the figure, outside all axes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.10, 0.75, 0.10),
                  label=f"TP  {tp.sum()}  ({100*tp.sum()/n_total:.1f}%)"),
            Patch(facecolor=(1.00, 0.85, 0.00),
                  label=f"FP  {fp.sum()}  ({100*fp.sum()/n_total:.1f}%)"),
            Patch(facecolor=(0.95, 0.10, 0.10),
                  label=f"FN  {fn.sum()}  ({100*fn.sum()/n_total:.1f}%)"),
            Patch(facecolor=(0.85, 0.85, 0.85),
                  label=f"TN  {tn.sum()}  ({100*tn.sum()/n_total:.1f}%)"),
        ]
        fig.legend(handles=legend_elements,
                   loc="lower center",
                   bbox_to_anchor=(0.88, -0.12),
                   ncol=2, fontsize=7.5, framealpha=0.92,
                   title="Confusion map", title_fontsize=7)

        fig.suptitle(
            f"{patch_name}  --  burned pixels (dNBR)={tp.sum()+fn.sum()}  "
            f"predicted={tp.sum()+fp.sum()}  F1={f1_val:.3f}",
            fontsize=9, y=1.01,
        )
        fig.subplots_adjust(bottom=0.18)
        figures.append(fig)
        names.append(patch_name)

    return figures, names
