"""
analyze_results.py
------------------
Comprehensive post-hoc analysis of a trained model checkpoint.

Generates:
  1. Per-sample figures (binary masks, reconstruction, difference maps)
  2. Aggregate confusion matrix with counts + percentages
  3. IoU, Dice, Cohen's Kappa metrics (pixel-level)
  4. KML fire-hotspot overlay (INPE focos de queimada as point reference)

Usage
-----
    python -m burnseg_xai.analysis.analyze_results \\
        --checkpoint ./outputs/mlruns/.../artifacts/checkpoint_best.pt \\
        --split      ./outputs/mlruns/.../artifacts/split_indices.json \\
        --output_dir ./outputs/analysis_results \\
        --region     kayapo \\
        --kml        "./data/aoi/kayapo_focos_qmd_inpe_2024-08-01_2024-11-03_01.248323.kml"

If --region / --kml are omitted the KML overlay is skipped.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from sklearn.metrics import (
    cohen_kappa_score,
)
from tqdm import tqdm

# helpers

def _load_kml_points(kml_path: str):
    """
    Parse INPE focos-de-queimada KML file.
    Returns list of (lon, lat) tuples for every fire hotspot point.
    """
    from lxml import etree
    tree = etree.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    points = []
    for pm in root.findall(".//kml:Placemark", ns):
        coord_el = pm.find(".//kml:coordinates", ns)
        if coord_el is not None and coord_el.text:
            parts = coord_el.text.strip().split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                points.append((lon, lat))
    return points


def _points_to_patch_mask(points, transform, shape):
    """
    Rasterize (lon, lat) point list onto a patch raster grid.

    Returns boolean (H, W) array: True where at least one hotspot point falls
    in that pixel's footprint.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if not points:
        return mask
    inv = ~transform
    for lon, lat in points:
        col, row = inv * (lon, lat)
        col, row = int(col), int(row)
        if 0 <= row < H and 0 <= col < W:
            mask[row, col] = True
    return mask


def _kml_mask_dilated(points, transform, shape, dilation: int = 3):
    """
    Rasterize KML hotspot points and dilate by ``dilation`` pixels.

    INPE MODIS hotspots have ~1 km spatial resolution; each point is the
    centroid of a ~33×33 Landsat pixel block.  Dilating by a few pixels
    accounts for geolocation uncertainty so the model is not penalised for
    predicting the correct neighbourhood rather than the exact centroid pixel.

    Returns uint8 (H, W) mask: 1 = fire confirmed by INPE hotspot.
    """
    from scipy.ndimage import binary_dilation
    raw = _points_to_patch_mask(points, transform, shape)
    if dilation > 0 and raw.any():
        raw = binary_dilation(raw, iterations=dilation)
    return raw.astype(np.uint8)


def _prepare_single(raw_tensor: torch.Tensor, device: str):
    """
    Mirrors trainer._prepare_batch for a single (1, T, H, W, 22) tensor.
    Returns (x, prior, dnbr_raw) all on device.
    """
    raw = raw_tensor.to(device)                         # (1, T, H, W, 22)

    dnbr_raw = raw[0, :, :, :, 20].max(dim=0).values   # (H, W) - temporal peak raw dNBR

    x_raw = torch.cat([raw[..., :20], raw[..., 21:22]], dim=-1)  # (1, T, H, W, 21)
    x = x_raw.permute(0, 4, 1, 2, 3).contiguous()      # (1, 21, T, H, W)

    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    std  = x.std(dim=(1, 2, 3, 4),  keepdim=True) + 1e-8
    x    = (x - mean) / std

    prior = dnbr_raw.clone()
    prior = torch.relu(prior)
    prior = prior * (prior > 0.1).float()
    prior = prior / prior.amax().clamp(min=1e-8)

    return x, prior, dnbr_raw


def _recon_error_map(model, x: torch.Tensor) -> np.ndarray:
    """Per-pixel MSE averaged over channels and time. Returns (H, W) float32."""
    with torch.no_grad():
        x_hat, _ = model(x)
        err = F.mse_loss(x_hat, x, reduction="none")  # (1, C, T, H, W)
        err = err.mean(dim=(1, 2))[0]                 # (H, W)
    return err.cpu().numpy().astype(np.float32)


def _find_optimal_threshold(model, dataset, val_indices, device, dnbr_threshold=0.1,
                            n_candidates=200):
    """Grid-search threshold on val set maximising pixel-level F1 vs dNBR mask."""
    from sklearn.metrics import precision_recall_curve

    all_scores, all_labels = [], []
    for idx in tqdm(val_indices, desc="Val threshold search", leave=False):
        raw = dataset[idx]  # (1, T, H, W, 22)
        x, _, dnbr_raw = _prepare_single(
            raw.unsqueeze(0) if raw.dim() == 4 else raw.unsqueeze(0), device
        )
        err_map = _recon_error_map(model, x)
        gt_mask = (dnbr_raw.cpu().numpy() > dnbr_threshold).astype(np.uint8)
        all_scores.append(err_map.ravel())
        all_labels.append(gt_mask.ravel())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float(np.median(scores))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision + recall
    denom[denom == 0] = 1e-8
    f1_arr = 2 * precision * recall / denom
    if len(thresholds) == 0:
        return float(np.median(scores))
    best_idx = int(np.argmax(f1_arr[:-1]))
    return float(thresholds[best_idx])


# per-sample figures

def _make_sample_figure(
    sample_idx: int,
    patch_path: str,
    region: str,
    dnbr_raw_np: np.ndarray,
    err_map: np.ndarray,
    pixel_threshold: float,
    dnbr_threshold: float = 0.1,
    kml_points: list = None,
    kml_dilation: int = 3,
) -> plt.Figure:
    """
    8-panel figure per patch.

    Reference hierarchy for metrics (IoU, Dice, Kappa, F1):
      1. KML focos de queimada (INPE validated, dilated by kml_dilation pixels),
         used when the KML file contains at least one hotspot inside this patch.
      2. dNBR > dnbr_threshold proxy, fallback when no KML hotspot is present.

    Both references are always shown in separate panels for comparison.
    The panel header says "PRIMARY REFERENCE (KML)" or "PRIMARY REFERENCE (dNBR)".

    Title includes: dataset index, filename, region.
    Output filenames: {NNN}_{region}_{patch_name}_idx{idx}.png
    """
    H, W = dnbr_raw_np.shape
    patch_name = os.path.splitext(os.path.basename(patch_path))[0]

    pred_mask  = (err_map >= pixel_threshold).astype(np.uint8)
    dnbr_mask  = (dnbr_raw_np > dnbr_threshold).astype(np.uint8)

    # Determine primary reference
    kml_mask_raw = None
    kml_mask_dilated = None
    if kml_points is not None:
        try:
            with rasterio.open(patch_path) as src:
                kml_mask_raw = _points_to_patch_mask(kml_points, src.transform, (H, W))
                kml_mask_dilated = _kml_mask_dilated(
                    kml_points, src.transform, (H, W), dilation=kml_dilation
                )
        except Exception:
            pass

    if kml_mask_dilated is not None and kml_mask_dilated.any():
        ref_mask   = kml_mask_dilated
        ref_label  = f"KML focos INPE (±{kml_dilation}px dilation)"
        using_kml  = True
    else:
        ref_mask   = dnbr_mask
        ref_label  = f"dNBR proxy (>{dnbr_threshold})"
        using_kml  = False

    # Confusion components
    tp = (ref_mask == 1) & (pred_mask == 1)
    fp = (ref_mask == 0) & (pred_mask == 1)
    fn = (ref_mask == 1) & (pred_mask == 0)
    tn = (ref_mask == 0) & (pred_mask == 0)

    n_total = H * W
    n_tp = int(tp.sum())
    n_fp = int(fp.sum())
    n_fn = int(fn.sum())
    n_tn = int(tn.sum())

    eps = 1e-8
    precision = n_tp / (n_tp + n_fp + eps)
    recall    = n_tp / (n_tp + n_fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = n_tp / (n_tp + n_fp + n_fn + eps)
    dice      = 2 * n_tp / (2 * n_tp + n_fp + n_fn + eps)

    flat_ref  = ref_mask.ravel()
    flat_pred = pred_mask.ravel()
    kappa = float("nan")
    if flat_ref.sum() > 0 and (1 - flat_ref).sum() > 0:
        kappa = float(cohen_kappa_score(flat_ref, flat_pred))

    diff_rgba = np.zeros((H, W, 4), dtype=np.float32)
    diff_rgba[tn]  = [0.00, 0.00, 0.00, 1.0]
    diff_rgba[tp]  = [1.00, 1.00, 1.00, 1.0]
    diff_rgba[fp]  = [1.00, 0.85, 0.00, 1.0]
    diff_rgba[fn]  = [0.85, 0.10, 0.10, 1.0]

    err_vmax = max(float(err_map.max()), 1e-6)

    # Figure layout (8 panels)
    ncols = 8
    fig = plt.figure(figsize=(ncols * 3.5, 4.2))
    gs  = gridspec.GridSpec(1, ncols, wspace=0.06)
    fig.patch.set_facecolor("#111111")

    def _ax(n, title, title_color="white"):
        ax = fig.add_subplot(gs[n])
        ax.set_title(title, fontsize=7.5, color=title_color, pad=3)
        ax.axis("off")
        ax.set_facecolor("black")
        return ax

    # 1 - Raw dNBR (continuous)
    ax = _ax(0, "dNBR (raw continuous)")
    im = ax.imshow(dnbr_raw_np, cmap="RdYlGn_r", vmin=-0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
        color="white", labelcolor="white"
    )

    # 2 - PRIMARY REFERENCE (KML when available, else dNBR proxy)
    ref_title = f"PRIMARY REFERENCE\n({ref_label})"
    ax = _ax(1, ref_title, title_color="cyan" if using_kml else "yellow")
    ax.imshow(ref_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    pct_ref = 100 * ref_mask.sum() / n_total
    kml_n_pts = int(kml_mask_raw.sum()) if kml_mask_raw is not None else 0
    suffix = f"  [{kml_n_pts} hotspot pts]" if using_kml else ""
    ax.set_xlabel(f"Burned: {pct_ref:.1f}%{suffix}", fontsize=6.5, color="white")

    # 3 - dNBR proxy binary (always shown for comparison)
    ax = _ax(2, f"dNBR proxy (>{dnbr_threshold})\n[comparison only]")
    ax.imshow(dnbr_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    pct_dnbr = 100 * dnbr_mask.sum() / n_total
    ax.set_xlabel(f"dNBR burned: {pct_dnbr:.1f}%", fontsize=6.5, color="white")
    if using_kml:
        # Overlay KML dilated contour in cyan on dNBR panel so user can compare
        if kml_mask_dilated is not None and kml_mask_dilated.any():
            ax.contour(kml_mask_dilated, levels=[0.5], colors=["cyan"],
                       linewidths=0.7, alpha=0.9)

    # 4 - Prediction binary
    ax = _ax(3, "Prediction binary\n(recon ≥ threshold)")
    ax.imshow(pred_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    pct_pred = 100 * pred_mask.sum() / n_total
    ax.set_xlabel(f"Predicted burned: {pct_pred:.1f}%", fontsize=6.5, color="white")

    # 5 - Difference map (pred vs primary ref)
    ax = _ax(4, "Difference\n(pred − primary ref)")
    ax.imshow(diff_rgba, interpolation="nearest")
    legend_els = [
        Patch(fc=(1.0, 1.0, 1.0), label=f"TP  {100*n_tp/n_total:.1f}%"),
        Patch(fc=(1.0, 0.85, 0.0), label=f"FP  {100*n_fp/n_total:.1f}%"),
        Patch(fc=(0.85, 0.1, 0.1), label=f"FN  {100*n_fn/n_total:.1f}%"),
        Patch(fc=(0.1, 0.1, 0.1), ec="gray", label=f"TN  {100*n_tn/n_total:.1f}%"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=5.5, framealpha=0.7,
              facecolor="#222222", labelcolor="white")

    # 6 - Recon error heatmap with primary ref contour
    ax = _ax(5, "Error + primary ref contour")
    ax.imshow(err_map, cmap="hot", vmin=0, vmax=err_vmax, alpha=0.9)
    if ref_mask.sum() > 0:
        color = "cyan" if using_kml else "lime"
        ax.contour(ref_mask, levels=[0.5], colors=[color], linewidths=0.7, alpha=0.85)

    # 7 - Confusion matrix (2×2 vs primary ref)
    ref_short = "KML" if using_kml else "dNBR"
    ax = _ax(6, f"Confusion matrix\n(vs {ref_short})")
    cm_arr = np.array([[n_tn, n_fp], [n_fn, n_tp]])
    cm_pct = 100 * cm_arr / n_total
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for r in range(2):
        for c in range(2):
            lbl = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(r, c)]
            ax.text(c, r, f"{lbl}\n{cm_arr[r,c]:,}\n({cm_pct[r,c]:.1f}%)",
                    ha="center", va="center", fontsize=6.5,
                    color="black" if cm_pct[r, c] < 60 else "white")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred\nClean", "Pred\nBurned"], fontsize=6.0, color="white")
    ax.set_yticklabels([f"Ref\nClean\n({ref_short})", f"Ref\nBurned\n({ref_short})"],
                       fontsize=5.5, color="white")
    ax.tick_params(colors="white")

    # 8 - Metrics text (clearly labelled with which reference was used)
    ref_color = "cyan" if using_kml else "yellow"
    ax = _ax(7, f"Metrics (ref = {ref_short})", title_color=ref_color)
    metrics_lines = [
        f"F1       {f1:.4f}",
        f"IoU      {iou:.4f}",
        f"Dice     {dice:.4f}",
        f"Kappa    {kappa:.4f}" if kappa == kappa else "Kappa    n/a",
        f"Prec     {precision:.4f}",
        f"Recall   {recall:.4f}",
        "",
        f"ref src: {'KML INPE' if using_kml else 'dNBR proxy'}",
    ]
    ypos = 0.95
    for line in metrics_lines:
        ax.text(0.05, ypos, line, transform=ax.transAxes, fontsize=7,
                color="lime", family="monospace", va="top")
        ypos -= 0.12

    burn_ref_n  = int(ref_mask.sum())
    burn_pred_n = int(pred_mask.sum())
    ref_tag = "KML" if using_kml else "dNBR"
    fig.suptitle(
        f"idx={sample_idx} | {patch_name} | region={region}  "
        f"[ref={ref_tag}]\n"
        f"ref burned={burn_ref_n} px ({100*burn_ref_n/n_total:.1f}%)  "
        f"pred burned={burn_pred_n} px ({100*burn_pred_n/n_total:.1f}%)  "
        f"F1={f1:.3f}  IoU={iou:.3f}  Dice={dice:.3f}  κ={kappa:.3f}",
        fontsize=8.5, color="white", y=1.02,
    )
    fig.patch.set_facecolor("#111111")
    return fig, {
        "sample_idx":       sample_idx,
        "patch_name":       patch_name,
        "region":           region,
        "n_total":          n_total,
        "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn, "n_tn": n_tn,
        "pct_ref_burned":   100 * burn_ref_n / n_total,
        "pct_pred_burned":  100 * burn_pred_n / n_total,
        "f1": f1, "iou": iou, "dice": dice, "kappa": kappa,
        "precision": precision, "recall": recall,
        "using_kml":        using_kml,
        "kml_hotspot_pts":  kml_n_pts,
    }


# aggregate confusion matrix

def _make_aggregate_cm_figure(all_stats: list, title: str = "") -> plt.Figure:
    """
    Single aggregate confusion matrix heatmap + summary statistics table.
    """
    totals = {k: sum(s[k] for s in all_stats) for k in ("n_tp", "n_fp", "n_fn", "n_tn", "n_total")}
    n_total = totals["n_total"]
    n_tp = totals["n_tp"]
    n_fp = totals["n_fp"]
    n_fn = totals["n_fn"]
    n_tn = totals["n_tn"]

    eps = 1e-8
    agg_prec  = n_tp / (n_tp + n_fp + eps)
    agg_rec   = n_tp / (n_tp + n_fn + eps)
    agg_f1    = 2 * agg_prec * agg_rec / (agg_prec + agg_rec + eps)
    agg_iou   = n_tp / (n_tp + n_fp + n_fn + eps)
    agg_dice  = 2 * n_tp / (2 * n_tp + n_fp + n_fn + eps)
    agg_kappa = float("nan")
    try:
        flat_ref  = np.array([0] * n_tn + [0] * n_fp + [1] * n_fn + [1] * n_tp)
        flat_pred = np.array([0] * n_tn + [1] * n_fp + [0] * n_fn + [1] * n_tp)
        agg_kappa = float(cohen_kappa_score(flat_ref, flat_pred))
    except Exception:
        pass

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#111111")
    gs = gridspec.GridSpec(1, 2, wspace=0.25)

    # Left: confusion matrix heatmap
    ax_cm = fig.add_subplot(gs[0])
    cm = np.array([[n_tn, n_fp], [n_fn, n_tp]])
    cm_pct = 100 * cm / n_total
    im = ax_cm.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="% of total pixels")
    for r in range(2):
        for c in range(2):
            label = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(r, c)]
            ax_cm.text(
                c, r,
                f"{label}\n{cm[r,c]:,}\n({cm_pct[r,c]:.1f}%)",
                ha="center", va="center", fontsize=11,
                color="black" if cm_pct[r, c] < 55 else "white", weight="bold"
            )
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted\nClean", "Predicted\nBurned"], fontsize=10, color="white")
    n_kml_patches = sum(1 for s in all_stats if s.get("using_kml", False))
    ref_note = (
        f"KML INPE: {n_kml_patches}/{len(all_stats)} patches  |  "
        f"dNBR proxy: {len(all_stats)-n_kml_patches}/{len(all_stats)} patches"
    )
    ax_cm.set_yticklabels(["Reference\nClean\n(KML/dNBR)", "Reference\nBurned\n(KML/dNBR)"],
                          fontsize=9, color="white")
    ax_cm.tick_params(colors="white")
    ax_cm.set_title(
        f"Aggregate Confusion Matrix (pixel-level)\n{ref_note}",
        color="white", fontsize=10
    )

    # Right: metrics table
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    ax_txt.set_facecolor("#111111")

    rows = [
        ("Total pixels", f"{n_total:,}", ""),
        ("TP", f"{n_tp:,}", f"{100*n_tp/n_total:.2f}%"),
        ("FP", f"{n_fp:,}", f"{100*n_fp/n_total:.2f}%"),
        ("FN", f"{n_fn:,}", f"{100*n_fn/n_total:.2f}%"),
        ("TN", f"{n_tn:,}", f"{100*n_tn/n_total:.2f}%"),
        ("", "", ""),
        ("Precision", f"{agg_prec:.4f}", f"{100*agg_prec:.1f}%"),
        ("Recall", f"{agg_rec:.4f}", f"{100*agg_rec:.1f}%"),
        ("F1", f"{agg_f1:.4f}", ""),
        ("IoU (Jaccard)", f"{agg_iou:.4f}", ""),
        ("Dice", f"{agg_dice:.4f}", ""),
        ("Cohen's κ", f"{agg_kappa:.4f}" if agg_kappa == agg_kappa else "n/a", ""),
        ("", "", ""),
        ("# test patches", str(len(all_stats)), ""),
        ("Patches w/ burn ref", str(sum(1 for s in all_stats if s["n_tp"] + s["n_fn"] > 0)), ""),
    ]
    colors = ["white"] * len(rows)
    for i, (k, v, p) in enumerate(rows):
        if not k:
            continue
        ax_txt.text(0.02, 1.0 - 0.065 * i, k + ":", fontsize=10,
                    color="#aaaaaa", transform=ax_txt.transAxes, va="top", family="monospace")
        ax_txt.text(0.45, 1.0 - 0.065 * i, v, fontsize=10,
                    color="white", transform=ax_txt.transAxes, va="top", family="monospace")
        if p:
            ax_txt.text(0.72, 1.0 - 0.065 * i, p, fontsize=10,
                        color="lime", transform=ax_txt.transAxes, va="top", family="monospace")

    fig.suptitle(
        f"Aggregate Results - {title}\n"
        f"F1={agg_f1:.4f}  IoU={agg_iou:.4f}  Dice={agg_dice:.4f}  κ={agg_kappa:.4f}",
        fontsize=11, color="white",
    )
    return fig, {
        "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn, "n_tn": n_tn,
        "n_total": n_total,
        "precision": agg_prec, "recall": agg_rec, "f1": agg_f1,
        "iou": agg_iou, "dice": agg_dice, "kappa": agg_kappa,
    }


# KML coverage summary

def _make_kml_coverage_figure(all_stats: list, kml_hit_counts: list,
                               title: str = "") -> plt.Figure:
    """
    Bar chart: per-patch F1 coloured by whether the patch has KML hotspots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")

    # Panel 1: per-patch F1 bar chart, colored by KML presence
    ax = axes[0]
    ax.set_facecolor("#111111")
    xs = range(len(all_stats))
    colors = ["cyan" if h > 0 else "#555555" for h in kml_hit_counts]
    bars = ax.bar(xs, [s["f1"] for s in all_stats], color=colors, width=0.7)
    ax.axhline(np.mean([s["f1"] for s in all_stats]), color="orange", lw=1.5,
               linestyle="--", label=f"mean F1 = {np.mean([s['f1'] for s in all_stats]):.3f}")
    ax.set_xlabel("Test patch index", color="white", fontsize=9)
    ax.set_ylabel("F1", color="white", fontsize=9)
    ax.set_title("Per-patch F1\n(cyan = KML hotspot present)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    from matplotlib.patches import Patch as MPatch
    ax.legend(handles=[
        MPatch(fc="cyan", label="KML hotspot in patch"),
        MPatch(fc="#555555", label="No KML hotspot"),
    ] + [plt.Line2D([0], [0], color="orange", lw=1.5, ls="--",
                    label=f"mean F1 = {np.mean([s['f1'] for s in all_stats]):.3f}")],
        fontsize=7.5, facecolor="#222222", labelcolor="white")

    # Panel 2: IoU distribution
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    iou_vals = [s["iou"] for s in all_stats]
    dice_vals = [s["dice"] for s in all_stats]
    ax2.hist(iou_vals,  bins=20, alpha=0.65, color="#3b82f6",
             label=f"IoU  μ={np.mean(iou_vals):.3f}  σ={np.std(iou_vals):.3f}")
    ax2.hist(dice_vals, bins=20, alpha=0.55, color="#ef4444",
             label=f"Dice μ={np.mean(dice_vals):.3f}  σ={np.std(dice_vals):.3f}")
    ax2.set_xlabel("Value", color="white", fontsize=9)
    ax2.set_ylabel("Patches", color="white", fontsize=9)
    ax2.set_title("IoU / Dice distribution across test patches", color="white", fontsize=9)
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=8, facecolor="#222222", labelcolor="white")

    fig.suptitle(f"KML & metric overview - {title}", color="white", fontsize=10)
    return fig


# main entry

def run_analysis(
    checkpoint_path: str,
    split_path: str,
    output_dir: str,
    dataset_root: str = r"./data",
    kml_path: str = None,
    region: str = None,
    dnbr_threshold: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_sample_figures: int = 10,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from burnseg_xai.dataset import BurnedAreaDataset
    from burnseg_xai.models.autoencoder import Autoencoder

    model = Autoencoder(in_channels=21)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[analyze] Model loaded from {checkpoint_path}")

    # Load dataset
    dataset = BurnedAreaDataset(root_dir=dataset_root, temporal_length=1)
    print(f"[analyze] Dataset: {len(dataset)} patches")

    # Load split
    with open(split_path) as f:
        split = json.load(f)
    val_indices  = split["val"]
    test_indices = split["test"]
    print(f"[analyze] Split: val={len(val_indices)}  test={len(test_indices)}")

    # Find optimal pixel threshold from val set
    print("[analyze] Searching for optimal pixel threshold on val set …")
    pixel_threshold = _find_optimal_threshold(
        model, dataset, val_indices, device, dnbr_threshold
    )
    print(f"[analyze] pixel_threshold = {pixel_threshold:.6f}")

    # Load KML points
    kml_points = None
    if kml_path and os.path.exists(kml_path):
        kml_points = _load_kml_points(kml_path)
        print(f"[analyze] KML hotspots loaded: {len(kml_points)} points from {kml_path}")
    else:
        print("[analyze] No KML file provided or not found, skipping hotspot overlay")

    # Run inference on all test patches
    all_stats = []
    kml_hit_counts = []
    sample_fig_count = 0

    # Select patches with burn signal for figure selection
    patch_stats_list = []

    for idx in tqdm(test_indices, desc="Test inference"):
        raw = dataset[idx]  # (1, T, H, W, 22)
        if raw.dim() == 4:
            raw = raw.unsqueeze(0)  # (1, 1, H, W, 22)

        x, _, dnbr_raw_t = _prepare_single(raw, device)
        dnbr_np = dnbr_raw_t.cpu().numpy()

        with torch.no_grad():
            err_map = _recon_error_map(model, x)

        ref_mask = (dnbr_np > dnbr_threshold).astype(np.uint8)
        n_burn = int(ref_mask.sum())
        patch_name = os.path.splitext(os.path.basename(dataset.samples[idx]))[0]
        patch_region = dataset.get_region(idx)
        patch_stats_list.append((idx, dnbr_np, err_map, n_burn, patch_name, patch_region))

    # Sort by burn signal (most burned first) for figures
    patch_stats_list.sort(key=lambda t: t[3], reverse=True)

    for idx, dnbr_np, err_map, n_burn, patch_name, patch_region in tqdm(
        patch_stats_list, desc="Figures + stats"
    ):
        # Get patch path for rasterio transform (KML overlay)
        patch_path = dataset.samples[idx]

        # KML hotspot count for this patch
        n_kml = 0
        if kml_points is not None:
            try:
                with rasterio.open(patch_path) as src:
                    patch_kml_mask = _points_to_patch_mask(
                        kml_points, src.transform, dnbr_np.shape
                    )
                    n_kml = int(patch_kml_mask.sum())
            except Exception:
                n_kml = 0
        kml_hit_counts.append(n_kml)

        # Per-sample figure (only for top N_burned patches)
        save_fig = sample_fig_count < n_sample_figures

        kml_pts_for_fig = kml_points if save_fig else None
        fig, stats = _make_sample_figure(
            sample_idx=idx,
            patch_path=patch_path,
            region=patch_region,
            dnbr_raw_np=dnbr_np,
            err_map=err_map,
            pixel_threshold=pixel_threshold,
            dnbr_threshold=dnbr_threshold,
            kml_points=kml_pts_for_fig,
        )

        if save_fig:
            # Filename: {region}_{patch_name}_idx{idx}.png for full traceability
            fig_path = os.path.join(
                output_dir, f"{sample_fig_count:03d}_{patch_region}_{patch_name}_idx{idx}.png"
            )
            fig.savefig(fig_path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            sample_fig_count += 1
        plt.close(fig)
        all_stats.append(stats)

    # Aggregate confusion matrix
    print("[analyze] Generating aggregate confusion matrix …")
    run_label = os.path.basename(os.path.dirname(os.path.dirname(split_path)))
    agg_fig, agg_metrics = _make_aggregate_cm_figure(all_stats, title=run_label)
    agg_path = os.path.join(output_dir, "aggregate_confusion_matrix.png")
    agg_fig.savefig(agg_path, dpi=130, bbox_inches="tight",
                    facecolor=agg_fig.get_facecolor())
    plt.close(agg_fig)
    print(f"[analyze] Saved: {agg_path}")

    # KML / IoU overview
    overview_fig = _make_kml_coverage_figure(all_stats, kml_hit_counts, title=run_label)
    ov_path = os.path.join(output_dir, "metrics_overview.png")
    overview_fig.savefig(ov_path, dpi=130, bbox_inches="tight",
                         facecolor=overview_fig.get_facecolor())
    plt.close(overview_fig)
    print(f"[analyze] Saved: {ov_path}")

    # Print and save summary JSON
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    m = agg_metrics
    print(f"  Total pixels:  {m['n_total']:,}")
    print(f"  TP:  {m['n_tp']:,}  ({100*m['n_tp']/m['n_total']:.2f}%)")
    print(f"  FP:  {m['n_fp']:,}  ({100*m['n_fp']/m['n_total']:.2f}%)")
    print(f"  FN:  {m['n_fn']:,}  ({100*m['n_fn']/m['n_total']:.2f}%)")
    print(f"  TN:  {m['n_tn']:,}  ({100*m['n_tn']/m['n_total']:.2f}%)")
    print(f"  Precision:  {m['precision']:.4f}  ({100*m['precision']:.1f}%)")
    print(f"  Recall:     {m['recall']:.4f}  ({100*m['recall']:.1f}%)")
    print(f"  F1:         {m['f1']:.4f}")
    print(f"  IoU:        {m['iou']:.4f}")
    print(f"  Dice:       {m['dice']:.4f}")
    print(f"  Kappa:      {m['kappa']:.4f}" if m['kappa'] == m['kappa'] else "  Kappa:  n/a")
    print("=" * 70)
    print(f"  KML patches (hotspot present): {sum(1 for n in kml_hit_counts if n > 0)} / {len(kml_hit_counts)}")
    print("=" * 70)

    summary = {"pixel_threshold": pixel_threshold, "aggregate": agg_metrics,
                "per_patch": all_stats}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[analyze] All outputs saved to: {output_dir}")


# CLI

def main():
    parser = argparse.ArgumentParser(description="Post-hoc analysis of trained model checkpoint")
    parser.add_argument("--checkpoint",    required=True,  help="Path to checkpoint_best.pt")
    parser.add_argument("--split",         required=True,  help="Path to split_indices.json")
    parser.add_argument("--output_dir",    required=True,  help="Where to save figures and JSON")
    parser.add_argument("--dataset_root",  default=r"./data")
    parser.add_argument("--kml",           default=None,   help="Path to KML fire hotspot file")
    parser.add_argument("--region",        default=None,   help="Region name (for display)")
    parser.add_argument("--dnbr_threshold",type=float, default=0.1)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples",     type=int, default=10, help="Number of per-patch figures")
    args = parser.parse_args()

    run_analysis(
        checkpoint_path=args.checkpoint,
        split_path=args.split,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        kml_path=args.kml,
        region=args.region,
        dnbr_threshold=args.dnbr_threshold,
        device=args.device,
        n_sample_figures=args.n_samples,
    )


if __name__ == "__main__":
    main()
