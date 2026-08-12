"""
rerun_eval.py: re-run evaluation + figures for a completed run whose
evaluation step crashed (e.g. missing scikit-image).

Usage:
    python scripts/rerun_eval.py --config configs/config_runpod.yaml \
        --checkpoint /workspace/experimento_queimadas/checkpoints/baseline_seed43_2026-05-01/checkpoint_best.pt \
        --run_name baseline_seed43_2026-05-01 \
        --lambda_rrr 0.0

The script creates a NEW MLflow run tagged with reeval=true so the original
failed run is kept for audit and the new one has complete metrics + figures.
"""

import argparse
import os

import mlflow
import torch
from torch.utils.data import DataLoader, Subset

from burnseg_xai.config import load_config
from burnseg_xai.dataset import BurnedAreaDataset
from burnseg_xai.evaluator import (
    evaluate,
    find_optimal_threshold,
    find_pixel_threshold,
    otsu_segmentation_metrics,
    pixel_segmentation_metrics,
    proxy_auc,
    recon_separation,
    segmentation_metrics,
)
from burnseg_xai.logging.mlflow_logger import MLflowLogger
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.split import save_master_split
from burnseg_xai.utils.seed import set_seed
from burnseg_xai.visualization import (
    _log_fig,
    log_figures_to_mlflow,
    make_comparison_figures,
    make_saliency_figures,
    plot_recon_error_distribution,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--run_name",   required=True, help="Original run name (used for labelling)")
    parser.add_argument("--lambda_rrr", type=float, default=0.0)
    parser.add_argument("--rrr_distance_metric", default="mse")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.lambda_rrr = args.lambda_rrr
    cfg.rrr_distance_metric = args.rrr_distance_metric

    set_seed(cfg.seed)

    # Dataset + split
    full_dataset = BurnedAreaDataset(root_dir=cfg.dataset_root, temporal_length=cfg.temporal_length)
    split_path = os.path.join(cfg.output_root, "split_master.json")
    train_idx, val_idx, test_idx = save_master_split(full_dataset, split_path, seed=cfg.seed)

    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
    )

    # Model
    model = Autoencoder(in_channels=cfg.in_channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # New MLflow run (reeval)
    reeval_name = f"{args.run_name}_reeval"
    logger = MLflowLogger(cfg)
    logger.start_run(run_name=reeval_name)
    logger.set_tag("reeval", "true")
    logger.set_tag("original_run", args.run_name)
    logger.set_tag("description", f"Re-evaluation of {args.run_name} after scikit-image fix")
    logger.log_params({
        "lambda_rrr":          cfg.lambda_rrr,
        "rrr_distance_metric": cfg.rrr_distance_metric,
        "seed":                cfg.seed,
        "checkpoint":          args.checkpoint,
    })

    # Evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, cfg.device, desc="Test eval")
    logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    auc = proxy_auc(model, test_loader, cfg.device)
    if auc == auc:
        logger.log_metric("test_auc_dnbr_proxy", auc)

    sep = recon_separation(model, test_loader, cfg.device)
    if sep == sep:
        logger.log_metric("test_recon_separation", sep)

    opt_thr = find_optimal_threshold(model, val_loader, cfg.device)
    seg = segmentation_metrics(model, test_loader, cfg.device, threshold=opt_thr)
    logger.log_metrics({
        "test_f1":                seg["f1"],
        "test_precision":         seg["precision"],
        "test_recall":            seg["recall"],
        "test_accuracy":          seg["accuracy"],
        "test_balanced_accuracy": seg["balanced_accuracy"],
        "test_opt_threshold":     seg["threshold"],
        "test_n_burned":          float(seg["n_burned"]),
        "test_n_clean":           float(seg["n_clean"]),
    })

    print(f"  test_recon_error_mean  = {test_metrics['recon_error_mean']:.5f}")
    print(f"  test_saliency_cosine   = {test_metrics['saliency_cosine']:.5f}")
    print(f"  test_auc_dnbr_proxy    = {auc:.5f}")
    print(f"  test_recon_separation  = {sep:+.5f}")
    print(f"  test_f1                = {seg['f1']:.5f}")

    print("\nPixel-level segmentation...")
    px_thr = find_pixel_threshold(model, val_loader, cfg.device)
    px_seg = pixel_segmentation_metrics(model, test_loader, cfg.device, threshold=px_thr)
    logger.log_metrics({
        "test_pixel_f1":        px_seg["pixel_f1"],
        "test_pixel_iou":       px_seg["pixel_iou"],
        "test_pixel_precision": px_seg["pixel_precision"],
        "test_pixel_recall":    px_seg["pixel_recall"],
        "test_pixel_accuracy":  px_seg["pixel_accuracy"],
        "test_pixel_threshold": px_seg["threshold"],
        "test_pixel_n_burned":  float(px_seg["pixel_n_burned"]),
        "test_pixel_n_clean":   float(px_seg["pixel_n_clean"]),
    })
    print(f"  test_pixel_f1 = {px_seg['pixel_f1']:.5f}  iou = {px_seg['pixel_iou']:.5f}")

    auc_03 = proxy_auc(model, test_loader, cfg.device, dnbr_threshold=0.3, desc="Proxy AUC (dNBR>0.3)")
    sep_03 = recon_separation(model, test_loader, cfg.device, dnbr_threshold=0.3)
    if auc_03 == auc_03:
        logger.log_metric("test_auc_dnbr_proxy_0.3", auc_03)
    if sep_03 == sep_03:
        logger.log_metric("test_recon_separation_0.3", sep_03)

    otsu = otsu_segmentation_metrics(model, test_loader, cfg.device)
    logger.log_metrics({
        "test_otsu_f1":        otsu["otsu_f1"],
        "test_otsu_iou":       otsu["otsu_iou"],
        "test_otsu_kappa":     otsu["otsu_kappa"],
        "test_otsu_precision": otsu["otsu_precision"],
        "test_otsu_recall":    otsu["otsu_recall"],
        "test_otsu_threshold": otsu["otsu_threshold"],
    })
    print(f"  test_otsu_f1 = {otsu['otsu_f1']:.5f}  kappa = {otsu['otsu_kappa']:.5f}")

    # Figures
    print("\nGenerating figures...")
    try:
        figs, figs_names = make_saliency_figures(model, val_loader, cfg.device, n_samples=6)
        log_figures_to_mlflow(figs, prefix="saliency", names=figs_names)
        print(f"  {len(figs)} saliency figures logged")

        fig_dist = plot_recon_error_distribution(model, test_loader, cfg.device, run_name=reeval_name)
        _log_fig(fig_dist, "recon_error_distribution.png", artifact_path="plots")
        print("  recon_error_distribution logged")

        cmp_figs, cmp_names = make_comparison_figures(
            model, test_loader, cfg.device, pixel_threshold=px_thr, n_samples=6,
        )
        log_figures_to_mlflow(cmp_figs, prefix="comparison", names=cmp_names)
        print(f"  {len(cmp_figs)} comparison figures logged")

    except Exception as exc:
        print(f"  [warning] Visualization error: {exc}")

    logger.log_artifact(split_path)
    logger.end_run()
    print(f"\nDone: MLflow run: {reeval_name}")


if __name__ == "__main__":
    main()
