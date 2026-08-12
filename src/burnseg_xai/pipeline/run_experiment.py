"""
run_experiment.py: canonical pipeline orchestrator.

Usage examples
--------------
Baseline (no RRR):
    python -m burnseg_xai.pipeline.run_experiment --config configs/config.yaml

RRR experiment:
    python -m burnseg_xai.pipeline.run_experiment --config configs/config.yaml --lambda_rrr 0.1

Quick test (verifies XAI signal):
    python -m burnseg_xai.pipeline.run_experiment --config configs/config_quick.yaml

Resume interrupted experiment:
    (automatic -- just run the same command; checkpoint_latest.pt is detected)
"""

import argparse
import os
from datetime import date

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from burnseg_xai.config import ProjectConfig, load_config
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
from burnseg_xai.sanity_checks import run_sanity_checks
from burnseg_xai.split import create_split, save_master_split, save_split
from burnseg_xai.training.trainer import Trainer
from burnseg_xai.utils.seed import set_seed
from burnseg_xai.visualization import (
    _log_fig,
    log_figures_to_mlflow,
    make_comparison_figures,
    make_saliency_figures,
    plot_recon_error_distribution,
    plot_training_curves,
)


def _run_name(cfg: ProjectConfig) -> str:
    today = date.today().isoformat()
    if cfg.lambda_rrr == 0.0:
        return f"baseline_seed{cfg.seed}_{today}"
    all_terms = {"grad", "gradcam", "attn"}
    active = set(getattr(cfg, "xai_terms", all_terms))
    if active != all_terms:
        terms_str = "-".join(t for t in ("grad", "gradcam", "attn") if t in active)
        return f"ablation_{terms_str}_l{cfg.lambda_rrr}_{cfg.rrr_distance_metric}_seed{cfg.seed}_{today}"
    return f"rrr_l{cfg.lambda_rrr}_{cfg.rrr_distance_metric}_seed{cfg.seed}_{today}"


def run(cfg: ProjectConfig, run_name: str | None = None) -> dict:
    """
    Full training pipeline. Returns the final metrics dict from the trainer.

    Parameters
    ----------
    cfg      : ProjectConfig, full configuration.
    run_name : Optional override for the MLflow run name.
    """
    # 0. Reproducibility
    set_seed(cfg.seed)

    if run_name is None:
        run_name = _run_name(cfg)

    # 1. Dataset
    full_dataset = BurnedAreaDataset(
        root_dir=cfg.dataset_root,
        temporal_length=cfg.temporal_length,
    )

    # 1b. Optional subset (for quick tests)
    if cfg.max_samples and len(full_dataset) > cfg.max_samples:
        rng = np.random.default_rng(cfg.seed)
        sub_idx = sorted(rng.choice(len(full_dataset), cfg.max_samples, replace=False).tolist())
        dataset = Subset(full_dataset, sub_idx)
        print(f"[max_samples] using {cfg.max_samples}/{len(full_dataset)} patches")
    else:
        dataset = full_dataset

    # 2. Sanity checks
    run_sanity_checks(dataset, cfg)

    # 3. Split
    # Full-dataset runs: load persistent master split (or create on first run).
    # Subset runs: simple seeded random split into the subset indices.
    os.makedirs(cfg.output_root, exist_ok=True)
    if cfg.max_samples and len(full_dataset) > cfg.max_samples:
        train_idx, val_idx, test_idx = create_split(len(dataset), seed=cfg.seed)
        split_path = os.path.join(cfg.output_root, "split_indices.json")
        save_split({"train": train_idx, "val": val_idx, "test": test_idx}, split_path)
    else:
        split_path = os.path.join(cfg.output_root, "split_master.json")
        train_idx, val_idx, test_idx = save_master_split(
            full_dataset, split_path, seed=cfg.seed
        )

    # 4. DataLoaders
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=g,
        pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    # 5. Model and optimizer
    model = Autoencoder(in_channels=cfg.in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 6. Checkpoint subdirectory (unique per run_name)
    checkpoint_subdir = os.path.join(cfg.checkpoint_dir, run_name)
    os.makedirs(checkpoint_subdir, exist_ok=True)

    # 7. MLflow: resume if a checkpoint already exists with a run_id
    logger = MLflowLogger(cfg)
    is_resuming = False

    latest_ckpt = os.path.join(checkpoint_subdir, "checkpoint_latest.pt")
    if os.path.exists(latest_ckpt):
        saved = torch.load(latest_ckpt, map_location="cpu")
        saved_run_id = saved.get("mlflow_run_id")
        if saved_run_id:
            logger.resume_run(saved_run_id)
            is_resuming = True
            print(f"[Resume] MLflow run {saved_run_id}")

    if not is_resuming:
        mlflow_run_id = logger.start_run(run_name=run_name)

        description = (
            "Baseline autoencoder -- no RRR regularization"
            if cfg.lambda_rrr == 0.0
            else f"RRR autoencoder -- lambda={cfg.lambda_rrr} metric={cfg.rrr_distance_metric}"
        )
        logger.set_tag("description", description)
        logger.log_params({
            "batch_size":          cfg.batch_size,
            "epochs":              cfg.epochs,
            "lr":                  cfg.lr,
            "optimizer":           "adam",
            "seed":                cfg.seed,
            "temporal_length":     cfg.temporal_length,
            "in_channels":         cfg.in_channels,
            "lambda_rrr":          cfg.lambda_rrr,
            "rrr_distance_metric": cfg.rrr_distance_metric,
            "xai_terms":           ",".join(cfg.xai_terms),
            "normalization":       "zscore_per_patch",
            "model_arch":          "3dcnn_autoencoder",
            "dataset_root":        cfg.dataset_root,
            "regions":             ",".join(cfg.regions),
            "early_stopping_patience": cfg.early_stopping_patience,
        })
    else:
        mlflow_run_id = mlflow.active_run().info.run_id

    assert mlflow.active_run() is not None, "MLflow run must be active before training"

    # 8. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=cfg.device,
        logger=logger,
        lambda_rrr=cfg.lambda_rrr,
        rrr_distance_metric=cfg.rrr_distance_metric,
        xai_terms=cfg.xai_terms,
        checkpoint_dir=checkpoint_subdir,
        early_stopping_patience=cfg.early_stopping_patience,
        mlflow_run_id=mlflow_run_id,
    )

    # 9. Train (auto-resumes if checkpoint_latest.pt exists)
    final_metrics = trainer.train(train_loader, val_loader, cfg.epochs)

    # 10. Test-set evaluation
    print("\nEvaluating on test set ...")
    test_metrics = evaluate(model, test_loader, cfg.device, desc="Test eval")
    logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    auc = proxy_auc(model, test_loader, cfg.device)
    if auc == auc:
        logger.log_metric("test_auc_dnbr_proxy", auc)

    sep = recon_separation(model, test_loader, cfg.device)
    if sep == sep:
        logger.log_metric("test_recon_separation", sep)

    # Segmentation metrics: threshold found on val set, applied to test set
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
    if auc == auc:
        print(f"  test_auc_dnbr_proxy    = {auc:.5f}")
    if sep == sep:
        print(f"  test_recon_separation  = {sep:+.5f}  (burned - clean mean error)")
    print(f"  test_f1                = {seg['f1']:.5f}")
    print(f"  test_precision         = {seg['precision']:.5f}")
    print(f"  test_recall            = {seg['recall']:.5f}")
    print(f"  test_accuracy          = {seg['accuracy']:.5f}")
    print(f"  test_balanced_accuracy = {seg['balanced_accuracy']:.5f}")
    print(f"  (burned n={seg['n_burned']}, clean n={seg['n_clean']}, thr={seg['threshold']:.5f})")

    # Pixel-level evaluation: dNBR > 0.1 mask (USGS) vs recon error map
    print("\nPixel-level segmentation (dNBR > 0.10 reference, USGS) ...")
    px_thr  = find_pixel_threshold(model, val_loader, cfg.device)
    px_seg  = pixel_segmentation_metrics(model, test_loader, cfg.device, threshold=px_thr)
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
    print(f"  test_pixel_f1          = {px_seg['pixel_f1']:.5f}")
    print(f"  test_pixel_iou         = {px_seg['pixel_iou']:.5f}")
    print(f"  test_pixel_precision   = {px_seg['pixel_precision']:.5f}")
    print(f"  test_pixel_recall      = {px_seg['pixel_recall']:.5f}")
    print(f"  test_pixel_accuracy    = {px_seg['pixel_accuracy']:.5f}")
    print(f"  (burned px={px_seg['pixel_n_burned']}, clean px={px_seg['pixel_n_clean']}, thr={px_thr:.6f})")

    # dNBR > 0.3 high-confidence subset (USGS high severity)
    auc_03 = proxy_auc(model, test_loader, cfg.device, dnbr_threshold=0.3,
                       desc="Proxy AUC (dNBR>0.3)")
    sep_03 = recon_separation(model, test_loader, cfg.device, dnbr_threshold=0.3)
    if auc_03 == auc_03:
        logger.log_metric("test_auc_dnbr_proxy_0.3", auc_03)
        print(f"  test_auc_dnbr_proxy_0.3 = {auc_03:.5f}")
    if sep_03 == sep_03:
        logger.log_metric("test_recon_separation_0.3", sep_03)
        print(f"  test_recon_separation_0.3 = {sep_03:+.5f}")

    # Otsu segmentation (threshold-free, no val-set calibration)
    otsu = otsu_segmentation_metrics(model, test_loader, cfg.device)
    logger.log_metrics({
        "test_otsu_f1":        otsu["otsu_f1"],
        "test_otsu_iou":       otsu["otsu_iou"],
        "test_otsu_kappa":     otsu["otsu_kappa"],
        "test_otsu_precision": otsu["otsu_precision"],
        "test_otsu_recall":    otsu["otsu_recall"],
        "test_otsu_threshold": otsu["otsu_threshold"],
    })
    print(f"  test_otsu_f1           = {otsu['otsu_f1']:.5f}")
    print(f"  test_otsu_iou          = {otsu['otsu_iou']:.5f}")
    print(f"  test_otsu_kappa        = {otsu['otsu_kappa']:.5f}")

    # 11. Figures -> MLflow artifacts
    print("\nGenerating figures ...")
    try:
        # a) Training curves
        epoch_metrics = trainer.epoch_metrics_history
        if epoch_metrics:
            fig_curves = plot_training_curves(epoch_metrics, run_name=run_name)
            _log_fig(fig_curves, "training_curves.png", artifact_path="plots")

        # b) Saliency figures (filtered to patches with burn signal)
        figs, figs_names = make_saliency_figures(model, val_loader, cfg.device, n_samples=6)
        log_figures_to_mlflow(figs, prefix="saliency", names=figs_names)
        print(f"  {len(figs)} saliency figures -> MLflow (saliency/)")

        # c) Reconstruction error distribution
        fig_dist = plot_recon_error_distribution(
            model, test_loader, cfg.device, run_name=run_name
        )
        _log_fig(fig_dist, "recon_error_distribution.png", artifact_path="plots")
        print("  recon error distribution -> MLflow (plots/)")

        # d) dNBR mask vs model prediction comparison
        cmp_figs, cmp_names = make_comparison_figures(
            model, test_loader, cfg.device,
            pixel_threshold=px_thr, n_samples=6,
        )
        log_figures_to_mlflow(cmp_figs, prefix="comparison", names=cmp_names)
        print(f"  {len(cmp_figs)} comparison figures -> MLflow (comparison/)")

    except Exception as exc:
        print(f"  [warning] Visualization skipped: {exc}")

    # 12. Save final model and artifacts
    model_path = os.path.join(cfg.output_root, f"{run_name}_model_final.pt")
    torch.save(model.state_dict(), model_path)

    for path in [split_path, model_path]:
        if os.path.exists(path):
            logger.log_artifact(path)

    best_ckpt = os.path.join(checkpoint_subdir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt):
        logger.log_artifact(best_ckpt)

    config_yaml = "configs/config.yaml"
    if os.path.exists(config_yaml):
        logger.log_artifact(config_yaml)

    # 12. Close run
    logger.end_run()

    print(f"\nExperiment complete -- {run_name}")
    print(f"  Model  : {model_path}")
    print(f"  Ckpts  : {checkpoint_subdir}")
    print(f"  Split  : {split_path}")

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run burnseg-xai experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lambda_rrr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--rrr_distance_metric", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--xai_terms", type=str, default=None,
                        help="Comma-separated XAI terms to activate: grad,gradcam,attn")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.lambda_rrr is not None:          cfg.lambda_rrr = args.lambda_rrr
    if args.epochs is not None:              cfg.epochs = args.epochs
    if args.lr is not None:                  cfg.lr = args.lr
    if args.batch_size is not None:          cfg.batch_size = args.batch_size
    if args.rrr_distance_metric is not None: cfg.rrr_distance_metric = args.rrr_distance_metric
    if args.seed is not None:                cfg.seed = args.seed
    if args.xai_terms is not None:           cfg.xai_terms = tuple(args.xai_terms.split(","))

    run(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
