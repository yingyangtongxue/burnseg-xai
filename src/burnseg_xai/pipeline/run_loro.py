"""
run_loro.py -- Leave-One-Region-Out (LORO) cross-validation.

Fold types
----------
Standard LORO (4 folds)
    Each fold holds out one of the four regions for test; trains on the other three.
    Regions: yanomami, kayapo, karipuna, parna_chapada_dos_guimaraes

Cross-biome fold (1 fold)
    Train exclusively on all three Amazonian regions; test on Cerrado (Chapada).
    Evaluates biome-level generalisation across a domain shift.
    The model has never seen Cerrado vegetation during training.

Patch reproducibility
---------------------
A patch manifest is written to ``output_root/patch_manifest.json`` before any
fold runs.  Every fold then saves its own split file for auditability.
All splits are deterministic given the same dataset ordering and seed.

Usage
-----
    python -m burnseg_xai.pipeline.run_loro --config configs/config.yaml
    python -m burnseg_xai.pipeline.run_loro --config configs/config.yaml \\
        --lambda_rrr 0.1 --regions yanomami karipuna
    # cross-biome only:
    python -m burnseg_xai.pipeline.run_loro --config configs/config.yaml \\
        --cross_biome_only
"""

import argparse
import json
import os
from datetime import date
from typing import List, Optional, Tuple

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from burnseg_xai.config import load_config
from burnseg_xai.config.schema import ProjectConfig
from burnseg_xai.dataset import BurnedAreaDataset
from burnseg_xai.evaluator import (
    evaluate,
    find_optimal_threshold,
    find_pixel_threshold,
    pixel_segmentation_metrics,
    proxy_auc,
    recon_separation,
    segmentation_metrics,
)
from burnseg_xai.logging.mlflow_logger import MLflowLogger
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.sanity_checks import run_sanity_checks
from burnseg_xai.split import (
    build_patch_manifest,
    create_cross_biome_split,
    create_loro_split,
    save_split,
)
from burnseg_xai.training.trainer import Trainer
from burnseg_xai.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Single fold runner
# ---------------------------------------------------------------------------

def run_loro_fold(
    cfg: ProjectConfig,
    dataset: BurnedAreaDataset,
    test_label: str,
    run_name: str,
    split: Tuple[List[int], List[int], List[int]],
    fold_type: str = "loro",
) -> dict:
    """
    Trains and evaluates a single LORO-style fold.

    Parameters
    ----------
    test_label : region name (LORO folds) or biome name (cross-biome fold)
    split      : pre-computed (train_idx, val_idx, test_idx) into ``dataset``
    fold_type  : 'loro' | 'cross_biome'
    """
    set_seed(cfg.seed)

    train_idx, val_idx, test_idx = split

    if len(test_idx) == 0:
        tqdm.write(
            f"[LORO] No patches for '{test_label}'. "
            "Verify region/biome names in filenames. Skipping."
        )
        return {}

    split_path = os.path.join(
        cfg.output_root,
        f"split_{fold_type}_{test_label}.json",
    )
    os.makedirs(cfg.output_root, exist_ok=True)
    save_split({"train": train_idx, "val": val_idx, "test": test_idx}, split_path)

    tqdm.write(
        f"[{fold_type.upper()} test={test_label}]  "
        f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )

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

    model     = Autoencoder(in_channels=cfg.in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    checkpoint_subdir = os.path.join(cfg.checkpoint_dir, run_name)
    os.makedirs(checkpoint_subdir, exist_ok=True)

    logger       = MLflowLogger(cfg)
    is_resuming  = False
    latest_ckpt  = os.path.join(checkpoint_subdir, "checkpoint_latest.pt")

    if os.path.exists(latest_ckpt):
        saved = torch.load(latest_ckpt, map_location="cpu")
        if saved.get("mlflow_run_id"):
            logger.resume_run(saved["mlflow_run_id"])
            is_resuming = True

    if not is_resuming:
        mlflow_run_id = logger.start_run(run_name=run_name)
        logger.set_tag(
            "description",
            f"{fold_type} test={test_label} lambda={cfg.lambda_rrr} "
            f"metric={cfg.rrr_distance_metric}",
        )
        logger.log_params({
            "batch_size":              cfg.batch_size,
            "epochs":                  cfg.epochs,
            "lr":                      cfg.lr,
            "optimizer":               "adam",
            "seed":                    cfg.seed,
            "temporal_length":         cfg.temporal_length,
            "in_channels":             cfg.in_channels,
            "lambda_rrr":              cfg.lambda_rrr,
            "rrr_distance_metric":     cfg.rrr_distance_metric,
            "normalization":           "zscore_per_patch",
            "model_arch":              "3dcnn_autoencoder",
            "dataset_root":            cfg.dataset_root,
            "fold_type":               fold_type,
            "test_label":              test_label,
            "n_train":                 len(train_idx),
            "n_val":                   len(val_idx),
            "n_test":                  len(test_idx),
            "early_stopping_patience": cfg.early_stopping_patience,
        })
    else:
        mlflow_run_id = mlflow.active_run().info.run_id

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=cfg.device,
        logger=logger,
        lambda_rrr=cfg.lambda_rrr,
        rrr_distance_metric=cfg.rrr_distance_metric,
        checkpoint_dir=checkpoint_subdir,
        early_stopping_patience=cfg.early_stopping_patience,
        mlflow_run_id=mlflow_run_id,
    )

    final_metrics = trainer.train(train_loader, val_loader, cfg.epochs)

    # Test evaluation
    test_metrics = evaluate(
        model, test_loader, cfg.device,
        desc=f"Test [{test_label}]",
    )
    logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    auc = proxy_auc(model, test_loader, cfg.device, desc=f"AUC [{test_label}]")
    if auc == auc:
        logger.log_metric("test_auc_dnbr_proxy", auc)

    sep = recon_separation(model, test_loader, cfg.device)
    if sep == sep:
        logger.log_metric("test_recon_separation", sep)

    # Patch-level segmentation metrics (threshold from val set -> applied to test)
    opt_thr = find_optimal_threshold(model, val_loader, cfg.device)
    seg = segmentation_metrics(
        model, test_loader, cfg.device, threshold=opt_thr,
        desc=f"Seg [{test_label}]",
    )
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

    # Pixel-level segmentation vs dNBR mask (USGS threshold)
    px_thr = find_pixel_threshold(model, val_loader, cfg.device)
    px_seg = pixel_segmentation_metrics(
        model, test_loader, cfg.device, threshold=px_thr,
        desc=f"Pixel [{test_label}]",
    )
    logger.log_metrics({
        "test_pixel_f1":        px_seg["pixel_f1"],
        "test_pixel_iou":       px_seg["pixel_iou"],
        "test_pixel_precision": px_seg["pixel_precision"],
        "test_pixel_recall":    px_seg["pixel_recall"],
        "test_pixel_accuracy":  px_seg["pixel_accuracy"],
        "test_pixel_threshold": px_thr,
        "test_pixel_n_burned":  float(px_seg["pixel_n_burned"]),
        "test_pixel_n_clean":   float(px_seg["pixel_n_clean"]),
    })

    # Artifacts
    model_path = os.path.join(cfg.output_root, f"{run_name}_model_final.pt")
    torch.save(model.state_dict(), model_path)
    for path in [split_path, model_path]:
        if os.path.exists(path):
            logger.log_artifact(path)
    best_ckpt = os.path.join(checkpoint_subdir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt):
        logger.log_artifact(best_ckpt)

    logger.end_run()

    return {
        "fold_type":             fold_type,
        "test_label":            test_label,
        "val_saliency_cosine":   final_metrics.get("val_saliency_cosine", 0.0),
        "val_loss":              final_metrics.get("val_loss", float("nan")),
        "test_saliency_cosine":  test_metrics.get("saliency_cosine", 0.0),
        "test_recon_error_mean": test_metrics.get("recon_error_mean", float("nan")),
        "test_auc":              auc,
        "test_recon_separation": sep if sep == sep else float("nan"),
        "test_f1":               seg["f1"],
        "test_precision":        seg["precision"],
        "test_recall":           seg["recall"],
        "test_accuracy":         seg["accuracy"],
        "test_balanced_acc":     seg["balanced_accuracy"],
        "test_n_burned":         seg["n_burned"],
        "test_n_clean":          seg["n_clean"],
        "test_pixel_f1":         px_seg["pixel_f1"],
        "test_pixel_iou":        px_seg["pixel_iou"],
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_loro(
    cfg: ProjectConfig,
    regions: Optional[List[str]] = None,
    include_cross_biome: bool = True,
    cross_biome_only: bool = False,
) -> None:
    """
    Runs all LORO folds (and optionally the cross-biome fold) then prints a
    summary table.

    Parameters
    ----------
    regions             : list of region names to hold out; defaults to cfg.regions
    include_cross_biome : append a cross-biome (Amazonia -> Cerrado) fold
    cross_biome_only    : skip all standard LORO folds; run only cross-biome
    """
    if regions is None:
        regions = cfg.regions

    dataset = BurnedAreaDataset(
        root_dir=cfg.dataset_root,
        temporal_length=cfg.temporal_length,
    )
    run_sanity_checks(dataset, cfg)

    # Save full patch manifest before any fold runs
    os.makedirs(cfg.output_root, exist_ok=True)
    manifest_path = os.path.join(cfg.output_root, "patch_manifest.json")
    if not os.path.exists(manifest_path):
        manifest = build_patch_manifest(dataset)
        with open(manifest_path, "w") as f:
            json.dump({"created": date.today().isoformat(), "patches": manifest}, f, indent=2)
        print(f"[manifest] Saved {len(manifest)} patches -> {manifest_path}")
    else:
        print(f"[manifest] Loaded existing manifest from {manifest_path}")

    today        = date.today().isoformat()
    fold_results = []

    # ---- Standard LORO folds ------------------------------------------------
    if not cross_biome_only:
        outer = tqdm(regions, desc="LORO folds", unit="fold", position=0)
        for test_region in outer:
            outer.set_postfix(region=test_region)
            tqdm.write(f"\n{'-'*60}\n[LORO] test_region={test_region}\n{'-'*60}")

            split = create_loro_split(dataset, test_region=test_region, seed=cfg.seed)
            rname = (
                f"loro_{test_region}_l{cfg.lambda_rrr}_{cfg.rrr_distance_metric}"
                f"_seed{cfg.seed}_{today}"
            )
            result = run_loro_fold(
                cfg, dataset,
                test_label=test_region,
                run_name=rname,
                split=split,
                fold_type="loro",
            )
            if result:
                fold_results.append(result)

    # ---- Cross-biome fold ---------------------------------------------------
    if include_cross_biome or cross_biome_only:
        tqdm.write(
            f"\n{'='*60}\n"
            "[cross_biome] train=amazonia  test=cerrado (parna_chapada_dos_guimaraes)\n"
            f"{'='*60}"
        )
        split = create_cross_biome_split(dataset, test_biome="cerrado", seed=cfg.seed)
        rname = (
            f"cross_biome_cerrado_l{cfg.lambda_rrr}_{cfg.rrr_distance_metric}"
            f"_seed{cfg.seed}_{today}"
        )
        result = run_loro_fold(
            cfg, dataset,
            test_label="cerrado",
            run_name=rname,
            split=split,
            fold_type="cross_biome",
        )
        if result:
            fold_results.append(result)

    # ---- Summary table ------------------------------------------------------
    if not fold_results:
        print("No folds completed.")
        return

    print("\n" + "=" * 110)
    print(
        f"{'fold_type':<14} {'test_label':<28} {'val_cos':>7} "
        f"{'test_cos':>8} {'patch_f1':>8} {'px_f1':>7} {'px_iou':>7} {'auc':>7}"
    )
    print("=" * 110)
    for r in fold_results:
        def _fmt(v): return f"{v:.4f}" if v == v else "  nan "
        print(
            f"{r['fold_type']:<14} "
            f"{r['test_label']:<28} "
            f"{r['val_saliency_cosine']:>7.4f} "
            f"{r['test_saliency_cosine']:>8.4f} "
            f"{_fmt(r['test_f1']):>8} "
            f"{_fmt(r['test_pixel_f1']):>7} "
            f"{_fmt(r['test_pixel_iou']):>7} "
            f"{_fmt(r['test_auc']):>7}"
        )

    loro_results = [r for r in fold_results if r["fold_type"] == "loro"]
    if loro_results:
        macro_cos    = sum(r["test_saliency_cosine"] for r in loro_results) / len(loro_results)
        macro_pxf1   = sum(r["test_pixel_f1"] for r in loro_results if r["test_pixel_f1"] == r["test_pixel_f1"]) / max(len(loro_results), 1)
        print("-" * 110)
        print(f"{'loro macro-avg':<14} {'':>28} {'':>7} {macro_cos:>8.4f} {'':>8} {macro_pxf1:>7.4f}")

    print("=" * 110)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LORO cross-validation")
    parser.add_argument("--config",              type=str,  default=None)
    parser.add_argument("--lambda_rrr",          type=float, default=None)
    parser.add_argument("--rrr_distance_metric", type=str,  default=None)
    parser.add_argument(
        "--regions", nargs="+", default=None,
        help="Regions to use as LORO test sets (default: all in config)",
    )
    parser.add_argument(
        "--cross_biome_only", action="store_true",
        help="Skip standard LORO folds; run only the cross-biome fold",
    )
    parser.add_argument(
        "--no_cross_biome", action="store_true",
        help="Skip the cross-biome fold",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.lambda_rrr is not None:
        cfg.lambda_rrr = args.lambda_rrr
    if args.rrr_distance_metric is not None:
        cfg.rrr_distance_metric = args.rrr_distance_metric

    run_loro(
        cfg,
        regions=args.regions,
        include_cross_biome=not args.no_cross_biome,
        cross_biome_only=args.cross_biome_only,
    )


if __name__ == "__main__":
    main()
