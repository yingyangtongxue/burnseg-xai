"""
run_ablation.py: ablation study over lambda_rrr x rrr_distance_metric.

Each combination is a separate MLflow run within the same experiment.
All runs share the same seed and master split for fair comparison.

Default grid:
    lambda_rrr:          [0.0, 0.001, 0.01, 0.1, 1.0]
    rrr_distance_metric: ["mse", "cosine"]

Usage:
    python -m burnseg_xai.pipeline.run_ablation --config configs/config.yaml
    python -m burnseg_xai.pipeline.run_ablation --config configs/config.yaml \\
        --lambdas 0.0 0.01 0.1 --metrics mse cosine
"""

import argparse
import copy
from typing import List

from tqdm import tqdm

from burnseg_xai.config import load_config
from burnseg_xai.config.schema import ProjectConfig
from burnseg_xai.pipeline.run_experiment import run

# Default ablation grid
_LAMBDA_GRID  = [0.0, 0.001, 0.01, 0.1, 1.0]
_METRIC_GRID  = ["mse", "cosine"]


def run_ablation(
    cfg: ProjectConfig,
    lambdas: List[float] = _LAMBDA_GRID,
    metrics: List[str]   = _METRIC_GRID,
) -> None:
    """
    Runs all ablation combinations.  Baseline (lambda=0) is run once.
    All runs share the same seed and master split (split_master.json).
    """
    combos = []
    for lam in lambdas:
        for met in metrics:
            if lam == 0.0 and met != "mse":
                continue  # baseline doesn't depend on distance metric
            combos.append((lam, met))

    print(f"Ablation grid: {len(combos)} runs")
    for lam, met in combos:
        print(f"  lambda_rrr={lam:<6}  metric={met}")

    outer = tqdm(combos, desc="Ablation", unit="run", position=0)
    results = []

    for lam, met in outer:
        outer.set_postfix(lambda_rrr=lam, metric=met)

        run_cfg = copy.deepcopy(cfg)
        run_cfg.lambda_rrr = lam
        run_cfg.rrr_distance_metric = met

        rname = (
            f"ablation_l{lam}_{met}_seed{cfg.seed}"
            if lam > 0.0
            else f"ablation_baseline_seed{cfg.seed}"
        )
        tqdm.write(f"\n{'-'*60}\n[Ablation] {rname}\n{'-'*60}")

        metrics_out = run(run_cfg, run_name=rname)
        results.append({"run_name": rname, "lambda_rrr": lam,
                        "metric": met, **metrics_out})

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'run':<45} {'val_loss':>10} {'val_cos':>10}")
    print("=" * 70)
    for r in results:
        print(
            f"{r['run_name']:<45} "
            f"{r.get('val_loss', float('nan')):>10.5f} "
            f"{r.get('val_saliency_cosine', float('nan')):>10.5f}"
        )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=_LAMBDA_GRID,
        help="lambda_rrr values to sweep",
    )
    parser.add_argument(
        "--metrics", nargs="+", type=str, default=_METRIC_GRID,
        help="rrr_distance_metric values to sweep",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_ablation(cfg, lambdas=args.lambdas, metrics=args.metrics)


if __name__ == "__main__":
    main()
