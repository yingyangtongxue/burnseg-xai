#!/usr/bin/env bash
# Sequential full-experiment runner for burned-area detection.
# Run from the repo root:
#   bash run_full_experiments.sh 2>&1 | tee ./outputs/full_run_log.txt
#
# Experiments in order:
#   1. Baseline  (lambda=0)   -- no XAI regularization
#   2. RRR mse   (lambda=0.1) -- proposed method: RRR with MSE distance
#   3. RRR cos   (lambda=0.1) -- variant: RRR with cosine distance
#   4. LORO + cross-biome     -- spatial and biome generalization

set -e

REPO="."
CONFIG="$REPO/configs/config.yaml"

echo "========================================================"
echo "[SUITE] Full experiment suite starting at $(date)"
echo "========================================================"

echo ""
echo "--- [1/4] BASELINE (lambda=0) ---"
echo "Start: $(date)"
python -m burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" \
    --lambda_rrr 0.0
echo "Done: $(date)"

echo ""
echo "--- [2/4] RRR-MSE (lambda=0.1, metric=mse) ---"
echo "Start: $(date)"
python -m burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" \
    --lambda_rrr 0.1 \
    --rrr_distance_metric mse
echo "Done: $(date)"

echo ""
echo "--- [3/4] RRR-COSINE (lambda=0.1, metric=cosine) ---"
echo "Start: $(date)"
python -m burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" \
    --lambda_rrr 0.1 \
    --rrr_distance_metric cosine
echo "Done: $(date)"

echo ""
echo "--- [4/4] LORO + CROSS-BIOME (lambda=0.1) ---"
echo "Start: $(date)"
python -m burnseg_xai.pipeline.run_loro \
    --config "$CONFIG" \
    --lambda_rrr 0.1 \
    --rrr_distance_metric mse
echo "Done: $(date)"

echo ""
echo "========================================================"
echo "[SUITE] All experiments complete at $(date)"
echo "========================================================"
