#!/usr/bin/env bash
# run_ablation_single_term.sh: single-term XAI ablation vs three-term
#
# Tests each XAI term in isolation at lambda=0.1 cosine (best alignment config).
# The combined loss is divided by n_active (not always 3), so lambda has the
# same effective scale in single-term and three-term runs: results are
# directly comparable at the same lambda value.
#
# Usage (on pod):
#   bash /workspace/burnseg-xai/scripts/run_ablation_single_term.sh
#   # or in tmux: tmux new -s ablation && bash /workspace/burnseg-xai/scripts/run_ablation_single_term.sh

set -uo pipefail

PYTHON="/workspace/venv/bin/python3"
REPO="/workspace/burnseg-xai"
CONFIG="$REPO/configs/config_runpod.yaml"
LOGS="/workspace/experimento_queimadas/logs"
LAMBDA="0.1"
METRIC="cosine"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOGS"
cd "$REPO"

run_ablation() {
    local label="$1"; shift
    echo ""
    echo "========================================================"
    echo "ABLATION: $label: $(date)"
    echo "========================================================"
    if "$PYTHON" -m burnseg_xai.pipeline.run_experiment "$@"; then
        echo "OK: $label: $(date)"
    else
        echo "ERROR (exit $?): $label: continuing"
    fi
}

echo "========================================================"
echo "[ABLATION SUITE] Starting at $(date)"
echo "Config: lambda_rrr=$LAMBDA metric=$METRIC (single-term vs three-term)"
echo "========================================================"

# [1/3] Gradient saliency only (loss_grad_sal term)
run_ablation "grad-only" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --xai_terms "grad" \
    --run_name "ablation_grad_l${LAMBDA}_${METRIC}_seed43_${DATE}" \
    2>&1 | tee "$LOGS/ablation_grad_only.txt"

# [2/3] GradCAM only (loss_gradcam term)
run_ablation "gradcam-only" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --xai_terms "gradcam" \
    --run_name "ablation_gradcam_l${LAMBDA}_${METRIC}_seed43_${DATE}" \
    2>&1 | tee "$LOGS/ablation_gradcam_only.txt"

# [3/3] Attention gate only (loss_attn term)
run_ablation "attn-only" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --xai_terms "attn" \
    --run_name "ablation_attn_l${LAMBDA}_${METRIC}_seed43_${DATE}" \
    2>&1 | tee "$LOGS/ablation_attn_only.txt"

echo ""
echo "========================================================"
echo "[ABLATION SUITE] All runs complete at $(date)"
echo ""
echo "Compare val_saliency_cosine against the three-term baseline run in MLflow."
echo "========================================================"

echo ""
echo "--- [BACKUP] Saving results ---"
bash "$REPO/scripts/backup_pod_results.sh"
