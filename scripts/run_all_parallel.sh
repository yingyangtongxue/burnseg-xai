#!/usr/bin/env bash
# run_all_parallel.sh: 7 experiments across 5 GPUs
#
# Job assignment (5 GPUs, 7 jobs):
#   GPU 0  LORO cross-biome / leave-chapada
#   GPU 1  LORO leave-yanomami                 (smallest train set ~1007 patches → finishes first)
#           └→ Ablation attn-only              (reuses GPU 1 after yanomami finishes)
#   GPU 2  LORO leave-kayapo
#   GPU 3  LORO leave-karipuna
#   GPU 4  Ablation grad-only   (λ=0.1 cosine)
#           └→ Ablation gradcam-only            (sequential on GPU 4 after grad-only)
#
# Wall time: ~4 h.
# Total GPU-hours: ~14 h (7 jobs × ~2 h each).
#
# Prerequisites:
#   bash /workspace/burnseg-xai/scripts/runpod_setup_s3.sh
#
# Usage:
#   bash /workspace/burnseg-xai/scripts/run_all_parallel.sh
#   # always run inside tmux: tmux new -s all && bash .../run_all_parallel.sh

set -uo pipefail

PYTHON="/workspace/venv/bin/python3"
REPO="/workspace/burnseg-xai"
CONFIG="$REPO/configs/config_runpod.yaml"
LOGS="/workspace/experimento_queimadas/logs"
LAMBDA_LORO="0.1"
METRIC_LORO="mse"
LAMBDA_ABL="0.1"
METRIC_ABL="cosine"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOGS"
cd "$REPO"

# Auto-backup: trap + periodic
# On SIGTERM/SIGINT/EXIT the script uploads all checkpoints/mlruns/logs to S3.
# Handles pod preemption, billing stops, and manual interruptions.
_backup_and_exit() {
    echo ""
    echo "[AUTO-BACKUP] $(date): uploading checkpoints/mlruns/logs to S3 ..."
    kill $PERIODIC_PID 2>/dev/null
    bash "$REPO/scripts/backup_to_s3.sh" || echo "[AUTO-BACKUP] WARNING: S3 upload failed"
    echo "[AUTO-BACKUP] Done."
}
trap '_backup_and_exit' SIGTERM SIGINT EXIT

# Periodic backup every 30 minutes in the background
(
  while true; do
    sleep 1800
    echo "[PERIODIC BACKUP] $(date)"
    bash "$REPO/scripts/backup_to_s3.sh" || echo "[PERIODIC BACKUP] WARNING: upload failed"
  done
) &
PERIODIC_PID=$!

echo "========================================================"
echo "[ALL PARALLEL] Starting at $(date)"
echo "LORO:    λ=$LAMBDA_LORO metric=$METRIC_LORO"
echo "Ablation: λ=$LAMBDA_ABL metric=$METRIC_ABL"
echo "Auto-backup: every 30 min + on SIGTERM/EXIT → S3"
echo "========================================================"

# LORO folds, GPUs 0-3

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m burnseg_xai.pipeline.run_loro \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_LORO" \
    --rrr_distance_metric "$METRIC_LORO" \
    --cross_biome_only \
    > "$LOGS/loro_cross_biome.txt" 2>&1 &
PID_XBIOME=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m burnseg_xai.pipeline.run_loro \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_LORO" \
    --rrr_distance_metric "$METRIC_LORO" \
    --regions yanomami \
    --no_cross_biome \
    > "$LOGS/loro_yanomami.txt" 2>&1 &
PID_YANO=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m burnseg_xai.pipeline.run_loro \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_LORO" \
    --rrr_distance_metric "$METRIC_LORO" \
    --regions kayapo \
    --no_cross_biome \
    > "$LOGS/loro_kayapo.txt" 2>&1 &
PID_KAYA=$!

CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m burnseg_xai.pipeline.run_loro \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_LORO" \
    --rrr_distance_metric "$METRIC_LORO" \
    --regions karipuna \
    --no_cross_biome \
    > "$LOGS/loro_karipuna.txt" 2>&1 &
PID_KARI=$!

# Ablation, GPUs 4-5 (attn waits for GPU 1)

CUDA_VISIBLE_DEVICES=4 "$PYTHON" -m burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_ABL" \
    --rrr_distance_metric "$METRIC_ABL" \
    --xai_terms "grad" \
    --run_name "ablation_grad_l${LAMBDA_ABL}_${METRIC_ABL}_seed43_${DATE}" \
    > "$LOGS/ablation_grad.txt" 2>&1 &
PID_GRAD=$!

echo ""
echo "GPU 0 → LORO cross-biome    PID=$PID_XBIOME  log: loro_cross_biome.txt"
echo "GPU 1 → LORO leave-yanomami PID=$PID_YANO    log: loro_yanomami.txt  (→ attn after)"
echo "GPU 2 → LORO leave-kayapo   PID=$PID_KAYA    log: loro_kayapo.txt"
echo "GPU 3 → LORO leave-karipuna PID=$PID_KARI    log: loro_karipuna.txt"
echo "GPU 4 → ablation grad-only  PID=$PID_GRAD    log: ablation_grad.txt  (→ gradcam after)"
echo ""
echo "Waiting for leave-yanomami (GPU 1, ~1007 train patches: finishes first)..."
echo ""

# yanomami has the smallest training set; GPU 1 frees up first
wait $PID_YANO
YANO_EXIT=$?
[ $YANO_EXIT -ne 0 ] && echo "WARNING: leave-yanomami exit $YANO_EXIT: check loro_yanomami.txt"
echo "GPU 1 free at $(date): launching ablation attn-only"

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_ABL" \
    --rrr_distance_metric "$METRIC_ABL" \
    --xai_terms "attn" \
    --run_name "ablation_attn_l${LAMBDA_ABL}_${METRIC_ABL}_seed43_${DATE}" \
    > "$LOGS/ablation_attn.txt" 2>&1 &
PID_ATTN=$!

echo "Waiting for grad-only (GPU 4) to free up for gradcam..."
wait $PID_GRAD
GRAD_EXIT=$?
[ $GRAD_EXIT -ne 0 ] && echo "WARNING: ablation grad exit $GRAD_EXIT: check ablation_grad.txt"
echo "GPU 4 free at $(date): launching ablation gradcam-only"

CUDA_VISIBLE_DEVICES=4 "$PYTHON" -m burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA_ABL" \
    --rrr_distance_metric "$METRIC_ABL" \
    --xai_terms "gradcam" \
    --run_name "ablation_gradcam_l${LAMBDA_ABL}_${METRIC_ABL}_seed43_${DATE}" \
    > "$LOGS/ablation_gradcam.txt" 2>&1 &
PID_GCAM=$!

echo "Waiting for all remaining jobs..."
wait $PID_XBIOME && echo "cross-biome OK"  || echo "WARNING: cross-biome failed"
wait $PID_KAYA   && echo "leave-kayapo OK" || echo "WARNING: leave-kayapo failed"
wait $PID_KARI   && echo "leave-karipuna OK" || echo "WARNING: leave-karipuna failed"
wait $PID_ATTN   && echo "ablation attn OK" || echo "WARNING: ablation attn failed"
wait $PID_GCAM   && echo "ablation gradcam OK" || echo "WARNING: ablation gradcam failed"

kill $PERIODIC_PID 2>/dev/null  # stop periodic backups: EXIT trap will do the final one

echo ""
echo "========================================================"
echo "[ALL PARALLEL] All 7 jobs complete at $(date)"
echo "========================================================"
echo ""
echo "Log summary:"
for f in "$LOGS"/loro_*.txt "$LOGS"/ablation_*.txt; do
    [ -f "$f" ] || continue
    tail_line=$(tail -1 "$f")
    printf "  %-40s %s\n" "$(basename "$f")" "$tail_line"
done
# EXIT trap fires here and uploads everything to S3 automatically
