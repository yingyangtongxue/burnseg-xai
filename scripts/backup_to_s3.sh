#!/usr/bin/env bash
# backup_to_s3.sh: Upload experiment results to S3.
#
# Use after experiments complete on a pod WITHOUT a network volume.
# Safe to call multiple times (aws s3 sync is incremental).
# Also useful mid-run to checkpoint progress before the pod auto-terminates.
#
# Usage:
#   bash /workspace/burnseg-xai/scripts/backup_to_s3.sh

set -e

S3_BUCKET="s3://3faohy6bsm"
AWS_PROFILE="runpod"
SRC="/workspace/experimento_queimadas"

echo "========================================================"
echo "[BACKUP → S3] Starting at $(date)"
echo "Destination: $S3_BUCKET/experimento_queimadas/"
echo "========================================================"

echo ""
echo "[1/4] Uploading checkpoints ..."
S3_OPTS="--profile $AWS_PROFILE --endpoint-url https://s3api-eu-ro-1.runpod.io --no-progress"

aws s3 sync "$SRC/checkpoints/" "$S3_BUCKET/experimento_queimadas/checkpoints/" $S3_OPTS

echo ""
echo "[2/4] Uploading MLflow tracking ..."
aws s3 sync "$SRC/mlruns/" "$S3_BUCKET/experimento_queimadas/mlruns/" $S3_OPTS

echo ""
echo "[3/4] Uploading logs ..."
aws s3 sync "$SRC/logs/" "$S3_BUCKET/experimento_queimadas/logs/" $S3_OPTS

echo ""
echo "[4/4] Uploading final model weights (*.pt at output root) ..."
aws s3 sync "$SRC/" "$S3_BUCKET/experimento_queimadas/" $S3_OPTS \
    --exclude "*" --include "*.pt"

echo ""
echo "========================================================"
echo "[BACKUP → S3] Done at $(date)"
echo ""
echo "To download results locally:"
echo "  aws s3 sync $S3_BUCKET/experimento_queimadas/ ./outputs/ --profile runpod"
echo "========================================================"
