#!/usr/bin/env bash
# Run locally (Windows/bash) BEFORE starting the RunPod pod.
# Uploads the code and split index to the RunPod Network Volume via S3.
# The dataset upload (dataset_mestrado/) is handled separately with aws s3 cp.
#
# Usage: bash scripts/upload_to_runpod.sh

set -e

BUCKET="s3://3faohy6bsm"
PROFILE="--profile runpod --endpoint-url https://s3api-eu-ro-1.runpod.io"

echo "=== Uploading split_master.json ==="
aws s3 cp "./outputs/split_master.json" \
    "$BUCKET/split_master.json" $PROFILE

echo "=== Uploading burnseg-xai code ==="
aws s3 cp --recursive \
    "." \
    "$BUCKET/burnseg-xai/" $PROFILE \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --exclude ".git/*"

echo ""
echo "=== Upload complete. ==="
echo "Start the pod, then run:"
echo "  bash /workspace/data/burnseg-xai/scripts/runpod_setup.sh"
echo "  bash /workspace/data/burnseg-xai/scripts/run_experiments_runpod.sh"
