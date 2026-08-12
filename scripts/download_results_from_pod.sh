#!/usr/bin/env bash
# Runs LOCALLY on Windows (Git Bash).
# Downloads checkpoints and MLflow results from the RunPod Network Volume (S3)
# to the local machine at ./outputs/.
#
# Usage:
#   bash scripts/download_results_from_pod.sh [TIMESTAMP]
#
# If TIMESTAMP is given, downloads only that backup snapshot:
#   bash scripts/download_results_from_pod.sh 20240501_143000
#
# If TIMESTAMP is omitted, downloads the full backups/ directory.

set -eo pipefail

BUCKET="s3://3faohy6bsm"
PROFILE="--profile runpod --endpoint-url https://s3api-eu-ro-1.runpod.io"
LOCAL_DST="./outputs"
TIMESTAMP="${1:-}"

mkdir -p "$LOCAL_DST"

echo "============================================================"
echo "Downloading RunPod results to: $LOCAL_DST"
echo "============================================================"
echo ""

if [ -n "$TIMESTAMP" ]; then
    # Download a specific backup snapshot
    REMOTE="$BUCKET/backups/$TIMESTAMP"
    echo "Snapshot: $TIMESTAMP"
    echo ""

    echo "[1/3] Downloading checkpoints (weights)..."
    aws s3 cp --recursive \
        "$REMOTE/checkpoints/" \
        "$LOCAL_DST/checkpoints/" \
        $PROFILE \
        || echo "      No checkpoints in this snapshot."

    echo ""
    echo "[2/3] Downloading MLflow (mlruns)..."
    aws s3 cp --recursive \
        "$REMOTE/mlruns/" \
        "$LOCAL_DST/mlruns/" \
        $PROFILE \
        || echo "      No mlruns in this snapshot."

    echo ""
    echo "[3/3] Downloading logs..."
    aws s3 cp --recursive \
        "$REMOTE/logs/" \
        "$LOCAL_DST/logs/" \
        $PROFILE \
        || echo "      No logs in this snapshot."

    # Also try the single-file archive (faster if available)
    echo ""
    echo "[+] Checking for compressed archive..."
    aws s3 cp \
        "$REMOTE/results.tar.gz" \
        "$LOCAL_DST/results_${TIMESTAMP}.tar.gz" \
        $PROFILE \
        && echo "    Archive downloaded: $LOCAL_DST/results_${TIMESTAMP}.tar.gz" \
        || echo "    (No archive file found.)"

else
    # Download everything from the live experiment directory
    echo "No TIMESTAMP given: downloading live experiment directory."
    echo ""

    echo "[1/2] Downloading checkpoints (weights)..."
    aws s3 cp --recursive \
        "$BUCKET/experimento_queimadas/checkpoints/" \
        "$LOCAL_DST/checkpoints/" \
        $PROFILE

    echo ""
    echo "[2/2] Downloading MLflow (mlruns)..."
    aws s3 cp --recursive \
        "$BUCKET/experimento_queimadas/mlruns/" \
        "$LOCAL_DST/mlruns/" \
        $PROFILE
fi

echo ""
echo "============================================================"
echo "Download complete: $LOCAL_DST"
echo ""
echo "To browse MLflow locally:"
echo "  mlflow ui --backend-store-uri ./outputs/mlruns"
echo "============================================================"
