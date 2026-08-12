#!/usr/bin/env bash
# Runs INSIDE the RunPod pod.
# Verifies that checkpoints and MLflow data are on the Network Volume (/workspace/)
# and creates a compressed archive as extra safety before stopping the pod.
#
# Usage (inside pod):
#   bash /workspace/burnseg-xai/scripts/backup_pod_results.sh

set -e

SRC="/workspace/experimento_queimadas"
ARCHIVE="/workspace/results_backup.tar.gz"

echo "=== RunPod backup: $(date) ==="
echo ""

# Verify key files are on the volume
echo "[1] Checking checkpoints on volume..."
if compgen -G "$SRC/checkpoints/*/checkpoint_best.pt" > /dev/null 2>&1; then
    ls -lh "$SRC"/checkpoints/*/checkpoint_best.pt
else
    echo "WARNING: no checkpoint_best.pt found: experiment may not have completed an epoch yet."
fi

echo ""
echo "[2] Checking MLflow on volume..."
if [ -d "$SRC/mlruns" ]; then
    du -sh "$SRC/mlruns"
    echo "    MLflow OK."
else
    echo "WARNING: mlruns not found."
fi

echo ""
echo "[3] Creating compressed archive on volume..."
tar -czf "$ARCHIVE" \
    -C "/workspace" \
    "experimento_queimadas/checkpoints" \
    "experimento_queimadas/mlruns" \
    "experimento_queimadas/logs" \
    2>/dev/null || \
tar -czf "$ARCHIVE" -C "/workspace" "experimento_queimadas"

du -sh "$ARCHIVE"

echo ""
echo "=== Done. Safe to stop the pod. ==="
echo "    Archive: $ARCHIVE"
echo "    The Network Volume persists after the pod is stopped."
