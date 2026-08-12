#!/usr/bin/env bash
# runpod_setup_s3.sh: Setup for pods WITHOUT a network volume.
#
# Use this instead of runpod_setup.sh when running in any region other than
# eu-ro-1 (where the network volume lives). Downloads dataset and split from
# S3, installs packages, and installs burnseg-xai from the cloned repo.
#
# Prerequisites on the pod:
#   - aws CLI installed (available in most RunPod templates)
#   - AWS credentials configured for profile "runpod":
#       mkdir -p ~/.aws
#       cat > ~/.aws/credentials << 'EOF'
#       [runpod]
#       aws_access_key_id     = YOUR_KEY
#       aws_secret_access_key = YOUR_SECRET
#       EOF
#   - Repo already in /workspace/burnseg-xai (see step 1 below)
#
# Step 1: get the repo on the pod (choose one):
#   Option A (recommended): copy from local machine:
#       rsync -avz --exclude='.git' \
#           /path/to/burnseg-xai/ root@<POD_IP>:/workspace/burnseg-xai/ \
#           -e "ssh -p <PORT> -i ~/.ssh/id_ed25519"
#   Option B: clone from GitHub if repo is private/public there:
#       git clone https://github.com/<user>/burnseg-xai /workspace/burnseg-xai
#
# Step 2: run this script:
#   bash /workspace/burnseg-xai/scripts/runpod_setup_s3.sh
#
# Step 3: run experiments:
#   tmux new -s all && bash /workspace/burnseg-xai/scripts/run_all_parallel.sh

set -e

S3_BUCKET="s3://3faohy6bsm"
AWS_PROFILE="runpod"
WORKSPACE="/workspace"
REPO="$WORKSPACE/burnseg-xai"
PYTHON="$WORKSPACE/venv/bin/python3"

echo "========================================================"
echo "[SETUP S3] Starting at $(date)"
echo "========================================================"

# 1. Create venv if needed (some RunPod templates don't have one at /workspace/venv)
if [ ! -f "$PYTHON" ]; then
    echo "[1/6] Creating venv at $WORKSPACE/venv ..."
    python3 -m venv "$WORKSPACE/venv"
else
    echo "[1/6] Venv already exists at $PYTHON"
fi

# 2. Download dataset from S3
echo ""
echo "[2/6] Downloading dataset from S3 ..."
echo "      Source: $S3_BUCKET/dataset_mestrado/"
echo "      Dest:   $WORKSPACE/dataset_mestrado/"
aws s3 sync "$S3_BUCKET/dataset_mestrado/" "$WORKSPACE/dataset_mestrado/" \
    --profile "$AWS_PROFILE" --endpoint-url https://s3api-eu-ro-1.runpod.io --no-progress

# 3. Download split_master.json
echo ""
echo "[3/6] Downloading split_master.json ..."
aws s3 cp "$S3_BUCKET/split_master.json" "$WORKSPACE/split_master.json" \
    --profile "$AWS_PROFILE" --endpoint-url https://s3api-eu-ro-1.runpod.io

# 4. Install Python dependencies
echo ""
echo "[4/6] Installing dependencies ..."
TMPDIR="$WORKSPACE" "$WORKSPACE/venv/bin/pip" install \
    rasterio mlflow matplotlib scikit-learn scikit-image scipy tqdm PyYAML \
    -q --ignore-installed blinker

# 5. Install burnseg-xai (editable, no deps: torch/CUDA already in template)
echo ""
echo "[5/6] Installing burnseg-xai ..."
"$WORKSPACE/venv/bin/pip" install -e "$REPO" --no-deps -q

# 6. Create output directories and copy split
echo ""
echo "[6/6] Creating output directories ..."
mkdir -p "$WORKSPACE/experimento_queimadas/checkpoints"
mkdir -p "$WORKSPACE/experimento_queimadas/mlruns"
mkdir -p "$WORKSPACE/experimento_queimadas/logs"
cp "$WORKSPACE/split_master.json" "$WORKSPACE/experimento_queimadas/split_master.json"

# Verify
echo ""
echo "=== Verification ==="
"$PYTHON" -c "
import glob, torch
tifs = glob.glob('/workspace/dataset_mestrado/**/*.tif', recursive=True)
print(f'  Patches found: {len(tifs)}')
assert len(tifs) > 5000, f'Dataset incomplete: only {len(tifs)} patches'
n_gpus = torch.cuda.device_count()
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPUs detected:  {n_gpus}')
for i in range(n_gpus):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
import burnseg_xai
print('  burnseg_xai:   OK')
"

echo ""
echo "========================================================"
echo "[SETUP S3] Done at $(date)"
echo "Next: tmux new -s all"
echo "      bash $REPO/scripts/run_all_parallel.sh"
echo "========================================================"
