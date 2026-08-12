#!/bin/bash
# Run inside the RunPod pod. The Network Volume is mounted at /workspace/.
# Code is at /workspace/burnseg-xai/ (uploaded via upload_to_runpod.sh).
# Usage: bash /workspace/burnseg-xai/scripts/runpod_setup.sh

set -e

echo "=== Installing non-CUDA dependencies ==="
# Install only the packages not in the PyTorch template.
# TMPDIR=/workspace avoids container disk space issues during download.
# torch/CUDA packages are already in the template: do NOT reinstall them.
TMPDIR=/workspace pip install \
    rasterio mlflow matplotlib scikit-learn scikit-image scipy tqdm PyYAML \
    -q --ignore-installed blinker --break-system-packages

echo "=== Installing burnseg-xai (editable, no deps) ==="
# --no-deps: torch and CUDA are already installed; do not resolve/upgrade them.
pip install -e /workspace/burnseg-xai --no-deps -q --break-system-packages

echo "=== Creating output directories ==="
mkdir -p /workspace/experimento_queimadas/checkpoints
mkdir -p /workspace/experimento_queimadas/mlruns
mkdir -p /workspace/experimento_queimadas/logs

echo "=== Copying split_master.json ==="
cp /workspace/split_master.json /workspace/experimento_queimadas/split_master.json

echo "=== Verifying setup ==="
python -c "
import glob, torch
tifs = glob.glob('/workspace/dataset_mestrado/**/*.tif', recursive=True)
print(f'Patches: {len(tifs)}')
assert len(tifs) > 5000, 'Dataset incomplete'
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
import burnseg_xai
print('burnseg_xai: OK')
"

echo "=== Setup complete. Run experiments with: ==="
echo "  bash /workspace/burnseg-xai/scripts/run_experiments_runpod.sh"
