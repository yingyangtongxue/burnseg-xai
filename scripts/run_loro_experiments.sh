#!/usr/bin/env bash
# run_loro_experiments.sh: LORO folds with lambda=0.1 MSE (best detection config)
#
# Covers all 4 LORO folds: geographic generalization across the three
# Amazonia regions plus the cross-biome fold (Amazonia -> Cerrado):
#   1. cross-biome (cerrado)  = leave-chapada
#   2. leave-yanomami        : smallest train set
#   3. leave-kayapo          : medium train set
#   4. leave-karipuna        : largest train set
#
# Usage (on pod):
#   bash /workspace/burnseg-xai/scripts/run_loro_experiments.sh
#   # or in tmux: tmux new -s loro && bash /workspace/burnseg-xai/scripts/run_loro_experiments.sh

set -uo pipefail

PYTHON="/workspace/venv/bin/python3"
REPO="/workspace/burnseg-xai"
CONFIG="$REPO/configs/config_runpod.yaml"
LOGS="/workspace/experimento_queimadas/logs"
LAMBDA="0.1"
METRIC="mse"

mkdir -p "$LOGS"
cd "$REPO"

run_loro() {
    local label="$1"; shift
    echo ""
    echo "========================================================"
    echo "LORO: $label: $(date)"
    echo "========================================================"
    if "$PYTHON" -m burnseg_xai.pipeline.run_loro "$@"; then
        echo "OK: $label: $(date)"
    else
        echo "ERRO (exit $?): $label: continuando"
    fi
}

echo "========================================================"
echo "[LORO SUITE] Iniciando em $(date)"
echo "Configuração: lambda_rrr=$LAMBDA metric=$METRIC"
echo "========================================================"

# [1/4] Cross-biome: Amazônia → Cerrado
run_loro "cross-biome (cerrado)" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --cross_biome_only \
    2>&1 | tee "$LOGS/loro_cross_biome.txt"

# [2/4] Leave-Yanomami: menor train set, treino mais rápido
run_loro "leave-yanomami" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --regions yanomami \
    --no_cross_biome \
    2>&1 | tee "$LOGS/loro_yanomami.txt"

# [3/4] Leave-Kayapó
run_loro "leave-kayapo" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --regions kayapo \
    --no_cross_biome \
    2>&1 | tee "$LOGS/loro_kayapo.txt"

# [4/4] Leave-Karipuna
run_loro "leave-karipuna" \
    --config "$CONFIG" \
    --lambda_rrr "$LAMBDA" \
    --rrr_distance_metric "$METRIC" \
    --regions karipuna \
    --no_cross_biome \
    2>&1 | tee "$LOGS/loro_karipuna.txt"

echo ""
echo "========================================================"
echo "[LORO SUITE] Todos os folds concluídos em $(date)"
echo "========================================================"

echo ""
echo "--- [BACKUP] Salvando resultados no volume ---"
bash "$REPO/scripts/backup_pod_results.sh"
