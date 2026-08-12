#!/usr/bin/env bash
# Suite completa de experimentos: RunPod RTX 5090
# Já concluídos (não re-rodar): rrr_l0.1_cosine, rrr_l0.001_mse
# Prerequisites: bash /workspace/burnseg-xai/scripts/runpod_setup.sh
# Usage: bash /workspace/burnseg-xai/scripts/run_experiments_runpod.sh

# Sem set -e: cada experimento roda independentemente mesmo se outro falhar
set -uo pipefail

REPO="/workspace/burnseg-xai"
CONFIG="$REPO/configs/config_runpod.yaml"
LOGS="/workspace/experimento_queimadas/logs"

mkdir -p "$LOGS"
cd "$REPO"

run_exp() {
    local label="$1"; shift
    echo ""
    echo "========================================================"
    echo "$label: $(date)"
    echo "========================================================"
    if python -m "$@"; then
        echo "OK: $(date)"
    else
        echo "ERRO (exit $?): $label: continuando suite"
    fi
}

echo "========================================================"
echo "[SUITE] Iniciando em $(date)"
echo "Experimentos já concluídos: rrr_l0.1_cosine, rrr_l0.001_mse"
echo "========================================================"

# [1/5] Baseline: hipótese nula, sem RRR
# Roda primeiro: mais rápido (sem overhead de create_graph) e dá referência cedo.
run_exp "[1/5] BASELINE lambda=0.0 (sem RRR)" \
    burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" --lambda_rrr 0.0 \
    2>&1 | tee "$LOGS/baseline.txt"

# [2/5] Método principal: RRR-MSE lambda=0.1
# Era o [2/4] do design original e foi omitido do RunPod anterior por engano.
# É o experimento central da dissertação: permite comparar diretamente
# cosine vs. MSE para o mesmo lambda (0.1).
run_exp "[2/5] RRR-MSE lambda=0.1 (método principal)" \
    burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" --lambda_rrr 0.1 --rrr_distance_metric mse \
    2>&1 | tee "$LOGS/grid_l0.1_mse.txt"

# [3/5] Grid MSE lambda=0.01: re-run
# Pod anterior morreu no epoch 193/300 sem logar test metrics.
run_exp "[3/5] GRID lambda=0.01, mse (re-run: anterior interrompido)" \
    burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" --lambda_rrr 0.01 --rrr_distance_metric mse \
    2>&1 | tee "$LOGS/grid_l0.01_mse.txt"

# [4/5] Grid MSE lambda=1.0: extremo superior
# Testa se regularização forte demais degrada reconstrução.
# Junto com 0.001 (já feito) e 0.01, completa a curva de sensibilidade ao lambda.
run_exp "[4/5] GRID lambda=1.0, mse (extremo superior)" \
    burnseg_xai.pipeline.run_experiment \
    --config "$CONFIG" --lambda_rrr 1.0 --rrr_distance_metric mse \
    2>&1 | tee "$LOGS/grid_l1.0_mse.txt"

# [5/5] LORO: comentado: rodar após análise do grid
# run_exp "[5/5] LORO + CROSS-BIOME lambda=0.1, mse" \
#     burnseg_xai.pipeline.run_loro \
#     --config "$CONFIG" --lambda_rrr 0.1 --rrr_distance_metric mse \
#     2>&1 | tee "$LOGS/loro.txt"

echo ""
echo "========================================================"
echo "[SUITE] Todos os experimentos concluídos em $(date)"
echo "========================================================"

echo ""
echo "--- [BACKUP] Salvando checkpoints e MLflow no volume ---"
bash "$REPO/scripts/backup_pod_results.sh"
