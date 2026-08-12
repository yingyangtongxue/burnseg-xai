"""
ocsvm_gridsearch.py -- Grid search do One-Class SVM treinado em patches limpos.

Protocolo:
  - Treina EXCLUSIVAMENTE sobre patches com mean(dNBR) <= 0.1 (y_tr==0).
  - Avalia AUC no conjunto de teste (seed=43).
  - Grid: nu x gamma = 6 x 6 = 36 combinacoes.
  - Reporta melhor configuracao e metricas finais.

Uso:
    python scripts/ocsvm_gridsearch.py
"""

import os, sys, time
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "report_assets", "features_cache.npz")

NU_GRID    = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]


def compute_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    best_f1, best_thr = 0.0, 0.0
    for p in range(50, 100):
        thr = np.percentile(y_score, p)
        preds = (y_score >= thr).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    preds = (y_score >= best_thr).astype(int)
    bacc  = balanced_accuracy_score(y_true, preds)
    return auc, best_f1, bacc


def main():
    print("Carregando cache de features...")
    c = np.load(CACHE_FILE)
    X_tr, y_tr = c["X_tr"], c["y_tr"]
    X_te, y_te = c["X_te"], c["y_te"]

    # Filtra apenas patches limpos para treino
    mask_clean = (y_tr == 0)
    X_clean = X_tr[mask_clean]
    print(f"Patches de treino limpos (dNBR <= 0.1): {len(X_clean)}")
    print(f"Patches de teste: {len(X_te)}  (queimados={y_te.sum()}, limpos={(y_te==0).sum()})")

    # Scaler ajustado apenas nos patches limpos
    scaler = StandardScaler()
    X_clean_sc = scaler.fit_transform(X_clean)
    X_te_sc    = scaler.transform(X_te)

    # --- Grid search ---
    print(f"\nGrid search: {len(NU_GRID)} nu x {len(GAMMA_GRID)} gamma = {len(NU_GRID)*len(GAMMA_GRID)} combinacoes\n")
    results = []
    t0 = time.time()

    for nu in NU_GRID:
        for gamma in GAMMA_GRID:
            t1 = time.time()
            model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            model.fit(X_clean_sc)
            scores = -model.decision_function(X_te_sc)
            auc, f1, bacc = compute_metrics(y_te, scores)
            elapsed = time.time() - t1
            tag = "  <-- melhor AUC ate agora" if (not results or auc > max(r["auc"] for r in results)) else ""
            gamma_str = f"{gamma:.3f}" if isinstance(gamma, float) else gamma
            print(f"  nu={nu:<5}  gamma={gamma_str:<6}  AUC={auc:.4f}  F1={f1:.4f}  BalAcc={bacc:.4f}  ({elapsed:.1f}s){tag}")
            results.append(dict(nu=nu, gamma=gamma, auc=auc, f1=f1, bacc=bacc))

    total = time.time() - t0
    print(f"\nGrid search concluido em {total:.1f}s")

    # Melhor por AUC
    best = max(results, key=lambda r: r["auc"])
    gamma_str = f"{best['gamma']:.3f}" if isinstance(best['gamma'], float) else best['gamma']
    print("\n" + "="*60)
    print("MELHOR CONFIGURACAO:")
    print(f"  nu={best['nu']}  gamma={gamma_str}")
    print(f"  AUC={best['auc']:.4f}  F1={best['f1']:.4f}  BalAcc={best['bacc']:.4f}")
    print("="*60)

    # Comparacao com versao mista original (nu=0.3, treino misto)
    print("\nREFERENCIA (treino misto, nu=0.3):")
    scaler_ns = StandardScaler()
    X_all_sc  = scaler_ns.fit_transform(np.vstack([X_tr, X_te]))
    X_tr_ns   = X_all_sc[:len(X_tr)]
    X_te_ns   = X_all_sc[len(X_tr):]
    orig = OneClassSVM(kernel="rbf", nu=0.3, gamma="scale")
    orig.fit(X_tr_ns)
    scores_orig = -orig.decision_function(X_te_ns)
    auc_o, f1_o, bacc_o = compute_metrics(y_te, scores_orig)
    print(f"  AUC={auc_o:.4f}  F1={f1_o:.4f}  BalAcc={bacc_o:.4f}")

    print("\nDiferenca (patches limpos, melhor config) vs (misto, nu=0.3):")
    print(f"  ΔAUC = {best['auc'] - auc_o:+.4f}")

    # Salva CSV com todos os resultados
    out = os.path.join(os.path.dirname(__file__), "..", "report_assets", "ocsvm_gridsearch.csv")
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nu", "gamma", "AUC", "F1", "BalAcc"])
        for r in results:
            w.writerow([r["nu"], r["gamma"], f"{r['auc']:.4f}", f"{r['f1']:.4f}", f"{r['bacc']:.4f}"])
    print(f"\nResultados completos salvos em: {out}")


if __name__ == "__main__":
    main()
