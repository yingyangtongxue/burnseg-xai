"""
supervised_baselines.py -- Baselines comparativos para o autoencoder com RRR.

Computa baselines usando o mesmo split_master.json e o mesmo pseudo-label
(mean(dNBR) > 0.1) usados em todos os experimentos do grid principal.

Familias avaliadas:
  1. Trivial (circular): dNBR direto como score -- teto AUC ~ 1.0 por construcao.
  2. Nao-supervisionados classicos: K-means (k=2), Isolation Forest, One-Class SVM,
     PCA anomaly score.
  3. Supervisionados: Random Forest, SVM, XGBoost -- mostra o custo da ausencia
     de labels (labels derivados do dNBR => avaliacao igualmente circular).

Features: mean + std de cada um dos 21 canais de entrada do modelo (42 features/patch).
Os 21 canais sao os mesmos que o autoencoder usa (canais 0-19 + canal 21 = dNDVI).
O canal 20 (dNBR) e EXCLUIDO das features -- usado apenas como label/referencia.

Cache: as features extraidas sao salvas em report_assets/features_cache.npz para
evitar re-ler o HDD em execucoes subsequentes.

Uso:
    python scripts/supervised_baselines.py [--dataset ./data]
                                           [--split path/to/split_master.json]
                                           [--no-cache]
"""

import argparse
import csv
import json
import sys
import os

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from burnseg_xai.dataset import BurnedAreaDataset

DNBR_THR   = 0.1
DNBR_CH    = 20           # canal 20 = dNBR (prior, nao entra no modelo)
MODEL_CHS  = list(range(20)) + [21]   # 21 canais que o autoencoder usa

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "report_assets",
                          "features_cache.npz")


def load_patches(dataset, indices, desc=""):
    """
    Retorna (features, labels, dnbr_means) para os indices fornecidos.

    features:   (N, 42) -- mean + std de cada um dos 21 canais de entrada
    labels:     (N,)    -- 1 se mean(dNBR) > DNBR_THR, 0 caso contrario
    dnbr_means: (N,)    -- mean(dNBR) continuo (para AUC do baseline direto)
    """
    features, labels, dnbr_means = [], [], []

    for idx in tqdm(indices, desc=desc):
        raw = dataset[idx].numpy()          # (T=1, H, W, C=22), float32
        model_data = raw[0, :, :, MODEL_CHS]   # (H, W, 21)
        feat = np.concatenate([
            model_data.mean(axis=(0, 1)),   # (21,)
            model_data.std(axis=(0, 1)),    # (21,)
        ])   # (42,)
        features.append(feat)

        dnbr = raw[0, :, :, DNBR_CH]       # (H, W)
        mean_dnbr = float(dnbr.mean())
        labels.append(1 if mean_dnbr > DNBR_THR else 0)
        dnbr_means.append(mean_dnbr)

    return (
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=int),
        np.array(dnbr_means, dtype=np.float32),
    )


def compute_metrics(y_true, y_score, threshold=None):
    """AUC, F1 e BalAcc. threshold=None usa varredura de percentis."""
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

    auc = roc_auc_score(y_true, y_score)

    if threshold is None:
        best_f1, best_thr = 0.0, 0.5
        for p in range(50, 100):
            thr = np.percentile(y_score, p)
            preds = (y_score >= thr).astype(int)
            f = f1_score(y_true, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr = thr
        threshold = best_thr

    preds = (y_score >= threshold).astype(int)
    f1   = f1_score(y_true, preds, zero_division=0)
    bacc = balanced_accuracy_score(y_true, preds)
    return auc, f1, bacc, threshold


def print_row(name, auc, f1, bacc, note=""):
    note_str = f"  [{note}]" if note else ""
    print(f"  {name:<40s}  AUC={auc:.3f}  F1={f1:.3f}  BalAcc={bacc:.3f}{note_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="./data")
    parser.add_argument("--split",        default=None)
    parser.add_argument("--output_root",  default="./outputs")
    parser.add_argument("--no-cache",     action="store_true",
                        help="Ignora cache e re-extrai features do disco.")
    args = parser.parse_args()

    # --- localiza split_master.json ---
    split_path = args.split
    if split_path is None:
        for root, _, files in os.walk(args.output_root):
            if "split_master.json" in files:
                split_path = os.path.join(root, "split_master.json")
                break
        if split_path is None:
            sys.exit("ERRO: split_master.json nao encontrado. Use --split para especificar.")

    print(f"Dataset  : {args.dataset}")
    print(f"Split    : {split_path}")

    with open(split_path) as f:
        split = json.load(f)

    def get_split(name):
        return split.get(name, split.get(f"{name}_idx", []))

    train_idx = get_split("train")
    test_idx  = get_split("test")

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    cache_exists = os.path.exists(CACHE_FILE) and not args.no_cache

    if cache_exists:
        print("Carregando features do cache...")
        c = np.load(CACHE_FILE)
        X_tr, y_tr, dnbr_tr = c["X_tr"], c["y_tr"], c["dnbr_tr"]
        X_te, y_te, dnbr_te = c["X_te"], c["y_te"], c["dnbr_te"]
        print(f"Cache: treino={len(X_tr)}, teste={len(X_te)} patches")
    else:
        dataset = BurnedAreaDataset(root_dir=args.dataset, temporal_length=1)
        print(f"Patches  : {len(dataset)}")
        print(f"Treino   : {len(train_idx)} patches  |  Teste: {len(test_idx)} patches\n")

        print("Extraindo features de treino...")
        X_tr, y_tr, dnbr_tr = load_patches(dataset, train_idx, desc="  train")

        print("Extraindo features de teste...")
        X_te, y_te, dnbr_te = load_patches(dataset, test_idx, desc="  test")

        np.savez(CACHE_FILE,
                 X_tr=X_tr, y_tr=y_tr, dnbr_tr=dnbr_tr,
                 X_te=X_te, y_te=y_te, dnbr_te=dnbr_te)
        print(f"Cache salvo em: {CACHE_FILE}")

    print(f"\nDistribuicao (teste) -- queimado: {y_te.sum()}  limpo: {(y_te == 0).sum()}")

    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("RESULTADOS (conjunto de teste, seed=43)")
    print("="*70)

    # --- Teto trivial ---
    auc_dnbr, f1_dnbr, bacc_dnbr, _ = compute_metrics(y_te, dnbr_te)
    print_row("dNBR direto (teto circular)", auc_dnbr, f1_dnbr, bacc_dnbr,
              note="score=mean(dNBR); AUC->1 trivial")

    # -----------------------------------------------------------------------
    # Modelos supervisionados (labels = dNBR > 0.1, avaliacao circular)
    # -----------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    print()
    # Random Forest
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("Treinando Random Forest (n=100, seed=43)...")
        rf = RandomForestClassifier(n_estimators=100, random_state=43, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        scores_rf = rf.predict_proba(X_te_sc)[:, 1]
        auc_rf, f1_rf, bacc_rf, _ = compute_metrics(y_te, scores_rf)
        print_row("Random Forest (supervisionado)", auc_rf, f1_rf, bacc_rf,
                  note="labels=dNBR>0.1")
    except ImportError:
        auc_rf = f1_rf = bacc_rf = float("nan")

    # SVM RBF
    try:
        from sklearn.svm import SVC
        print("Treinando SVM RBF (C=1.0, seed=43)...")
        svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=43)
        svm.fit(X_tr_sc, y_tr)
        scores_svm = svm.predict_proba(X_te_sc)[:, 1]
        auc_svm, f1_svm, bacc_svm, _ = compute_metrics(y_te, scores_svm)
        print_row("SVM RBF (supervisionado)", auc_svm, f1_svm, bacc_svm,
                  note="labels=dNBR>0.1")
    except ImportError:
        auc_svm = f1_svm = bacc_svm = float("nan")

    # XGBoost
    auc_xgb = f1_xgb = bacc_xgb = float("nan")
    try:
        from xgboost import XGBClassifier
        print("Treinando XGBoost (n=100, seed=43)...")
        xgb = XGBClassifier(n_estimators=100, random_state=43,
                            eval_metric="logloss", verbosity=0, n_jobs=-1)
        xgb.fit(X_tr_sc, y_tr)
        scores_xgb = xgb.predict_proba(X_te_sc)[:, 1]
        auc_xgb, f1_xgb, bacc_xgb, _ = compute_metrics(y_te, scores_xgb)
        print_row("XGBoost (supervisionado)", auc_xgb, f1_xgb, bacc_xgb,
                  note="labels=dNBR>0.1")
    except ImportError:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            print("xgboost nao encontrado. Usando GradientBoosting (sklearn)...")
            gbt = GradientBoostingClassifier(n_estimators=100, random_state=43)
            gbt.fit(X_tr_sc, y_tr)
            scores_xgb = gbt.predict_proba(X_te_sc)[:, 1]
            auc_xgb, f1_xgb, bacc_xgb, _ = compute_metrics(y_te, scores_xgb)
            print_row("GradientBoosting/sklearn (supervisionado)", auc_xgb, f1_xgb, bacc_xgb,
                      note="labels=dNBR>0.1")
        except Exception as e:
            print(f"  AVISO: XGBoost/GBT falhou: {e}")

    # -----------------------------------------------------------------------
    # Modelos nao-supervisionados classicos
    # -----------------------------------------------------------------------
    # Scaler separado: fit em treino+teste (sem usar labels)
    scaler_ns = StandardScaler()
    X_all_sc  = scaler_ns.fit_transform(np.vstack([X_tr, X_te]))
    X_tr_ns   = X_all_sc[:len(X_tr)]
    X_te_ns   = X_all_sc[len(X_tr):]

    print()
    # K-means
    try:
        from sklearn.cluster import KMeans
        print("Treinando K-means (k=2, seed=43)...")
        km = KMeans(n_clusters=2, random_state=43, n_init=10)
        km.fit(X_tr_ns)
        labels_tr_km  = km.predict(X_tr_ns)
        cluster_counts = np.bincount(labels_tr_km, minlength=2)
        burned_cluster = int(np.argmin(cluster_counts))
        dists_te = np.linalg.norm(X_te_ns - km.cluster_centers_[burned_cluster], axis=1)
        auc_km, f1_km, bacc_km, _ = compute_metrics(y_te, dists_te)
        print_row("K-means (k=2, dist. centroide)", auc_km, f1_km, bacc_km,
                  note="nao-supervisionado")
    except ImportError:
        auc_km = f1_km = bacc_km = float("nan")

    # Isolation Forest
    try:
        from sklearn.ensemble import IsolationForest
        print("Treinando Isolation Forest (n=100, seed=43)...")
        isofor = IsolationForest(n_estimators=100, random_state=43,
                                 contamination=0.3, n_jobs=-1)
        isofor.fit(X_tr_ns)
        scores_iso = -isofor.score_samples(X_te_ns)
        auc_iso, f1_iso, bacc_iso, _ = compute_metrics(y_te, scores_iso)
        print_row("Isolation Forest (nao-sup.)", auc_iso, f1_iso, bacc_iso,
                  note="nao-supervisionado")
    except ImportError:
        auc_iso = f1_iso = bacc_iso = float("nan")

    # One-Class SVM
    auc_ocsvm = f1_ocsvm = bacc_ocsvm = float("nan")
    try:
        from sklearn.svm import OneClassSVM
        print("Treinando One-Class SVM (RBF, nu=0.3)...")
        ocsvm = OneClassSVM(kernel="rbf", nu=0.3, gamma="scale")
        ocsvm.fit(X_tr_ns)
        # decision_function: negativo = anomalia; invertemos o sinal
        scores_ocsvm = -ocsvm.decision_function(X_te_ns)
        auc_ocsvm, f1_ocsvm, bacc_ocsvm, _ = compute_metrics(y_te, scores_ocsvm)
        print_row("One-Class SVM (nao-sup.)", auc_ocsvm, f1_ocsvm, bacc_ocsvm,
                  note="nao-supervisionado")
    except ImportError:
        print("  AVISO: sklearn.svm.OneClassSVM nao disponivel")

    # PCA reconstruction error
    try:
        from sklearn.decomposition import PCA
        print("Computando PCA anomaly score (n=10)...")
        pca = PCA(n_components=10, random_state=43)
        pca.fit(X_tr_ns)
        X_te_recon   = pca.inverse_transform(pca.transform(X_te_ns))
        scores_pca   = np.linalg.norm(X_te_ns - X_te_recon, axis=1)
        auc_pca, f1_pca, bacc_pca, _ = compute_metrics(y_te, scores_pca)
        print_row("PCA erro de reconstrucao (n=10)", auc_pca, f1_pca, bacc_pca,
                  note="nao-supervisionado")
    except ImportError:
        auc_pca = f1_pca = bacc_pca = float("nan")

    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("INTERPRETACAO:")
    print(f"  dNBR direto AUC={auc_dnbr:.3f} -> teto circular (score IS o label)")
    if not np.isnan(auc_rf):
        print(f"  RF   AUC={auc_rf:.3f}  SVM  AUC={auc_svm:.3f}  XGB AUC={auc_xgb:.3f}")
        print(f"  -> supervisionados usam labels=dNBR, avaliacao igualmente circular")
    print(f"  Nao-supervisionados classicos: IsoFor={auc_iso:.3f}  OC-SVM={auc_ocsvm:.3f}  "
          f"PCA={auc_pca:.3f}  Kmeans={auc_km:.3f}")
    print("  Compare estes resultados com as metricas do autoencoder+RRR em MLflow.")
    print("="*70)

    # -----------------------------------------------------------------------
    # Salva CSV
    # -----------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "..", "report_assets",
                            "baselines_comparativos.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuracao", "paradigma", "AUC", "F1_patch", "BalAcc", "nota"])
        writer.writerow(["dNBR direto", "trivial/circular",
                         f"{auc_dnbr:.3f}", f"{f1_dnbr:.3f}", f"{bacc_dnbr:.3f}",
                         "teto trivial - score=label"])
        writer.writerow(["K-means (k=2)", "nao-supervisionado-classico",
                         f"{auc_km:.3f}", f"{f1_km:.3f}", f"{bacc_km:.3f}", ""])
        writer.writerow(["Isolation Forest", "nao-supervisionado-classico",
                         f"{auc_iso:.3f}", f"{f1_iso:.3f}", f"{bacc_iso:.3f}", ""])
        writer.writerow(["One-Class SVM", "nao-supervisionado-classico",
                         f"{auc_ocsvm:.3f}", f"{f1_ocsvm:.3f}", f"{bacc_ocsvm:.3f}", ""])
        writer.writerow(["PCA recon error (n=10)", "nao-supervisionado-classico",
                         f"{auc_pca:.3f}", f"{f1_pca:.3f}", f"{bacc_pca:.3f}", ""])
        writer.writerow(["Random Forest", "supervisionado-circular",
                         f"{auc_rf:.3f}", f"{f1_rf:.3f}", f"{bacc_rf:.3f}",
                         "labels=dNBR>0.1 (circular)"])
        writer.writerow(["SVM RBF", "supervisionado-circular",
                         f"{auc_svm:.3f}", f"{f1_svm:.3f}", f"{bacc_svm:.3f}",
                         "labels=dNBR>0.1 (circular)"])
        writer.writerow(["XGBoost", "supervisionado-circular",
                         f"{auc_xgb:.3f}", f"{f1_xgb:.3f}", f"{bacc_xgb:.3f}",
                         "labels=dNBR>0.1 (circular)"])
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
