import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Number of candidate thresholds when searching for the optimal F1 cut-off
_N_THRESHOLD_CANDIDATES = 200


def _prepare_batch(batch: torch.Tensor, device: str):
    """
    Shared channel-extraction + normalization logic (mirrors Trainer._prepare_batch).
    Does NOT set requires_grad -- callers set it when needed.
    """
    raw = batch.to(device)  # (B, T, H, W, 22)

    dnbr = raw[..., 20]  # (B, T, H, W)

    x_raw = torch.cat([raw[..., :20], raw[..., 21:22]], dim=-1)  # (B, T, H, W, 21)
    x = x_raw.permute(0, 4, 1, 2, 3).contiguous()  # (B, 21, T, H, W)

    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
    x = (x - mean) / std

    # dNBR prior: threshold at 0.1 to suppress noise, normalise [0, 1]
    prior = dnbr.max(dim=1).values
    prior = torch.relu(prior)
    prior = prior * (prior > 0.1).float()
    prior = prior / prior.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    return x, prior


def saliency_prior_cosine(
    saliency: torch.Tensor, prior: torch.Tensor
) -> float:
    """Mean cosine similarity between saliency map and dNBR prior."""
    s_flat = saliency.view(saliency.size(0), -1).float()
    p_flat = prior.view(prior.size(0), -1).float()
    return F.cosine_similarity(s_flat, p_flat, dim=1).mean().item()


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    desc: str = "Evaluate",
) -> dict:
    """
    Compute reconstruction error statistics and saliency-prior cosine similarity
    over the full dataloader (typically test set).
    """
    model.eval()

    recon_errors: list = []
    cos_sims: list = []

    bar = tqdm(dataloader, desc=desc, leave=False)
    for batch in bar:
        x, prior = _prepare_batch(batch, device)
        x = x.requires_grad_(True)

        x_hat, _ = model(x)
        err_all = F.mse_loss(x_hat, x, reduction="none")
        per_sample = err_all.mean(dim=(1, 2, 3, 4))  # (B,)

        recon_errors.extend(per_sample.detach().cpu().tolist())

        # Saliency from batch-mean loss
        loss_scalar = per_sample.mean()
        grads = torch.autograd.grad(
            outputs=loss_scalar,
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]

        saliency = grads.abs().mean(dim=(1, 2))
        saliency = saliency / saliency.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)

        if prior.amax() >= 1e-8:
            cos_sims.append(saliency_prior_cosine(saliency, prior))

        bar.set_postfix(
            n=len(recon_errors),
            cos=f"{cos_sims[-1]:.4f}" if cos_sims else "n/a",
        )

    errors = np.array(recon_errors)
    return {
        "recon_error_mean":    float(errors.mean()),
        "recon_error_std":     float(errors.std()),
        "recon_error_p50":     float(np.percentile(errors, 50)),
        "recon_error_p90":     float(np.percentile(errors, 90)),
        "recon_error_p95":     float(np.percentile(errors, 95)),
        "recon_error_p99":     float(np.percentile(errors, 99)),
        "saliency_cosine":     float(np.mean(cos_sims)) if cos_sims else 0.0,
    }


def recon_separation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
) -> float:
    """
    Mean reconstruction error for burned-proxy patches minus mean for clean patches.

    A positive value means the model reconstructs clean patches better than burned
    ones -- the intended anomaly detection signal.

    ``dnbr_threshold`` is the same proxy used in proxy_auc; dNBR is never used as a label.
    """
    model.eval()
    burned_errors: list = []
    clean_errors:  list = []

    for batch in dataloader:
        raw = batch.to(device)
        mean_dnbr = raw[..., 20].mean(dim=(1, 2, 3)).cpu()

        x, _ = _prepare_batch(batch, device)
        with torch.no_grad():
            x_hat, _ = model(x)
            per_sample = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2, 3, 4)).cpu()

        for j in range(len(mean_dnbr)):
            err = per_sample[j].item()
            if mean_dnbr[j].item() > dnbr_threshold:
                burned_errors.append(err)
            else:
                clean_errors.append(err)

    if not burned_errors or not clean_errors:
        return 0.0  # val set has no burned-proxy patches (e.g. LORO yanomami val)
    return float(np.mean(burned_errors) - np.mean(clean_errors))


def proxy_auc(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
    desc: str = "Proxy AUC",
) -> float:
    """
    Proxy AUC: reconstruction error as anomaly score, dNBR threshold as proxy label.
    Evaluation-only: dNBR threshold is never introduced into training.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    scores: list = []
    labels: list = []

    bar = tqdm(dataloader, desc=desc, leave=False)
    for batch in bar:
        raw = batch.to(device)
        dnbr = raw[..., 20]
        mean_dnbr = dnbr.mean(dim=(1, 2, 3)).cpu().numpy()
        proxy_label = (mean_dnbr > dnbr_threshold).astype(int)

        x, _ = _prepare_batch(batch, device)

        with torch.no_grad():
            x_hat, _ = model(x)
            err = F.mse_loss(x_hat, x, reduction="none")
            per_sample = err.mean(dim=(1, 2, 3, 4)).cpu().numpy()

        scores.extend(per_sample.tolist())
        labels.extend(proxy_label.tolist())
        bar.set_postfix(n=len(scores))

    labels_arr = np.array(labels)
    if labels_arr.sum() == 0 or (1 - labels_arr).sum() == 0:
        return 0.5  # only one class present (e.g. LORO yanomami val set has no burned patches)

    return float(roc_auc_score(labels_arr, np.array(scores)))


# ---------------------------------------------------------------------------
# Patch-level segmentation metrics (F1, precision, recall, accuracy)
# ---------------------------------------------------------------------------

def _collect_scores_and_labels(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
    desc: str = "Scores",
):
    """Collect per-patch reconstruction error scores and dNBR proxy labels."""
    model.eval()
    scores: list = []
    labels: list = []

    bar = tqdm(dataloader, desc=desc, leave=False)
    for batch in bar:
        raw = batch.to(device)
        mean_dnbr = raw[..., 20].mean(dim=(1, 2, 3)).cpu().numpy()
        proxy_label = (mean_dnbr > dnbr_threshold).astype(int)

        x, _ = _prepare_batch(batch, device)
        with torch.no_grad():
            x_hat, _ = model(x)
            per_sample = (
                F.mse_loss(x_hat, x, reduction="none")
                .mean(dim=(1, 2, 3, 4))
                .cpu()
                .numpy()
            )

        scores.extend(per_sample.tolist())
        labels.extend(proxy_label.tolist())
        bar.set_postfix(n=len(scores))

    return np.array(scores), np.array(labels)


def find_optimal_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
) -> float:
    """
    Searches for the reconstruction-error threshold that maximises F1 on the
    validation set.

    This threshold is then applied to the test set so there is no leakage
    between optimisation and final evaluation.
    """
    from sklearn.metrics import f1_score

    scores, labels = _collect_scores_and_labels(
        model, val_loader, device, dnbr_threshold, desc="Threshold search (val)"
    )
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float(np.median(scores))

    candidates = np.percentile(scores, np.linspace(0, 100, _N_THRESHOLD_CANDIDATES))
    best_f1, best_thr = -1.0, float(np.median(scores))
    for thr in candidates:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def segmentation_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    threshold: float,
    dnbr_threshold: float = 0.1,
    desc: str = "Seg metrics",
) -> dict:
    """
    Patch-level burned area detection metrics using a fixed ``threshold``.

    Prediction rule: patch is classified as burned if its reconstruction error
    exceeds ``threshold``.  Ground truth: dNBR mean > ``dnbr_threshold``.

    Returns
    -------
    dict with: f1, precision, recall, accuracy, balanced_accuracy,
               n_burned, n_clean, threshold.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    scores, labels = _collect_scores_and_labels(
        model, dataloader, device, dnbr_threshold, desc=desc
    )

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        nan = float("nan")
        return {
            "f1": nan, "precision": nan, "recall": nan,
            "accuracy": nan, "balanced_accuracy": nan,
            "n_burned": int(labels.sum()), "n_clean": int((1 - labels).sum()),
            "threshold": threshold,
        }

    preds = (scores >= threshold).astype(int)
    return {
        "f1":                float(f1_score(labels, preds, zero_division=0)),
        "precision":         float(precision_score(labels, preds, zero_division=0)),
        "recall":            float(recall_score(labels, preds, zero_division=0)),
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "n_burned":          int(labels.sum()),
        "n_clean":           int((1 - labels).sum()),
        "threshold":         float(threshold),
    }


# ---------------------------------------------------------------------------
# Pixel-level segmentation metrics with dNBR > threshold as reference mask
# ---------------------------------------------------------------------------
# Reference: Key and Benson (2006, FIREMON); Miller and Thode (2007).
# USGS standard: dNBR > 0.10 = burned (any severity).
# The dNBR mask is evaluation-only, it is never used during training.
# ---------------------------------------------------------------------------

def find_pixel_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
    max_batches: int = 80,
) -> float:
    """
    Find the per-pixel reconstruction-error threshold that maximises F1 on
    the validation set, using dNBR > ``dnbr_threshold`` as the pixel mask.

    Search is limited to ``max_batches`` to keep runtime tractable.
    Uses sklearn's precision_recall_curve for an efficient sweep.
    """
    from sklearn.metrics import precision_recall_curve

    model.eval()
    all_scores: list = []
    all_labels: list = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        raw = batch.to(device)
        dnbr_peak = raw[..., 20].max(dim=1).values      # (B, H, W) raw dNBR
        pixel_gt  = (dnbr_peak > dnbr_threshold).cpu().numpy().astype(np.uint8)

        x, _ = _prepare_batch(batch, device)
        with torch.no_grad():
            x_hat, _ = model(x)
            # Per-pixel recon error: mean over channels (dim=1) and time (dim=2)
            err = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))  # (B, H, W)

        all_scores.append(err.cpu().numpy().ravel())
        all_labels.append(pixel_gt.ravel())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float(np.median(scores))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision + recall
    denom[denom == 0] = 1e-8
    f1_arr = 2 * precision * recall / denom
    # precision_recall_curve returns one extra element at the end
    if len(thresholds) == 0:
        return float(np.median(scores))
    best_idx = int(np.argmax(f1_arr[:-1]))
    return float(thresholds[best_idx])


def pixel_segmentation_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    threshold: float,
    dnbr_threshold: float = 0.1,
    desc: str = "Pixel seg",
) -> dict:
    """
    Pixel-level burned area segmentation metrics.

    Prediction: per-pixel reconstruction error >= ``threshold``  →  burned.
    Reference : raw dNBR > ``dnbr_threshold`` (USGS standard: 0.10)  →  burned.

    Returns
    -------
    dict with: pixel_f1, pixel_iou (Jaccard), pixel_precision, pixel_recall,
               pixel_accuracy, pixel_n_burned, pixel_n_clean, threshold.
    """
    from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

    model.eval()
    all_preds:  list = []
    all_labels: list = []

    bar = tqdm(dataloader, desc=desc, leave=False)
    for batch in bar:
        raw = batch.to(device)
        dnbr_peak  = raw[..., 20].max(dim=1).values      # (B, H, W)
        pixel_gt   = (dnbr_peak > dnbr_threshold).cpu().numpy().astype(np.uint8)

        x, _ = _prepare_batch(batch, device)
        with torch.no_grad():
            x_hat, _ = model(x)
            err       = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))

        pixel_pred = (err.cpu().numpy() >= threshold).astype(np.uint8)
        all_preds.append(pixel_pred.ravel())
        all_labels.append(pixel_gt.ravel())
        bar.set_postfix(n=len(all_preds) * pixel_pred.shape[0])

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        nan = float("nan")
        return {
            "pixel_f1": nan, "pixel_iou": nan, "pixel_precision": nan,
            "pixel_recall": nan, "pixel_accuracy": nan,
            "pixel_n_burned": int(labels.sum()), "pixel_n_clean": int((1 - labels).sum()),
            "threshold": float(threshold),
        }

    return {
        "pixel_f1":        float(f1_score(labels, preds, zero_division=0)),
        "pixel_iou":       float(jaccard_score(labels, preds, zero_division=0)),
        "pixel_precision": float(precision_score(labels, preds, zero_division=0)),
        "pixel_recall":    float(recall_score(labels, preds, zero_division=0)),
        "pixel_accuracy":  float((preds == labels).mean()),
        "pixel_n_burned":  int(labels.sum()),
        "pixel_n_clean":   int((1 - labels).sum()),
        "threshold":       float(threshold),
    }


# ---------------------------------------------------------------------------
# Otsu-threshold segmentation metrics (no val-set calibration required)
# ---------------------------------------------------------------------------

def otsu_segmentation_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    dnbr_threshold: float = 0.1,
    desc: str = "Otsu seg",
) -> dict:
    """
    Patch-level burned area detection using Otsu's threshold on reconstruction
    error. The threshold is derived from the score distribution itself, no
    validation set needed.

    Returns
    -------
    dict with: otsu_f1, otsu_iou (Jaccard), otsu_kappa (Cohen's kappa),
               otsu_precision, otsu_recall, otsu_threshold,
               n_burned, n_clean.
    """
    from skimage.filters import threshold_otsu
    from sklearn.metrics import (
        cohen_kappa_score,
        f1_score,
        jaccard_score,
        precision_score,
        recall_score,
    )

    scores, labels = _collect_scores_and_labels(
        model, dataloader, device, dnbr_threshold, desc=desc
    )

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        nan = float("nan")
        return {
            "otsu_f1": nan, "otsu_iou": nan, "otsu_kappa": nan,
            "otsu_precision": nan, "otsu_recall": nan,
            "otsu_threshold": nan,
            "n_burned": int(labels.sum()), "n_clean": int((1 - labels).sum()),
        }

    try:
        otsu_thr = float(threshold_otsu(scores))
    except Exception:
        otsu_thr = float(np.median(scores))

    preds = (scores >= otsu_thr).astype(int)
    return {
        "otsu_f1":        float(f1_score(labels, preds, zero_division=0)),
        "otsu_iou":       float(jaccard_score(labels, preds, zero_division=0)),
        "otsu_kappa":     float(cohen_kappa_score(labels, preds)),
        "otsu_precision": float(precision_score(labels, preds, zero_division=0)),
        "otsu_recall":    float(recall_score(labels, preds, zero_division=0)),
        "otsu_threshold": otsu_thr,
        "n_burned":       int(labels.sum()),
        "n_clean":        int((1 - labels).sum()),
    }
