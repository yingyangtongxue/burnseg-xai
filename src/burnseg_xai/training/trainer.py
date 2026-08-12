"""
Canonical trainer for the burned-area 3D CNN autoencoder.

Key features
------------
- Plain MSE reconstruction loss
- RRR (Right for the Right Reasons) saliency regularization via
  create_graph=True autograd gradients -- the XAI signal directly
  shapes the weight updates every step
- dNBR prior thresholded at 0.1 to suppress spectral noise
- AMP (automatic mixed precision) for memory efficiency
- Early stopping with configurable patience
- Checkpoint save/resume (latest + best)
- tqdm progress bars at epoch and step level
"""

import os
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm

from burnseg_xai.xai.losses import xai_loss


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        logger,
        lambda_rrr: float = 0.0,
        rrr_distance_metric: str = "mse",
        xai_terms: tuple = ("grad", "gradcam", "attn"),
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 20,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.lambda_rrr = lambda_rrr
        self.rrr_distance_metric = rrr_distance_metric
        self.xai_terms = xai_terms
        self.scaler = GradScaler(device=device)

        # Early stopping state
        self._es_patience = early_stopping_patience
        self._es_counter = 0
        self._best_val_loss = float("inf")

        # Checkpoint
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # MLflow run id (stored in checkpoint for resume)
        self._mlflow_run_id = mlflow_run_id

        # Stats tracked across epochs
        self.rrr_skip_count = 0
        self.epoch_metrics_history: list = []  # one dict per completed epoch

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(self, batch: torch.Tensor):
        """
        Split raw 22-channel batch into normalised model input and dNBR prior.

        Returns
        -------
        x     : (B, 21, T, H, W)   per-patch Z-score, requires_grad=True
        prior : (B, H, W)          dNBR normalised to [0, 1]
        """
        raw = batch.to(self.device)             # (B, T, H, W, 22)
        dnbr = raw[..., 20]                     # (B, T, H, W)  raw dNBR

        # Input channels: 0-19 + 21 (dNDVI)
        x_raw = torch.cat([raw[..., :20], raw[..., 21:22]], dim=-1)
        x = x_raw.permute(0, 4, 1, 2, 3).contiguous()   # (B, 21, T, H, W)

        # Per-sample Z-score (prevents data leakage between patches)
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        std  = x.std(dim=(1, 2, 3, 4),  keepdim=True) + 1e-8
        x    = (x - mean) / std

        # dNBR prior: temporal peak, threshold at 0.1 to suppress noise, normalise [0, 1]
        prior = dnbr.max(dim=1).values
        prior = torch.relu(prior)
        prior = prior * (prior > 0.1).float()   # zero out sub-threshold noise
        prior = prior / prior.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)

        x = x.requires_grad_(True)
        return x, prior

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _recon_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x)

    def _spatial_saliency(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregate (B, C, T, H, W) gradients -> normalised saliency (B, H, W).
        Mean over C (dim=1) and T (dim=2) gives a spatial sensitivity map.
        """
        sal = grads.abs().mean(dim=(1, 2))
        sal = torch.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)
        sal = sal / sal.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        return sal

    def _rrr_loss(
        self,
        loss_recon: torch.Tensor,
        x: torch.Tensor,
        prior: torch.Tensor,
    ):
        """
        Compute RRR (Right for the Right Reasons) loss.

        Uses create_graph=True so the saliency gradient path flows back
        to model weights: this is what makes RRR an actual regulariser.
        Uses retain_graph=True so loss_total.backward() can traverse the
        same computation graph afterwards.

        Returns (loss_rrr, saliency) where loss_rrr is a scalar tensor.
        """
        grads = torch.autograd.grad(
            outputs=loss_recon,
            inputs=x,
            create_graph=True,    # required, allows d(loss_rrr)/d(theta)
            retain_graph=True,    # required, needed by loss_total.backward()
        )[0]                      # (B, C, T, H, W)

        saliency = self._spatial_saliency(grads)

        if torch.isnan(saliency).any():
            self.rrr_skip_count += 1
            return torch.tensor(0.0, device=self.device), saliency.nan_to_num(0.0)

        if prior.amax() < 1e-8:
            # no burn signal, skip to avoid penalising noise
            self.rrr_skip_count += 1
            return torch.tensor(0.0, device=self.device), saliency

        if self.rrr_distance_metric == "cosine":
            s_flat = saliency.view(saliency.size(0), -1)
            p_flat = prior.view(prior.size(0), -1)
            loss_rrr = (1.0 - F.cosine_similarity(s_flat, p_flat, dim=1)).mean()
        else:  # "mse" default
            loss_rrr = F.mse_loss(saliency, prior.detach())

        return loss_rrr, saliency

    # ------------------------------------------------------------------
    # Train / val steps
    # ------------------------------------------------------------------

    def _train_step(self, batch: torch.Tensor) -> dict:
        x, prior = self._prepare_batch(batch)

        # No autocast here: create_graph=True requires float32 for gradient stability.
        # Float16 gradients overflow with 4-layer networks, producing NaN saliency.
        x_hat, z = self.model(x)
        loss_recon = self._recon_loss(x_hat, x)

        if self.lambda_rrr > 0:
            combined_xai, xai_components = xai_loss(
                loss_recon=loss_recon,
                x=x,
                z=z,
                prior=prior,
                attention_module=self.model.attention,
                distance_metric=self.rrr_distance_metric,
                terms=self.xai_terms,
            )
            loss_rrr = combined_xai
        else:
            loss_rrr = torch.tensor(0.0, device=self.device)
            xai_components = {"grad": 0.0, "gradcam": 0.0, "attn": 0.0}

        loss_total = loss_recon + self.lambda_rrr * loss_rrr

        self.optimizer.zero_grad()
        self.scaler.scale(loss_total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        def _item(v):
            """Return float from a tensor or pass through a plain float."""
            return v.item() if isinstance(v, torch.Tensor) else float(v)

        return {
            "total":        loss_total.item(),
            "recon":        loss_recon.item(),
            "rrr":          _item(loss_rrr),
            "xai_grad":     _item(xai_components["grad"]),
            "xai_gradcam":  _item(xai_components["gradcam"]),
            "xai_attn":     _item(xai_components["attn"]),
        }

    def _val_step(self, batch: torch.Tensor) -> dict:
        """
        Validation step, computes saliency for val_saliency_cosine.
        Gradients are enabled but optimizer is NOT stepped.
        No autocast: float16 gradient overflow produces NaN (same reason as _train_step).
        """
        x, prior = self._prepare_batch(batch)

        x_hat, _z = self.model(x)
        loss_recon = self._recon_loss(x_hat, x)

        # saliency (create_graph=False, we don't backprop through val saliency)
        grads = torch.autograd.grad(
            outputs=loss_recon,
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]
        saliency = self._spatial_saliency(grads)

        if prior.amax() < 1e-8:
            cos_sim = 0.0
            loss_rrr = torch.tensor(0.0, device=self.device)
        else:
            s_flat = saliency.view(saliency.size(0), -1).float()
            p_flat = prior.view(prior.size(0), -1).float()
            cos_sim = F.cosine_similarity(s_flat, p_flat, dim=1).mean().item()

            if self.lambda_rrr > 0:
                if self.rrr_distance_metric == "cosine":
                    loss_rrr = (1.0 - F.cosine_similarity(
                        saliency.view(saliency.size(0), -1),
                        prior.view(prior.size(0), -1),
                        dim=1,
                    )).mean()
                else:
                    loss_rrr = F.mse_loss(saliency, prior.detach())
            else:
                loss_rrr = torch.tensor(0.0, device=self.device)

        loss_total = loss_recon + self.lambda_rrr * loss_rrr

        # Per-sample errors for recon_separation (burned vs clean discrimination).
        # keys prefixed with "_" are lists, excluded from float accumulation.
        with torch.no_grad():
            per_sample = (
                F.mse_loss(x_hat, x, reduction="none")
                .mean(dim=(1, 2, 3, 4))
                .detach()
                .cpu()
                .tolist()
            )
        # Proxy burned: patches where any dNBR-prior pixel survived thresholding
        proxy_burned = (prior.amax(dim=(1, 2)) > 1e-3).cpu().tolist()

        return {
            "total":              loss_total.item(),
            "recon":              loss_recon.item(),
            "saliency_cosine":    cos_sim,
            "_per_sample_errors": per_sample,   # list, not accumulated as float
            "_proxy_burned":      proxy_burned, # list, not accumulated as float
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch":                epoch + 1,  # resume starts at next epoch
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "best_val_loss":        self._best_val_loss,
            "es_counter":           self._es_counter,
            "mlflow_run_id":        self._mlflow_run_id,
        }
        latest = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(state, latest)

        if is_best:
            best = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            torch.save(state, best)
            tqdm.write(
                f"  [best] val_loss={self._best_val_loss:.5f} -> {best}"
            )

    def load_checkpoint(self, path: str) -> int:
        """
        Loads all training state from a checkpoint file.
        Returns the epoch to start from.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self._best_val_loss = state.get("best_val_loss", float("inf"))
        self._es_counter    = state.get("es_counter", 0)
        if state.get("mlflow_run_id"):
            self._mlflow_run_id = state["mlflow_run_id"]
        start_epoch = state.get("epoch", 0)
        tqdm.write(f"Checkpoint loaded: resuming from epoch {start_epoch + 1}")
        return start_epoch

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> dict:
        """
        Runs the full training loop with early stopping and checkpointing.

        Auto-resumes from ``checkpoint_dir/checkpoint_latest.pt`` if it exists.

        Returns
        -------
        dict with final epoch metrics.
        """
        # --- Auto-resume ---
        start_epoch = 0
        if self.checkpoint_dir:
            latest = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            if os.path.exists(latest):
                start_epoch = self.load_checkpoint(latest)

        final_metrics: dict = {}

        epoch_bar = tqdm(
            range(start_epoch, epochs),
            initial=start_epoch,
            total=epochs,
            desc="Epochs",
            unit="ep",
            position=0,
        )

        for epoch in epoch_bar:
            # ---- Train ----
            self.model.train()
            self.rrr_skip_count = 0
            train_accum: dict = defaultdict(float)

            train_bar = tqdm(
                train_loader,
                desc="  train",
                leave=False,
                position=1,
            )
            for batch in train_bar:
                step = self._train_step(batch)
                for k, v in step.items():
                    train_accum[k] += v
                train_bar.set_postfix(
                    loss=f"{step['total']:.4f}",
                    recon=f"{step['recon']:.4f}",
                    rrr=f"{step['rrr']:.4f}",
                    grad=f"{step['xai_grad']:.4f}",
                    gcam=f"{step['xai_gradcam']:.4f}",
                    attn=f"{step['xai_attn']:.4f}",
                )

            n_tr = max(len(train_loader), 1)
            avg_tr = {k: v / n_tr for k, v in train_accum.items()}

            # ---- Validate ----
            self.model.eval()
            val_accum: dict = defaultdict(float)
            val_errors:  list = []
            val_proxy:   list = []

            val_bar = tqdm(
                val_loader,
                desc="  val  ",
                leave=False,
                position=1,
            )
            for batch in val_bar:
                step = self._val_step(batch)
                for k, v in step.items():
                    if not k.startswith("_"):
                        val_accum[k] += v
                val_errors.extend(step["_per_sample_errors"])
                val_proxy.extend(step["_proxy_burned"])
                val_bar.set_postfix(
                    loss=f"{step['total']:.4f}",
                    cos=f"{step['saliency_cosine']:.4f}",
                )

            n_val = max(len(val_loader), 1)
            avg_val = {k: v / n_val for k, v in val_accum.items()}

            # recon_separation: mean(burned_error) - mean(clean_error).
            # Positive = model assigns higher error to burned patches → anomaly
            # detection signal is present. Near-zero or negative = model is not
            # discriminating → segmentation will produce mostly black images.
            import numpy as _np
            _errors = _np.array(val_errors)
            _proxy  = _np.array(val_proxy, dtype=bool)
            if _proxy.any() and (~_proxy).any():
                recon_sep = float(_errors[_proxy].mean() - _errors[~_proxy].mean())
                try:
                    from sklearn.metrics import roc_auc_score as _auc
                    proxy_auc_val = float(_auc(_proxy.astype(int), _errors))
                except Exception:
                    proxy_auc_val = float("nan")
            else:
                recon_sep     = float("nan")
                proxy_auc_val = float("nan")

            # ---- Log metrics ----
            self.logger.log_metrics(
                {
                    "train_loss":              avg_tr["total"],
                    "train_recon_loss":        avg_tr["recon"],
                    "train_rrr_loss":          avg_tr["rrr"],
                    "train_xai_grad_loss":     avg_tr["xai_grad"],
                    "train_xai_gradcam_loss":  avg_tr["xai_gradcam"],
                    "train_xai_attn_loss":     avg_tr["xai_attn"],
                    "val_loss":                avg_val["total"],
                    "val_recon_loss":          avg_val["recon"],
                    "val_saliency_cosine":     avg_val["saliency_cosine"],
                    "val_recon_separation":    recon_sep,
                    "val_proxy_auc":           proxy_auc_val,
                    "rrr_skip_count":          float(self.rrr_skip_count),
                },
                step=epoch,
            )

            # ---- Early stopping + checkpoint ----
            is_best = avg_val["total"] < self._best_val_loss
            if is_best:
                self._best_val_loss = avg_val["total"]
                self._es_counter = 0
            else:
                self._es_counter += 1

            if self.checkpoint_dir:
                self._save_checkpoint(epoch, is_best=is_best)

            # ---- Update outer bar ----
            sep_str = f"{recon_sep:+.4f}" if recon_sep == recon_sep else "nan"
            epoch_bar.set_postfix(
                tr=f"{avg_tr['total']:.4f}",
                val=f"{avg_val['total']:.4f}",
                cos=f"{avg_val['saliency_cosine']:.4f}",
                sep=sep_str,
                auc=f"{proxy_auc_val:.3f}" if proxy_auc_val == proxy_auc_val else "nan",
                es=f"{self._es_counter}/{self._es_patience}",
                best=f"{self._best_val_loss:.4f}",
            )

            final_metrics = {
                "epoch":                   epoch + 1,
                "train_loss":              avg_tr["total"],
                "train_recon_loss":        avg_tr["recon"],
                "train_rrr_loss":          avg_tr["rrr"],
                "train_xai_grad_loss":     avg_tr["xai_grad"],
                "train_xai_gradcam_loss":  avg_tr["xai_gradcam"],
                "train_xai_attn_loss":     avg_tr["xai_attn"],
                "val_loss":                avg_val["total"],
                "val_recon_loss":          avg_val["recon"],
                "val_saliency_cosine":     avg_val["saliency_cosine"],
                "val_recon_separation":    recon_sep,
                "val_proxy_auc":           proxy_auc_val,
                "best_val_loss":           self._best_val_loss,
            }
            self.epoch_metrics_history.append(final_metrics)

            if self._es_counter >= self._es_patience:
                tqdm.write(
                    f"\n[Early stopping] No improvement for {self._es_patience} "
                    f"epochs. Stopped at epoch {epoch + 1}."
                )
                break

        tqdm.write(
            f"\n=== Training done ==="
            f"\n  Best val_loss : {self._best_val_loss:.5f}"
            f"\n  Checkpoint dir: {self.checkpoint_dir}"
        )
        return final_metrics
