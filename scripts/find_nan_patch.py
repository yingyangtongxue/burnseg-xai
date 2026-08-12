"""
Diagnostic: find which val patch(es) cause NaN in the validation loop.

Replicates exactly what _val_step does (autocast + autograd) and reports
the first patch that produces NaN in x_hat, loss_recon, or gradients.

Usage:
    python scripts/find_nan_patch.py
"""
import sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.dataset import BurnedAreaDataset

CHECKPOINT_BEST   = r"E:\experimento_queimadas\checkpoints\rrr_l0.1_mse_seed43_2026-04-27\checkpoint_best.pt"
CHECKPOINT_LATEST = r"E:\experimento_queimadas\checkpoints\rrr_l0.1_mse_seed43_2026-04-27\checkpoint_latest.pt"
SPLIT   = r"E:\experimento_queimadas\mlruns\276255147466562494\1ef5ae8d91174e92bef689b073ca6d30\artifacts\split_master.json"
DATASET = r"E:\dataset_mestrado"
DEVICE  = "cpu"
BATCH_SIZE = 2


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = Autoencoder(in_channels=21)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    epoch = ckpt.get("epoch", "?")
    nan_weights = sum(
        torch.isnan(p).any().item() for p in model.parameters()
    )
    print(f"  Loaded epoch={epoch}, NaN weights={nan_weights}")
    return model


def prepare_batch(batch, device):
    raw = batch.to(device)
    dnbr = raw[..., 20]
    x_raw = torch.cat([raw[..., :20], raw[..., 21:22]], dim=-1)
    x = x_raw.permute(0, 4, 1, 2, 3).contiguous()
    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    std  = x.std(dim=(1, 2, 3, 4),  keepdim=True) + 1e-8
    x    = (x - mean) / std
    # dNBR prior
    prior = dnbr.max(dim=1).values
    prior = torch.relu(prior)
    prior = prior * (prior > 0.1).float()
    prior = prior / prior.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    x = x.requires_grad_(True)
    return x, prior, std.squeeze()


def scan_val(model, val_loader, val_idx, device, label):
    print(f"\n{'='*60}")
    print(f"  Scanning with {label}")
    print(f"{'='*60}")
    nan_batches = []

    for batch_i, batch in enumerate(val_loader):
        x, prior, std_val = prepare_batch(batch, device)

        # Replicate _val_step exactly
        x_hat, _z = model(x)
        loss_recon = F.mse_loss(x_hat, x)

        grads = torch.autograd.grad(
            outputs=loss_recon,
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]

        sal = grads.abs().mean(dim=(1, 2))
        sal = sal / sal.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)

        loss_total = loss_recon + 0.1 * F.mse_loss(sal, prior.detach())

        # Check for NaN/inf at each stage
        flags = {
            "x_nan":        torch.isnan(x).any().item(),
            "x_inf":        torch.isinf(x).any().item(),
            "xhat_nan":     torch.isnan(x_hat).any().item(),
            "xhat_inf":     torch.isinf(x_hat).any().item(),
            "loss_nan":     math.isnan(loss_recon.item()),
            "loss_inf":     math.isinf(loss_recon.item()),
            "grads_nan":    torch.isnan(grads).any().item(),
            "grads_inf":    torch.isinf(grads).any().item(),
            "total_nan":    math.isnan(loss_total.item()),
        }

        if any(flags.values()):
            # Identify which dataset indices are in this batch
            start = batch_i * BATCH_SIZE
            patch_indices = val_idx[start: start + batch.shape[0]]
            nan_batches.append(batch_i)

            print(f"\n  *** NaN/Inf at batch {batch_i} (val positions {start}-{start+batch.shape[0]-1}) ***")
            print(f"      Dataset indices: {patch_indices}")
            print(f"      Flags: {flags}")
            print(f"      x      : min={x.min():.4f}  max={x.max():.4f}  std_raw={std_val.tolist()}")
            print(f"      x_hat  : min={x_hat.min():.4f}  max={x_hat.max():.4f}")
            print(f"      loss_recon={loss_recon.item()}")
            print(f"      prior  : max={prior.max():.4f}  nonzero={prior.gt(0).sum().item()}")

            # Per-sample breakdown
            for s in range(batch.shape[0]):
                xi = x[s]
                xhi = x_hat[s]
                print(f"      sample {s}: x=[{xi.min():.3f},{xi.max():.3f}]  "
                      f"x_hat=[{xhi.min():.3f},{xhi.max():.3f}]  "
                      f"per_sample_mse={F.mse_loss(xhi, xi).item():.6f}")

            if len(nan_batches) >= 5:
                print("\n  (stopping after 5 NaN batches)")
                break
        else:
            if batch_i % 50 == 0:
                print(f"  batch {batch_i:4d} / {len(val_loader)}: OK  "
                      f"loss={loss_recon.item():.5f}")

    if not nan_batches:
        print("\n  No NaN found in any val batch with this checkpoint.")
    else:
        print(f"\n  Summary: NaN in {len(nan_batches)} batch(es): {nan_batches}")
    return nan_batches


def main():
    print(f"Loading dataset: {DATASET}")
    dataset = BurnedAreaDataset(root_dir=DATASET, temporal_length=1)

    print(f"Loading split: {SPLIT}")
    with open(SPLIT) as f:
        split = json.load(f)
    val_idx = split.get("val", split.get("val_idx", []))
    print(f"  val size: {len(val_idx)} patches  ({len(val_idx)//BATCH_SIZE} batches)")

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,   # single-process for determinism
        pin_memory=False,
    )

    print(f"\n--- Checkpoint 1: BEST (epoch 23) ---")
    model_best = load_model(CHECKPOINT_BEST, DEVICE)
    scan_val(model_best, val_loader, val_idx, DEVICE, "checkpoint_best (epoch 23)")

    print(f"\n--- Checkpoint 2: LATEST (epoch 26) ---")
    model_latest = load_model(CHECKPOINT_LATEST, DEVICE)
    scan_val(model_latest, val_loader, val_idx, DEVICE, "checkpoint_latest (epoch 26)")


if __name__ == "__main__":
    main()
