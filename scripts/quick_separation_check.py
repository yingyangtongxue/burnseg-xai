"""
Quick diagnostic: recon_separation + proxy_auc on best checkpoint (CPU).
Answers: is the model actually discriminating burned vs clean patches?

Usage:
    python scripts/quick_separation_check.py
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "src")
from burnseg_xai.models.autoencoder import Autoencoder
from burnseg_xai.dataset import BurnedAreaDataset

CHECKPOINT = r"E:\experimento_queimadas\checkpoints\rrr_l0.1_mse_seed43_2026-04-27\checkpoint_best.pt"
SPLIT      = r"E:\experimento_queimadas\mlruns\276255147466562494\1ef5ae8d91174e92bef689b073ca6d30\artifacts\split_master.json"
DATASET    = r"E:\dataset_mestrado"
DEVICE     = "cpu"   # avoid conflict with GPU training
DNBR_THR   = 0.1


def prepare(batch, device):
    raw = batch.to(device)
    dnbr = raw[..., 20]
    x_raw = torch.cat([raw[..., :20], raw[..., 21:22]], dim=-1)
    x = x_raw.permute(0, 4, 1, 2, 3).contiguous()
    mean = x.mean(dim=(1,2,3,4), keepdim=True)
    std  = x.std(dim=(1,2,3,4),  keepdim=True) + 1e-8
    x    = (x - mean) / std
    mean_dnbr = dnbr.mean(dim=(1,2,3))
    return x, mean_dnbr


def main():
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model = Autoencoder(in_channels=21)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  epoch={ckpt['epoch']}  best_val_loss={ckpt['best_val_loss']:.5f}")

    print(f"Loading dataset: {DATASET}")
    dataset = BurnedAreaDataset(root_dir=DATASET, temporal_length=1)

    print(f"Loading split: {SPLIT}")
    with open(SPLIT) as f:
        split = json.load(f)

    # Keys may be val/test or val_idx/test_idx depending on which pipeline created the split
    key_map = {"val": "val_idx", "test": "test_idx"}
    for split_name in ("val", "test"):
        indices = split.get(split_name, split.get(key_map[split_name], []))
        indices = indices[:200]  # cap for speed on CPU
        if not indices:
            continue

        errors, labels = [], []
        for idx in tqdm(indices, desc=f"  {split_name} ({len(indices)} patches)"):
            raw = dataset[idx]
            if raw.dim() == 4:
                raw = raw.unsqueeze(0)
            x, mean_dnbr = prepare(raw, DEVICE)
            with torch.no_grad():
                x_hat, _ = model(x)
                err = F.mse_loss(x_hat, x, reduction="none").mean().item()
            errors.append(err)
            labels.append(1 if mean_dnbr.item() > DNBR_THR else 0)

        errors = np.array(errors)
        labels = np.array(labels)
        n_burned = labels.sum()
        n_clean  = (1-labels).sum()

        print(f"\n{'='*55}")
        print(f"  Split : {split_name}  ({len(indices)} patches)")
        print(f"  Burned proxy (dNBR>{DNBR_THR}): {n_burned}  Clean: {n_clean}")
        print(f"  Mean error ALL    : {errors.mean():.5f}")
        if n_burned > 0:
            print(f"  Mean error BURNED : {errors[labels==1].mean():.5f}")
        if n_clean > 0:
            print(f"  Mean error CLEAN  : {errors[labels==0].mean():.5f}")
        if n_burned > 0 and n_clean > 0:
            sep = errors[labels==1].mean() - errors[labels==0].mean()
            print(f"  recon_separation  : {sep:+.5f}  ({'✓ model discriminates' if sep > 0 else '✗ NO discrimination: all-black images expected'})")
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels, errors)
                print(f"  proxy_AUC         : {auc:.4f}  ({'✓ above chance' if auc > 0.5 else '✗ below chance'})")
            except Exception as e:
                print(f"  proxy_AUC         : error ({e})")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
