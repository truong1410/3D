# train_baseline.py — fixed
import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS     = 50
LR         = 1e-4
SAVE_PATH  = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/baseline.pth'

# Dataset
dataset = C3VDDataset()
n_train = int(len(dataset) * 0.8)
n_val   = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

print(f"Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

# Model
encoder = ResnetEncoder(18, pretrained=True).to(DEVICE)
decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=15, gamma=0.5)

# ── FIX: chuyển disp → depth đúng scale ──────────────
MIN_DEPTH = 0.1   # mm
MAX_DEPTH = 100.0 # mm

def disp_to_depth(disp):
    """
    disp ∈ (0,1) → depth ∈ (MIN_DEPTH, MAX_DEPTH) mm
    depth = 1 / (disp * (1/MIN - 1/MAX) + 1/MAX)
    """
    min_disp = 1.0 / MAX_DEPTH
    max_disp = 1.0 / MIN_DEPTH
    scaled   = min_disp + (max_disp - min_disp) * disp
    return 1.0 / scaled  # mm

def compute_metrics(pred_depth, gt):
    """pred_depth và gt đều đơn vị mm"""
    mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
    if mask.sum() < 10:
        return None
    p = pred_depth[mask]
    g = gt[mask]
    thresh  = torch.max(p / g, g / p)
    abs_rel = ((p - g).abs() / g).mean()
    rmse    = ((p - g)**2).mean().sqrt()
    delta1  = (thresh < 1.25).float().mean()
    return {
        'AbsRel': abs_rel.item(),
        'RMSE':   rmse.item(),
        'd1':     delta1.item()
    }

# ── Train loop ────────────────────────────────────────
best_absrel = 999
for epoch in range(EPOCHS):
    encoder.train(); decoder.train()
    train_loss = 0; n_batch = 0

    for batch in train_loader:
        rgb = batch['color'].to(DEVICE)
        gt  = batch['depth_gt'].to(DEVICE)   # mm

        disp       = decoder(encoder(rgb))[("disp", 0)]
        pred_depth = disp_to_depth(disp)      # → mm

        # Chỉ tính loss ở pixel valid
        mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
        if mask.sum() < 10:
            continue

        # Log scale loss — ổn định hơn L1 cho depth
        loss = (torch.log(pred_depth[mask]) -
                torch.log(gt[mask])).abs().mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(decoder.parameters()), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        n_batch    += 1

    scheduler.step()

    # Validate
    encoder.eval(); decoder.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in val_loader:
            rgb  = batch['color'].to(DEVICE)
            gt   = batch['depth_gt'].to(DEVICE)
            disp = decoder(encoder(rgb))[("disp", 0)]
            pred = disp_to_depth(disp)
            m    = compute_metrics(pred, gt)
            if m:
                all_metrics.append(m)

    if not all_metrics:
        print(f"Epoch {epoch+1:3d} — no valid metrics")
        continue

    avg_absrel = np.mean([m['AbsRel'] for m in all_metrics])
    avg_rmse   = np.mean([m['RMSE']   for m in all_metrics])
    avg_d1     = np.mean([m['d1']     for m in all_metrics])
    avg_loss   = train_loss / max(n_batch, 1)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"AbsRel: {avg_absrel:.4f} | "
          f"RMSE: {avg_rmse:.4f} | "
          f"d1: {avg_d1:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.2e}")

    if avg_absrel < best_absrel:
        best_absrel = avg_absrel
        torch.save({
            'epoch':   epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'absrel':  avg_absrel
        }, SAVE_PATH)
        print(f"  → Saved (AbsRel={avg_absrel:.4f})")

print(f"\nDone! Best AbsRel: {best_absrel:.4f}")
