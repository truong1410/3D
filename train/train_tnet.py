import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map
from tnet import TNet
from losses import (pose_vec_to_mat, photometric_loss,
                    depth_consistency_loss, warp_frame, disp_to_depth)

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS     = 50
LR         = 1e-4
BASELINE   = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/baseline.pth'
SAVE_PATH  = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/tnet_final.pth'
RESUME     = True   # ← train tiếp từ checkpoint
ALPHA      = 0.1

ray_map = generate_ray_map(INTRINSICS)

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

encoder = ResnetEncoder(18, pretrained=False).to(DEVICE)
decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
tnet    = TNet().to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': LR},
    {'params': decoder.parameters(), 'lr': LR},
    {'params': tnet.parameters(),    'lr': LR},
])
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=15, gamma=0.5)

# ── Resume từ checkpoint ──────────────────────────────
start_epoch  = 0
best_absrel  = 999

if RESUME and torch.cuda.is_available():
    import os
    if os.path.exists(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        tnet.load_state_dict(ckpt['tnet'])
        start_epoch = ckpt['epoch'] + 1
        best_absrel = ckpt['absrel']
        # Advance scheduler đến đúng epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed từ epoch {start_epoch} | Best AbsRel: {best_absrel:.4f}")
    else:
        # Không có checkpoint TNet → load baseline
        ckpt = torch.load(BASELINE, weights_only=False, map_location=DEVICE)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        print(f"Loaded baseline (AbsRel={ckpt['absrel']:.4f})")
else:
    ckpt = torch.load(BASELINE, weights_only=False, map_location=DEVICE)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    print(f"Loaded baseline (AbsRel={ckpt['absrel']:.4f})")

def compute_metrics(pred_depth, gt):
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

# ── Train loop từ start_epoch ─────────────────────────
for epoch in range(start_epoch, EPOCHS):
    encoder.train(); decoder.train(); tnet.train()
    total_l1 = total_lp = total_lc = 0
    n_batch  = 0

    for batch in train_loader:
        t   = batch['color'].to(DEVICE)
        t1  = batch['color_next'].to(DEVICE)
        gt  = batch['depth_gt'].to(DEVICE)

        disp_t   = decoder(encoder(t))[("disp", 0)]
        depth_t  = disp_to_depth(disp_t)
        disp_t1  = decoder(encoder(t1))[("disp", 0)]
        depth_t1 = disp_to_depth(disp_t1)

        pose_vec  = tnet(t, t1)
        T         = pose_vec_to_mat(pose_vec)
        t1_warped = warp_frame(t1, depth_t, T, ray_map)

        mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
        if mask.sum() < 10:
            continue

        l1    = (torch.log(depth_t[mask]) -
                 torch.log(gt[mask])).abs().mean()
        lp    = photometric_loss(t, t1_warped)
        lcons = depth_consistency_loss(depth_t, depth_t1, T, ray_map)
        loss  = l1 + lp + ALPHA * lcons

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(decoder.parameters()) +
            list(tnet.parameters()), max_norm=1.0)
        optimizer.step()

        total_l1 += l1.item()
        total_lp += lp.item()
        total_lc += lcons.item()
        n_batch  += 1

    scheduler.step()

    encoder.eval(); decoder.eval(); tnet.eval()
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
        continue

    avg_absrel = np.mean([m['AbsRel'] for m in all_metrics])
    avg_rmse   = np.mean([m['RMSE']   for m in all_metrics])
    avg_d1     = np.mean([m['d1']     for m in all_metrics])
    nb         = max(n_batch, 1)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"L1: {total_l1/nb:.4f} | "
          f"Lp: {total_lp/nb:.4f} | "
          f"Lc: {total_lc/nb:.4f} | "
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
            'tnet':    tnet.state_dict(),
            'absrel':  avg_absrel
        }, SAVE_PATH)
        print(f"  → Saved (AbsRel={avg_absrel:.4f})")

print(f"\nDone! Best AbsRel: {best_absrel:.4f}")
print(f"\nSo sánh:")
print(f"  Baseline (không TNet): 0.0193")
print(f"  Với TNet:              {best_absrel:.4f}")
