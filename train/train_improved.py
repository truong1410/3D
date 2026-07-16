import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import os

from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map
from tnet_multiframe import TNetMultiFrame
from depthnet_mobile import MobileDepthNet
from losses import (pose_vec_to_mat, photometric_loss,
                    depth_consistency_loss, warp_frame, disp_to_depth)

# ── Config ──────────────────────────────────────────
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS     = 50
LR         = 1e-4
BASE       = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
SAVE_PATH  = BASE + 'improved_final.pth'
ALPHA      = 0.1
# ────────────────────────────────────────────────────

# Dataset wrapper để trả về 3 frame liên tiếp
class C3VDTripleFrame(Dataset):
    """Wrap C3VDDataset để trả về frame t-1, t, t+1"""
    def __init__(self, data_dir, size=(640, 512)):
        self.ds = C3VDDataset(data_dir, size)

    def __len__(self):
        # Bỏ frame đầu (không có t-1) và frame cuối (không có t+1)
        return len(self.ds) - 1

    def __getitem__(self, idx):
        # idx+1 vì C3VDDataset[i] trả về (i, i+1)
        # Ta cần (i, i+1, i+2) → lấy từ ds[idx] và ds[idx+1]
        b0 = self.ds[idx]      # color=t-1, color_next=t
        b1 = self.ds[idx + 1]  # color=t,   color_next=t+1
        return {
            'color_tm1':  b0['color'],        # frame t-1
            'color_t':    b0['color_next'],   # frame t
            'color_tp1':  b1['color_next'],   # frame t+1
            'depth_gt':   b0['depth_gt'],     # depth GT tại t-1
        }

ray_map = generate_ray_map(INTRINSICS)

# Dataset
dataset  = C3VDTripleFrame(BASE + 'cecum_t1_a')
n_train  = int(len(dataset) * 0.8)
n_val    = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

print(f"Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

# Models — MobileDepthNet + TNetMultiFrame
depthnet = MobileDepthNet(pretrained=True).to(DEVICE)
tnet     = TNetMultiFrame().to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': depthnet.parameters(), 'lr': LR},
    {'params': tnet.parameters(),     'lr': LR},
])
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=15, gamma=0.5)

def compute_metrics(pred_depth, gt):
    mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
    if mask.sum() < 10:
        return None
    p = pred_depth[mask]; g = gt[mask]
    thresh  = torch.max(p/g, g/p)
    abs_rel = ((p-g).abs()/g).mean()
    rmse    = ((p-g)**2).mean().sqrt()
    delta1  = (thresh < 1.25).float().mean()
    return {
        'AbsRel': abs_rel.item(),
        'RMSE':   rmse.item(),
        'd1':     delta1.item()
    }

# ── Train loop ────────────────────────────────────────
best_absrel = 999
for epoch in range(EPOCHS):
    depthnet.train(); tnet.train()
    total_l1 = total_lp = total_lc = 0
    n_batch  = 0

    for batch in train_loader:
        tm1 = batch['color_tm1'].to(DEVICE)   # frame t-1
        t   = batch['color_t'].to(DEVICE)     # frame t
        tp1 = batch['color_tp1'].to(DEVICE)   # frame t+1
        gt  = batch['depth_gt'].to(DEVICE)    # depth GT

        # Depth prediction
        disp_t   = depthnet(t)[("disp", 0)]
        depth_t  = disp_to_depth(disp_t)

        disp_tp1  = depthnet(tp1)[("disp", 0)]
        depth_tp1 = disp_to_depth(disp_tp1)

        disp_tm1  = depthnet(tm1)[("disp", 0)]
        depth_tm1 = disp_to_depth(disp_tm1)

        # TNet predict 2 poses cùng lúc
        pose_back, pose_fwd = tnet(tm1, t, tp1)
        T_back = pose_vec_to_mat(pose_back)  # t-1 → t
        T_fwd  = pose_vec_to_mat(pose_fwd)   # t → t+1

        # Warp từ 2 hướng → photometric loss mạnh hơn
        t_warped_from_tm1 = warp_frame(tm1, depth_t,   T_back, ray_map)
        t_warped_from_tp1 = warp_frame(tp1, depth_t,   T_fwd,  ray_map)

        # Supervised loss
        mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
        if mask.sum() < 10:
            continue

        l1 = (torch.log(depth_t[mask]) -
              torch.log(gt[mask])).abs().mean()

        # Photometric từ 2 hướng — lấy min để tránh occlusion
        lp_back = photometric_loss(t, t_warped_from_tm1)
        lp_fwd  = photometric_loss(t, t_warped_from_tp1)
        lp      = torch.min(lp_back, lp_fwd)  # min reprojection

        # Depth consistency từ 2 hướng
        lc_back = depth_consistency_loss(depth_t, depth_tm1, T_back, ray_map)
        lc_fwd  = depth_consistency_loss(depth_t, depth_tp1, T_fwd,  ray_map)
        lcons   = (lc_back + lc_fwd) / 2

        loss = l1 + lp + ALPHA * lcons

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(depthnet.parameters()) +
            list(tnet.parameters()), max_norm=1.0)
        optimizer.step()

        total_l1 += l1.item()
        total_lp += lp.item()
        total_lc += lcons.item()
        n_batch  += 1

    scheduler.step()

    # Validate
    depthnet.eval(); tnet.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in val_loader:
            t    = batch['color_t'].to(DEVICE)
            gt   = batch['depth_gt'].to(DEVICE)
            pred = disp_to_depth(depthnet(t)[('disp', 0)])
            m    = compute_metrics(pred, gt)
            if m: all_metrics.append(m)

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
            'epoch':    epoch,
            'depthnet': depthnet.state_dict(),
            'tnet':     tnet.state_dict(),
            'absrel':   avg_absrel
        }, SAVE_PATH)
        print(f"  → Saved (AbsRel={avg_absrel:.4f})")

print(f"\nDone! Best AbsRel: {best_absrel:.4f}")
