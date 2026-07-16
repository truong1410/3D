import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map
from tnet_multiframe import TNetMultiFrame
from losses import (pose_vec_to_mat, photometric_loss,
                    depth_consistency_loss, warp_frame, disp_to_depth)
from train_improved import C3VDTripleFrame

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE      = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
SAVE_PATH = BASE + 'resnet3frame.pth'
EPOCHS    = 50; LR = 1e-4; ALPHA = 0.1

ray_map = generate_ray_map(INTRINSICS)
dataset = C3VDTripleFrame(BASE + 'cecum_t1_a')
n_train = int(len(dataset) * 0.8)
n_val   = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)
print(f"[ResNet+TNet3] Train: {n_train} | Val: {n_val}")

# Load pretrained baseline làm điểm khởi đầu
encoder = ResnetEncoder(18, pretrained=False).to(DEVICE)
decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
ckpt    = torch.load(BASE + 'baseline.pth', weights_only=False, map_location=DEVICE)
encoder.load_state_dict(ckpt['encoder'])
decoder.load_state_dict(ckpt['decoder'])
print(f"Loaded baseline (AbsRel={ckpt['absrel']:.4f})")

tnet = TNetMultiFrame().to(DEVICE)
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': LR},
    {'params': decoder.parameters(), 'lr': LR},
    {'params': tnet.parameters(),    'lr': LR},
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

def compute_metrics(pred, gt):
    mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
    if mask.sum() < 10: return None
    p = pred[mask]; g = gt[mask]
    thresh = torch.max(p/g, g/p)
    return {
        'AbsRel': ((p-g).abs()/g).mean().item(),
        'RMSE':   ((p-g)**2).mean().sqrt().item(),
        'd1':     (thresh < 1.25).float().mean().item()
    }

best_absrel = 999
for epoch in range(EPOCHS):
    encoder.train(); decoder.train(); tnet.train()
    total_l1 = total_lp = total_lc = 0; nb = 0

    for batch in train_loader:
        tm1 = batch['color_tm1'].to(DEVICE)
        t   = batch['color_t'].to(DEVICE)
        tp1 = batch['color_tp1'].to(DEVICE)
        gt  = batch['depth_gt'].to(DEVICE)

        disp_t   = decoder(encoder(t))[("disp", 0)]
        depth_t  = disp_to_depth(disp_t)
        depth_tp1 = disp_to_depth(decoder(encoder(tp1))[("disp", 0)])
        depth_tm1 = disp_to_depth(decoder(encoder(tm1))[("disp", 0)])

        pose_back, pose_fwd = tnet(tm1, t, tp1)
        T_back = pose_vec_to_mat(pose_back)
        T_fwd  = pose_vec_to_mat(pose_fwd)

        t_from_tm1 = warp_frame(tm1, depth_t, T_back, ray_map)
        t_from_tp1 = warp_frame(tp1, depth_t, T_fwd,  ray_map)

        mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
        if mask.sum() < 10: continue

        l1    = (torch.log(depth_t[mask]) - torch.log(gt[mask])).abs().mean()
        lp    = torch.min(photometric_loss(t, t_from_tm1),
                          photometric_loss(t, t_from_tp1))
        lcons = (depth_consistency_loss(depth_t, depth_tm1, T_back, ray_map) +
                 depth_consistency_loss(depth_t, depth_tp1, T_fwd,  ray_map)) / 2
        loss  = l1 + lp + ALPHA * lcons

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()) +
            list(tnet.parameters()), 1.0)
        optimizer.step()
        total_l1 += l1.item(); total_lp += lp.item()
        total_lc += lcons.item(); nb += 1

    scheduler.step()
    encoder.eval(); decoder.eval(); tnet.eval()
    all_m = []
    with torch.no_grad():
        for batch in val_loader:
            rgb  = batch['color_t'].to(DEVICE)
            gt   = batch['depth_gt'].to(DEVICE)
            pred = disp_to_depth(decoder(encoder(rgb))[('disp',0)])
            m = compute_metrics(pred, gt)
            if m: all_m.append(m)

    if not all_m: continue
    avg_abs = np.mean([m['AbsRel'] for m in all_m])
    avg_rms = np.mean([m['RMSE']   for m in all_m])
    avg_d1  = np.mean([m['d1']     for m in all_m])
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"L1:{total_l1/max(nb,1):.4f} | "
          f"AbsRel:{avg_abs:.4f} | RMSE:{avg_rms:.4f} | "
          f"d1:{avg_d1:.4f} | LR:{scheduler.get_last_lr()[0]:.2e}")
    if avg_abs < best_absrel:
        best_absrel = avg_abs
        torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(), 'tnet': tnet.state_dict(),
                    'absrel': avg_abs}, SAVE_PATH)
        print(f"  → Saved (AbsRel={avg_abs:.4f})")

print(f"Done! Best AbsRel: {best_absrel:.4f}")
