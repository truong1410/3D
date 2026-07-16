import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch, numpy as np, os
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map
from tnet import TNet
from losses import (pose_vec_to_mat, photometric_loss,
                    depth_consistency_loss, warp_frame,
                    disp_to_depth, smoothness_loss)

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE      = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE    = BASE + 'CV3Dv2/'
SAVE_PATH = BASE + 'tnet_joint.pth'
EPOCHS    = 50; LR = 1e-4
ALPHA     = 0.1   # Lcons weight
LAMBDA_S  = 0.01  # Lsmooth weight
LAMBDA_R  = 0.05  # Lrecon weight — joint optimization

ray_map = generate_ray_map(INTRINSICS)

# ── Dataset wrapper trả về 4 frame liên tiếp ─────────────────
class C3VDQuadFrame(Dataset):
    """
    Trả về 4 frame liên tiếp: t-1, t, t+1, t+2
    Cho phép tính multi-step consistency loss
    """
    def __init__(self, data_dir, size=(640, 512)):
        self.ds = C3VDDataset(data_dir, size)

    def __len__(self):
        return len(self.ds) - 2  # cần t-1, t, t+1, t+2

    def __getitem__(self, idx):
        b0 = self.ds[idx]       # color=t-1, color_next=t
        b1 = self.ds[idx+1]     # color=t,   color_next=t+1
        b2 = self.ds[idx+2]     # color=t+1, color_next=t+2
        return {
            'color_tm1': b0['color'],        # t-1
            'color_t':   b0['color_next'],   # t
            'color_tp1': b1['color_next'],   # t+1
            'color_tp2': b2['color_next'],   # t+2
            'depth_gt':  b1['depth_gt'],     # GT depth tại t
        }

# V2 clean + debris sequences
clean_seqs  = [V2BASE+s for s in ['c2_sigmoid_t1','c2_sigmoid_t2','c2_sigmoid_t3']]
debris_seqs = [V2BASE+s for s in ['c2_sigmoidv3_t1','c2_sigmoidv3_t2','c2_sigmoidv3_t3']]
val_seqs    = [V2BASE+s for s in ['c2_sigmoid_t4','c2_sigmoidv3_t4']]

train_ds = ConcatDataset([C3VDQuadFrame(p) for p in clean_seqs+debris_seqs
                          if os.path.exists(p)])
val_ds   = ConcatDataset([C3VDDataset(p)   for p in val_seqs
                          if os.path.exists(p)])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {DEVICE}")

encoder = ResnetEncoder(18, pretrained=True).to(DEVICE)
decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
tnet    = TNet().to(DEVICE)

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

def multi_step_consistency_loss(encoder, decoder, tnet,
                                tm1, t, tp1, tp2, ray_map):
    """
    Joint depth-pose consistency trên chuỗi 4 frame.

    Ý tưởng: tích lũy pose qua 3 bước t-1→t→t+1→t+2
    depth tại t+2 predicted trực tiếp phải khớp với
    depth tại t+2 warped từ t qua 2 bước pose liên tiếp.
    → Buộc pose và depth nhất quán trên chuỗi dài hơn
    """
    # Predict depth tất cả frames
    d_tm1 = disp_to_depth(decoder(encoder(tm1))[('disp',0)])
    d_t   = disp_to_depth(decoder(encoder(t))[('disp',0)])
    d_tp1 = disp_to_depth(decoder(encoder(tp1))[('disp',0)])
    d_tp2 = disp_to_depth(decoder(encoder(tp2))[('disp',0)])

    # Predict poses
    pose_01 = tnet(tm1, t)    # t-1 → t
    pose_12 = tnet(t, tp1)    # t   → t+1
    pose_23 = tnet(tp1, tp2)  # t+1 → t+2

    T_01 = pose_vec_to_mat(pose_01)
    T_12 = pose_vec_to_mat(pose_12)
    T_23 = pose_vec_to_mat(pose_23)

    # Photometric losses (single step)
    t_from_tm1 = warp_frame(tm1, d_t,   T_01, ray_map)
    t_from_tp1 = warp_frame(tp1, d_t,   T_12, ray_map)
    lp = (photometric_loss(t, t_from_tm1) +
          photometric_loss(t, t_from_tp1)) / 2

    # Depth consistency (single step)
    lc = (depth_consistency_loss(d_t, d_tm1, T_01, ray_map) +
          depth_consistency_loss(d_t, d_tp1, T_12, ray_map)) / 2

    # ── Joint reconstruction loss ─────────────────────────────
    # Compose pose: t → t+2 qua 2 bước
    T_02 = torch.bmm(T_23, T_12)  # T_{t→t+2} = T_{t+1→t+2} @ T_{t→t+1}

    # Warp tp2 về frame t dùng composed pose
    tp2_warped = warp_frame(tp2, d_t, T_02, ray_map)

    # Photometric loss multi-step
    lp_multi = photometric_loss(t, tp2_warped)

    # Depth warped từ tp2 về t
    B, _, H, W = d_t.shape
    import torch.nn.functional as F
    ray = torch.from_numpy(ray_map).float().to(d_t.device)
    ray = F.interpolate(ray.permute(2,0,1).unsqueeze(0),
                        size=(H,W), mode='bilinear',
                        align_corners=True).squeeze(0).permute(1,2,0)
    ray = ray.unsqueeze(0).repeat(B,1,1,1)

    d    = d_t[:,0]
    rx, ry, rz = ray[...,0], ray[...,1], ray[...,2]
    x3d = (d*rx)/rz; y3d = (d*ry)/rz; z3d = d
    ones = torch.ones_like(z3d)
    pts  = torch.stack([x3d,y3d,z3d,ones],dim=-1).reshape(B,-1,4)
    pts_tp2 = torch.bmm(pts, T_02.transpose(1,2))
    z_proj  = pts_tp2[:,:,2].reshape(B,1,H,W)

    valid = (z_proj>0) & ~torch.isnan(d_t) & ~torch.isnan(z_proj)
    if valid.sum() > 10:
        diff   = (d_t - z_proj).abs()
        denom  = d_t + z_proj + 1e-7
        lc_multi = (diff/denom)[valid].mean()
    else:
        lc_multi = torch.tensor(0.0, device=d_t.device)

    # Smoothness
    ls = smoothness_loss(d_t, t)

    return lp, lc, lp_multi, lc_multi, ls, d_t

# ── Train loop ────────────────────────────────────────────────
best_absrel = 999
for epoch in range(EPOCHS):
    encoder.train(); decoder.train(); tnet.train()
    totals = {'l1':0,'lp':0,'lc':0,'lp_m':0,'lc_m':0,'ls':0}
    nb = 0

    for batch in train_loader:
        tm1 = batch['color_tm1'].to(DEVICE)
        t   = batch['color_t'].to(DEVICE)
        tp1 = batch['color_tp1'].to(DEVICE)
        tp2 = batch['color_tp2'].to(DEVICE)
        gt  = batch['depth_gt'].to(DEVICE)

        lp, lc, lp_multi, lc_multi, ls, depth_t = \
            multi_step_consistency_loss(
                encoder, decoder, tnet,
                tm1, t, tp1, tp2, ray_map)

        mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
        if mask.sum() < 10: continue

        l1 = (torch.log(depth_t[mask]) -
              torch.log(gt[mask])).abs().mean()

        # Joint loss — kết hợp single-step và multi-step
        loss = (l1 +
                lp + ALPHA * lc + LAMBDA_S * ls +
                LAMBDA_R * (lp_multi + ALPHA * lc_multi))

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(decoder.parameters()) +
            list(tnet.parameters()), 1.0)
        optimizer.step()

        totals['l1']   += l1.item()
        totals['lp']   += lp.item()
        totals['lc']   += lc.item()
        totals['lp_m'] += lp_multi.item()
        totals['lc_m'] += lc_multi.item()
        totals['ls']   += ls.item()
        nb += 1

    scheduler.step()
    encoder.eval(); decoder.eval(); tnet.eval()
    all_m = []
    with torch.no_grad():
        for batch in val_loader:
            pred = disp_to_depth(decoder(encoder(
                batch['color'].to(DEVICE)))[('disp',0)])
            m = compute_metrics(pred, batch['depth_gt'].to(DEVICE))
            if m: all_m.append(m)

    if not all_m: continue
    avg_abs = np.mean([m['AbsRel'] for m in all_m])
    avg_rms = np.mean([m['RMSE']   for m in all_m])
    avg_d1  = np.mean([m['d1']     for m in all_m])
    nb = max(nb, 1)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"L1:{totals['l1']/nb:.4f} | "
          f"Lp:{totals['lp']/nb:.4f} | "
          f"Lc:{totals['lc']/nb:.4f} | "
          f"Lpm:{totals['lp_m']/nb:.4f} | "
          f"Lcm:{totals['lc_m']/nb:.4f} | "
          f"AbsRel:{avg_abs:.4f} | "
          f"RMSE:{avg_rms:.4f} | "
          f"d1:{avg_d1:.4f} | "
          f"LR:{scheduler.get_last_lr()[0]:.2e}")

    if avg_abs < best_absrel:
        best_absrel = avg_abs
        torch.save({
            'epoch':   epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'tnet':    tnet.state_dict(),
            'absrel':  avg_abs
        }, SAVE_PATH)
        print(f"  → Saved (AbsRel={avg_abs:.4f})")

print(f"\nDone! Best AbsRel: {best_absrel:.4f}")
