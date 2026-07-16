import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch, numpy as np, os
from torch.utils.data import DataLoader, ConcatDataset, random_split
from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map
from losses import disp_to_depth, smoothness_loss

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE      = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE    = BASE + 'CV3Dv2/'
SAVE_PATH = BASE + 'depthnet_smooth_v2.pth'
EPOCHS    = 50; LR = 1e-4; LAMBDA_S = 0.01

ray_map = generate_ray_map(INTRINSICS)

# v2 clean + debris — giống tnet_v2all nhưng không có TNet loss
all_seqs = [V2BASE+s for s in [
    'c2_sigmoid_t1','c2_sigmoid_t2','c2_sigmoid_t3','c2_sigmoid_t4',
    'c2_sigmoidv3_t1','c2_sigmoidv3_t2','c2_sigmoidv3_t3','c2_sigmoidv3_t4']]

full_ds = ConcatDataset([C3VDDataset(p) for p in all_seqs if os.path.exists(p)])
n_train = int(len(full_ds)*0.8)
n_val   = len(full_ds)-n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)
print(f"[v2 clean+debris] Train:{n_train} Val:{n_val} | Lsup+Lsmooth only")

encoder = ResnetEncoder(18, pretrained=True).to(DEVICE)
decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': LR},
    {'params': decoder.parameters(), 'lr': LR},
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

def compute_metrics(pred, gt):
    mask = ~torch.isnan(gt) & (gt>1.0) & (gt<99.0)
    if mask.sum()<10: return None
    p=pred[mask]; g=gt[mask]
    thresh=torch.max(p/g,g/p)
    d_log=torch.log(p)-torch.log(g)
    return {
        'AbsRel': ((p-g).abs()/g).mean().item(),
        'RMSE':   ((p-g)**2).mean().sqrt().item(),
        'SILog':  ((d_log**2).mean()-0.5*(d_log.mean()**2)).item(),
    }

best_absrel = 999
for epoch in range(EPOCHS):
    encoder.train(); decoder.train()
    total_l1=total_ls=0; nb=0

    for batch in train_loader:
        t  = batch['color'].to(DEVICE)
        gt = batch['depth_gt'].to(DEVICE)
        disp_t  = decoder(encoder(t))[("disp",0)]
        depth_t = disp_to_depth(disp_t)
        mask = ~torch.isnan(gt) & (gt>1.0) & (gt<99.0)
        if mask.sum()<10: continue

        l1   = (torch.log(depth_t[mask])-torch.log(gt[mask])).abs().mean()
        ls   = smoothness_loss(depth_t, t)
        loss = l1 + LAMBDA_S * ls

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters())+list(decoder.parameters()), 1.0)
        optimizer.step()
        total_l1+=l1.item(); total_ls+=ls.item(); nb+=1

    scheduler.step()
    encoder.eval(); decoder.eval()
    all_m=[]
    with torch.no_grad():
        for batch in val_loader:
            pred=disp_to_depth(decoder(encoder(
                batch['color'].to(DEVICE)))[('disp',0)])
            m=compute_metrics(pred, batch['depth_gt'].to(DEVICE))
            if m: all_m.append(m)

    if not all_m: continue
    avg_abs=np.mean([m['AbsRel'] for m in all_m])
    avg_rms=np.mean([m['RMSE']   for m in all_m])
    nb=max(nb,1)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"L1:{total_l1/nb:.4f} | Ls:{total_ls/nb:.4f} | "
          f"AbsRel:{avg_abs:.4f} | RMSE:{avg_rms:.4f} | "
          f"LR:{scheduler.get_last_lr()[0]:.2e}")

    if avg_abs<best_absrel:
        best_absrel=avg_abs
        torch.save({'epoch':epoch,'encoder':encoder.state_dict(),
                    'decoder':decoder.state_dict(),'absrel':avg_abs}, SAVE_PATH)
        print(f"  → Saved (AbsRel={avg_abs:.4f})")

print(f"\nDone! Best AbsRel: {best_absrel:.4f}")
