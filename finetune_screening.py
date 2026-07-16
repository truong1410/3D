import sys, torch, numpy as np, cv2, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from tnet import TNet
from losses import (disp_to_depth, photometric_loss,
                   depth_consistency_loss, smoothness_loss,
                   pose_vec_to_mat, warp_frame)
from c3vd_dataset import INTRINSICS, generate_ray_map
ray_map = generate_ray_map(INTRINSICS)
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE   = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
SCREEN = BASE + 'Screening/'
SAVE   = BASE + 'tnet_screening_ft.pth'

W, H = 640, 512
EPOCHS   = 30
LR       = 1e-5    # nhỏ hơn vì fine-tune
LAMBDA_S = 0.01
LAMBDA_P = 1.0
LAMBDA_C = 0.1

# ── Dataset — screening RGB only ─────────────────────────
class ScreeningDataset(Dataset):
    def __init__(self, seqs, step=3):
        self.pairs = []
        for seq_dir in seqs:
            rgb_dir = os.path.join(seq_dir, 'rgb')
            if not os.path.exists(rgb_dir): continue
            frames = sorted(os.listdir(rgb_dir),
                           key=lambda x: int(x.replace('.png','')))
            for i in range(0, len(frames)-step, 1):
                self.pairs.append((
                    os.path.join(rgb_dir, frames[i]),
                    os.path.join(rgb_dir, frames[i+step])
                ))
        print(f'Screening pairs: {len(self.pairs)}')

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        def load(p):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))
            return torch.from_numpy(
                img.astype(np.float32)/255.0).permute(2,0,1)
        t  = load(self.pairs[idx][0])
        t1 = load(self.pairs[idx][1])
        return {'color': t, 'color_next': t1}

# Dùng t1+t2+t3 để train, t4 để val
train_seqs = [SCREEN+s for s in [
    'c0_full_t1_v1','c0_full_t2_v1','c0_full_t3_v1']]
val_seqs   = [SCREEN+'c0_full_t4_v1']

train_ds = ScreeningDataset(train_seqs, step=3)
val_ds   = ScreeningDataset(val_seqs,   step=3)
train_loader = DataLoader(train_ds, batch_size=4,
                          shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=4,
                          shuffle=False, num_workers=2)

# ── Load pretrained weights ───────────────────────────────
enc  = ResnetEncoder(18, pretrained=False).to(DEVICE)
dec  = DepthDecoder(enc.num_ch_enc).to(DEVICE)
tnet = TNet().to(DEVICE)

# Load tnet_smooth (best phantom model)
ck = torch.load(BASE+'tnet_smooth.pth',
                weights_only=False, map_location=DEVICE)
enc.load_state_dict(ck['encoder'])
dec.load_state_dict(ck['decoder'])
tnet.load_state_dict(ck['tnet'])
print('Loaded tnet_smooth.pth — fine-tuning on real screening data')

optimizer = torch.optim.Adam([
    {'params': enc.parameters(),  'lr': LR},
    {'params': dec.parameters(),  'lr': LR},
    {'params': tnet.parameters(), 'lr': LR},
])
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)

# ── Training — self-supervised only (no GT depth) ────────
best_loss = 999

for epoch in range(EPOCHS):
    enc.train(); dec.train(); tnet.train()
    total_lp=total_ls=total_lc=0; nb=0

    for batch in train_loader:
        t  = batch['color'].to(DEVICE)
        t1 = batch['color_next'].to(DEVICE)

        # Depth từ DepthNet
        disp_t  = dec(enc(t))[('disp',0)]
        depth_t = disp_to_depth(disp_t)

        # Pose từ TNet
        pose_vec = tnet(t, t1)
        T_rel    = pose_vec_to_mat(pose_vec)

        # Warp t1 → t (photometric)
        t1_warped = warp_frame(t1, depth_t, T_rel, ray_map)

        # Losses — NO Lsup (không có GT depth)
        lp = photometric_loss(t, t1_warped)
        ls = smoothness_loss(depth_t, t)

        # Depth consistency
        disp_t1  = dec(enc(t1))[('disp',0)]
        depth_t1 = disp_to_depth(disp_t1)
        lc = depth_consistency_loss(depth_t, depth_t1, T_rel, ray_map)

        loss = (LAMBDA_P * lp +
                LAMBDA_S * ls +
                LAMBDA_C * lc)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) +
            list(dec.parameters()) +
            list(tnet.parameters()), 1.0)
        optimizer.step()

        total_lp+=lp.item()
        total_ls+=ls.item()
        total_lc+=lc.item()
        nb+=1

    scheduler.step()
    nb = max(nb,1)

    # Val loss
    enc.eval(); dec.eval(); tnet.eval()
    val_lp=0; vb=0
    with torch.no_grad():
        for batch in val_loader:
            t  = batch['color'].to(DEVICE)
            t1 = batch['color_next'].to(DEVICE)
            depth_t  = disp_to_depth(dec(enc(t))[('disp',0)])
            pose_vec = tnet(t, t1)
            T_rel    = pose_vec_to_mat(pose_vec)
            t1_w = warp_frame(t1, depth_t, T_rel, ray_map)
            val_lp += photometric_loss(t, t1_w).item()
            vb += 1
    vb = max(vb,1)
    val_loss = val_lp/vb

    print(f'Epoch {epoch+1:3d}/{EPOCHS} | '
          f'Lp:{total_lp/nb:.4f} '
          f'Ls:{total_ls/nb:.4f} '
          f'Lc:{total_lc/nb:.4f} | '
          f'Val Lp:{val_loss:.4f} | '
          f'LR:{scheduler.get_last_lr()[0]:.1e}')

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'epoch':   epoch,
            'encoder': enc.state_dict(),
            'decoder': dec.state_dict(),
            'tnet':    tnet.state_dict(),
            'val_lp':  val_loss,
        }, SAVE)
        print(f'  → Saved (val_lp={val_loss:.4f})')

print(f'\nDone! Best val_lp: {best_loss:.4f}')
print(f'Saved: {SAVE}')
