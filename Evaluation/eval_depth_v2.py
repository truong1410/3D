import sys, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset
from losses import disp_to_depth
from torch.utils.data import DataLoader, ConcatDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE   = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE = BASE + 'CV3Dv2/'

def load_model(ckpt):
    enc = ResnetEncoder(18, pretrained=False).to(DEVICE)
    dec = DepthDecoder(enc.num_ch_enc).to(DEVICE)
    ck  = torch.load(ckpt, weights_only=False, map_location=DEVICE)
    enc.load_state_dict(ck['encoder']); dec.load_state_dict(ck['decoder'])
    enc.eval(); dec.eval()
    return enc, dec

sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
    dtype=torch.float32).view(1,1,3,3).to(DEVICE)
sobel_y = sobel_x.transpose(2,3)

def compute_metrics(pred, gt):
    mask = ~torch.isnan(gt) & (gt>1.0) & (gt<99.0)
    if mask.sum()<10: return None
    p=pred[mask]; g=gt[mask]
    thresh = torch.max(p/g,g/p)
    d_log  = torch.log(p)-torch.log(g)
    pred_m = pred.clone(); pred_m[~mask]=0
    gt_m   = gt.clone();   gt_m[~mask]=0
    gx_p=F.conv2d(pred_m,sobel_x,padding=1)
    gy_p=F.conv2d(pred_m,sobel_y,padding=1)
    gx_g=F.conv2d(gt_m,  sobel_x,padding=1)
    gy_g=F.conv2d(gt_m,  sobel_y,padding=1)
    n=min(5000,len(p)); idx=torch.randperm(len(p))[:n*2]
    return {
        'AbsRel':  ((p-g).abs()/g).mean().item(),
        'RMSE':    ((p-g)**2).mean().sqrt().item(),
        'SILog':   ((d_log**2).mean()-0.5*(d_log.mean()**2)).item(),
        'GradErr': ((gx_p-gx_g).abs()+(gy_p-gy_g).abs()).mean().item()/2,
        'Ordinal': (torch.sign(p[idx[:n]]-p[idx[n:]])==
                    torch.sign(g[idx[:n]]-g[idx[n:]])).float().mean().item(),
    }

# Val set — v2 clean t4 + debris t4
val_seqs = [V2BASE+s for s in ['c2_sigmoid_t4','c2_sigmoidv3_t4']]
val_ds   = ConcatDataset([C3VDDataset(p) for p in val_seqs])
loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

models = {
    'Monodepth2':  BASE+'baseline.pth',
    'TNet':        BASE+'tnet_final.pth',
    'TNet+Smooth': BASE+'tnet_smooth.pth',
    'TNet v2all':  BASE+'tnet_v2all.pth',
}

print()
print('='*78)
print('Depth Estimation — v2 val set (c2_sigmoid_t4 + c2_sigmoidv3_t4)')
print(f"{'Method':<18} {'AbsRel':>8} {'RMSE':>8} {'SILog':>8} {'GradErr':>9} {'Ordinal':>9}")
print('-'*78)
for name, ckpt in models.items():
    enc, dec = load_model(ckpt)
    all_m = []
    with torch.no_grad():
        for b in loader:
            pred = disp_to_depth(dec(enc(b['color'].to(DEVICE)))[('disp',0)])
            m    = compute_metrics(pred, b['depth_gt'].to(DEVICE))
            if m: all_m.append(m)
    avg = {k: round(np.mean([x[k] for x in all_m]),4) for k in all_m[0]}
    print(f"{name:<18} {avg['AbsRel']:>8} {avg['RMSE']:>8} {avg['SILog']:>8} {avg['GradErr']:>9} {avg['Ordinal']:>9}")
print('='*78)
