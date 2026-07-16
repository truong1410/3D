import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch, numpy as np, json, os
from torch.utils.data import DataLoader, random_split
from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset
from losses import disp_to_depth

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
OUT     = BASE + 'metrics.json'

dataset = C3VDDataset(BASE + 'cecum_t1_a')
n_train = int(len(dataset) * 0.8)
n_val   = len(dataset) - n_train
_, val_ds = random_split(dataset, [n_train, n_val],
                         generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_ds, batch_size=4,
                        shuffle=False, num_workers=2)

def evaluate(ckpt_path, name):
    encoder = ResnetEncoder(18, pretrained=False).to(DEVICE)
    decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
    ckpt    = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    encoder.eval(); decoder.eval()

    abs_rels, rmses, d1s = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            rgb  = batch['color'].to(DEVICE)
            gt   = batch['depth_gt'].to(DEVICE)
            pred = disp_to_depth(decoder(encoder(rgb))[('disp', 0)])

            mask = ~torch.isnan(gt) & (gt > 1.0) & (gt < 99.0)
            if mask.sum() < 10:
                continue
            p = pred[mask]; g = gt[mask]
            thresh = torch.max(p/g, g/p)
            abs_rels.append(((p-g).abs()/g).mean().item())
            rmses.append(((p-g)**2).mean().sqrt().item())
            d1s.append((thresh < 1.25).float().mean().item())

    result = {
        'method':  name,
        'AbsRel':  round(np.mean(abs_rels), 4),
        'RMSE':    round(np.mean(rmses),    4),
        'd1':      round(np.mean(d1s),      4),
        'epoch':   ckpt['epoch'],
    }
    print(f"{name}: AbsRel={result['AbsRel']} | "
          f"RMSE={result['RMSE']} | d1={result['d1']}")
    return result

results = {}
results['Monodepth2']    = evaluate(BASE + 'baseline.pth',   'Monodepth2')
results['DepthNet+TNet'] = evaluate(BASE + 'tnet_final.pth', 'DepthNet+TNet')

# Lưu metrics 3D reconstruction
results['reconstruction'] = {
    'mean_distance': 7.3044,
    'std_distance':  4.3137,
    'median':        6.5267,
    'unit': 'mm'
}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {OUT}")
print(json.dumps(results, indent=2))
