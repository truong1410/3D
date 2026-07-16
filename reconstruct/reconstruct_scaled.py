import sys, numpy as np, open3d as o3d, torch, cv2, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from losses import disp_to_depth

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE  = BASE + 'CV3Dv2/'
OUT_DIR = BASE + 'output/scaled/'
os.makedirs(OUT_DIR, exist_ok=True)

W_vol, H_vol = 512, 640
fx=349.0; fy=349.0; cx=237.5; cy=237.5
intrinsic = o3d.camera.PinholeCameraIntrinsic(W_vol, H_vol, fx, fy, cx, cy)
ray_map   = generate_ray_map(INTRINSICS)

SCALE_BASE = 1.479
SCALE_TNET = 1.486
SCALE_V2   = 1.486  # dùng TNet scale cho v2all

def load_model(ckpt_path):
    enc = ResnetEncoder(18, pretrained=False).to(DEVICE)
    dec = DepthDecoder(enc.num_ch_enc).to(DEVICE)
    ck  = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    enc.load_state_dict(ck['encoder']); dec.load_state_dict(ck['decoder'])
    enc.eval(); dec.eval()
    return enc, dec

def reconstruct_one(enc, dec, seq_dir, scale=1.0):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    ds    = C3VDDataset(seq_dir)
    poses = load_poses(seq_dir)
    for i in range(min(len(poses), len(ds)+1)):
        color_path = ds._color_path(i)
        if not os.path.exists(color_path): continue
        color_np = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        color_np = cv2.resize(color_np, (W_vol, H_vol))
        rgb_t = torch.from_numpy(
            color_np.astype(np.float32)/255.0
        ).permute(2,0,1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth_np = disp_to_depth(
                dec(enc(rgb_t))[('disp',0)]
            ).squeeze().cpu().numpy() * scale
        color_o3d = o3d.geometry.Image(color_np.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                   color_o3d, depth_o3d, depth_scale=1.0,
                   depth_trunc=95.0, convert_rgb_to_intensity=False)
        T = np.linalg.inv(poses[i])
        volume.integrate(rgbd, intrinsic, T)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def eval_mesh(pred_mesh, gt_obj):
    gt = o3d.io.read_triangle_mesh(gt_obj)
    if not gt.has_vertices(): return None, None
    pred_pcd = pred_mesh.sample_points_uniformly(10000)
    gt_pcd   = gt.sample_points_uniformly(10000)
    dists = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    dists = dists[~np.isnan(dists)]
    return round(float(dists.mean()),3), round(float(dists.std()),3)

test_seqs = [
    ('c2_sigmoid_t1',   'clean',  V2BASE+'c2_sigmoid_t1/coverage_mesh.obj'),
    ('c2_sigmoid_t2',   'clean',  V2BASE+'c2_sigmoid_t2/coverage_mesh.obj'),
    ('c2_sigmoidv3_t1', 'debris', V2BASE+'c2_sigmoidv3_t1/coverage_mesh.obj'),
    ('c2_sigmoidv3_t2', 'debris', V2BASE+'c2_sigmoidv3_t2/coverage_mesh.obj'),
]

models = {
    'baseline':   (BASE+'baseline.pth',      SCALE_BASE),
    'tnet_v1':    (BASE+'tnet_final.pth',    SCALE_TNET),
    'tnet_v2all': (BASE+'tnet_v2all.pth',    SCALE_V2),
}

print()
print('=== Reconstruction với scale correction ===')
print(f"{'Model':<14} {'Sequence':<22} {'Type':<8} {'Mean':>8} {'Std':>8}")
print('-'*65)

all_results = {}
for model_name, (ckpt, scale) in models.items():
    enc, dec = load_model(ckpt)
    means_clean, means_debris = [], []
    for seq_name, seq_type, gt_obj in test_seqs:
        mesh = reconstruct_one(enc, dec, V2BASE+seq_name, scale=scale)
        out  = os.path.join(OUT_DIR, f'{model_name}_{seq_name}.ply')
        o3d.io.write_triangle_mesh(out, mesh)
        m, s = eval_mesh(mesh, gt_obj)
        if m:
            print(f"{model_name:<14} {seq_name:<22} {seq_type:<8} {m:>8} {s:>8}")
            if seq_type=='clean': means_clean.append(m)
            else:                 means_debris.append(m)
    c_avg = round(np.mean(means_clean),3)  if means_clean  else None
    d_avg = round(np.mean(means_debris),3) if means_debris else None
    all_results[model_name] = (c_avg, d_avg)
    print(f"  → avg clean={c_avg}  debris={d_avg}\n")

print('='*65)
print('Summary (scale corrected):')
print(f"{'Model':<14} {'Clean avg':>12} {'Debris avg':>12}")
print('-'*40)
for name, (c, d) in all_results.items():
    print(f"{name:<14} {str(c):>12} {str(d):>12}")
print('='*65)
