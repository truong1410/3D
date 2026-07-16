import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch, numpy as np, cv2, open3d as o3d, os
from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from losses import disp_to_depth

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE  = BASE + 'CV3Dv2/'
OUT_DIR = BASE + 'output/compare/'
os.makedirs(OUT_DIR, exist_ok=True)

W_vol, H_vol = 512, 640
fx=349.0; fy=349.0; cx=237.5; cy=237.5
intrinsic = o3d.camera.PinholeCameraIntrinsic(W_vol, H_vol, fx, fy, cx, cy)
ray_map   = generate_ray_map(INTRINSICS)

def load_model(ckpt_path):
    enc = ResnetEncoder(18, pretrained=False).to(DEVICE)
    dec = DepthDecoder(enc.num_ch_enc).to(DEVICE)
    ck  = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    enc.load_state_dict(ck['encoder'])
    dec.load_state_dict(ck['decoder'])
    enc.eval(); dec.eval()
    return enc, dec

def reconstruct(enc, dec, seq_dirs, name):
    """Reconstruct mesh từ list sequences"""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    total = 0
    for seq_dir in seq_dirs:
        if not os.path.exists(seq_dir):
            print(f"  Skip {seq_dir}")
            continue
        ds    = C3VDDataset(seq_dir)
        poses = load_poses(seq_dir)
        print(f"  {os.path.basename(seq_dir)}: {len(poses)} frames")

        for i in range(min(len(poses), len(ds)+1)):
            color_path = ds._color_path(i)
            if not os.path.exists(color_path):
                continue
            color_np = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
            color_np = cv2.resize(color_np, (W_vol, H_vol))

            rgb_t = torch.from_numpy(
                color_np.astype(np.float32)/255.0
            ).permute(2,0,1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth_np = disp_to_depth(
                    dec(enc(rgb_t))[('disp',0)]
                ).squeeze().cpu().numpy()

            color_o3d = o3d.geometry.Image(color_np.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                       color_o3d, depth_o3d,
                       depth_scale=1.0, depth_trunc=95.0,
                       convert_rgb_to_intensity=False)

            T = np.linalg.inv(poses[i])
            volume.integrate(rgbd, intrinsic, T)
            total += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = os.path.join(OUT_DIR, f'{name}.ply')
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"  Saved {out} | V:{len(mesh.vertices)} F:{len(mesh.triangles)}")
    return out

def evaluate_mesh(pred_ply, gt_obj):
    """Cloud-mesh distance"""
    if not os.path.exists(gt_obj):
        return None, None
    pred = o3d.io.read_triangle_mesh(pred_ply)
    gt   = o3d.io.read_triangle_mesh(gt_obj)
    pred_pcd = pred.sample_points_uniformly(10000)
    gt_pcd   = gt.sample_points_uniformly(10000)
    dists = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    dists = dists[~np.isnan(dists)]
    return round(float(dists.mean()), 4), round(float(dists.std()), 4)

# ── Sequences ────────────────────────────────────────────────
v2_clean  = [V2BASE + s for s in ['c2_sigmoid_t1','c2_sigmoid_t2',
                                   'c2_sigmoid_t3','c2_sigmoid_t4']]
v2_debris = [V2BASE + s for s in ['c2_sigmoidv3_t1','c2_sigmoidv3_t2',
                                   'c2_sigmoidv3_t3','c2_sigmoidv3_t4']]

# GT meshes — dùng t1 làm representative
gt_clean  = V2BASE + 'c2_sigmoid_t1/coverage_mesh.obj'
gt_debris = V2BASE + 'c2_sigmoidv3_t1/coverage_mesh.obj'

# ── Models ───────────────────────────────────────────────────
models = {
    'baseline':    BASE + 'baseline.pth',
    'tnet_v1':     BASE + 'tnet_final.pth',
    'tnet_v2all':  BASE + 'tnet_v2all.pth',
}

results = {}
for model_name, ckpt_path in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    enc, dec = load_model(ckpt_path)
    results[model_name] = {}

    # Clean only
    print("  → Reconstructing v2 clean...")
    ply = reconstruct(enc, dec, v2_clean, f'{model_name}_clean')
    mean, std = evaluate_mesh(ply, gt_clean)
    results[model_name]['clean'] = (mean, std)

    # Debris only
    print("  → Reconstructing v2 debris...")
    ply = reconstruct(enc, dec, v2_debris, f'{model_name}_debris')
    mean, std = evaluate_mesh(ply, gt_debris)
    results[model_name]['debris'] = (mean, std)

    # Clean + debris combined
    print("  → Reconstructing v2 combined...")
    ply = reconstruct(enc, dec, v2_clean + v2_debris, f'{model_name}_combined')
    mean, std = evaluate_mesh(ply, gt_clean)
    results[model_name]['combined'] = (mean, std)

# ── Print results ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("3D Reconstruction Results (Mean cloud-mesh distance, mm)")
print(f"{'Model':<20} {'Clean':>14} {'Debris':>14} {'Combined':>14}")
print('-'*70)
for name, r in results.items():
    c = f"{r['clean'][0]} ± {r['clean'][1]}"   if r['clean'][0]    else "N/A"
    d = f"{r['debris'][0]} ± {r['debris'][1]}" if r['debris'][0]   else "N/A"
    b = f"{r['combined'][0]} ± {r['combined'][1]}" if r['combined'][0] else "N/A"
    print(f"{name:<20} {c:>14} {d:>14} {b:>14}")
print('='*70)
