import sys, numpy as np, open3d as o3d, torch, cv2, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
OUT_DIR = BASE + 'output/tnetpose/'
os.makedirs(OUT_DIR, exist_ok=True)

W_vol, H_vol = 512, 640
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    W_vol, H_vol, 349.0, 349.0, 237.5, 237.5)
ray_map = generate_ray_map(INTRINSICS)

def load_full_model(ckpt):
    enc  = ResnetEncoder(18, pretrained=False).to(DEVICE)
    dec  = DepthDecoder(enc.num_ch_enc).to(DEVICE)
    tnet = TNet().to(DEVICE)
    ck   = torch.load(ckpt, weights_only=False, map_location=DEVICE)
    enc.load_state_dict(ck['encoder'])
    dec.load_state_dict(ck['decoder'])
    tnet.load_state_dict(ck['tnet'])
    enc.eval(); dec.eval(); tnet.eval()
    return enc, dec, tnet

def accumulate_poses(tnet, ds, n_frames):
    """Tích lũy pose từ TNet thay vì dùng GT"""
    poses = [np.eye(4)]  # frame 0 tại origin
    T_accum = np.eye(4)
    with torch.no_grad():
        for i in range(min(n_frames-1, len(ds))):
            b  = ds[i]
            t  = b['color'].unsqueeze(0).to(DEVICE)
            t1 = b['color_next'].unsqueeze(0).to(DEVICE)
            vec   = tnet(t, t1)
            T_rel = pose_vec_to_mat(vec).squeeze().cpu().numpy()
            T_accum = T_accum @ np.linalg.inv(T_rel)
            poses.append(T_accum.copy())
    return poses

def reconstruct_with_poses(enc, dec, seq_dir, poses):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    ds = C3VDDataset(seq_dir)
    for i in range(min(len(poses), len(ds)+1)):
        cp = ds._color_path(i)
        if not os.path.exists(cp): continue
        cn = cv2.cvtColor(cv2.imread(cp), cv2.COLOR_BGR2RGB)
        cn = cv2.resize(cn, (W_vol, H_vol))
        with torch.no_grad():
            dn = disp_to_depth(dec(enc(
                torch.from_numpy(cn.astype(np.float32)/255.0)
                .permute(2,0,1).unsqueeze(0).to(DEVICE)))[('disp',0)]
            ).squeeze().cpu().numpy()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cn.astype(np.uint8)),
            o3d.geometry.Image(dn.astype(np.float32)),
            depth_scale=1.0, depth_trunc=95.0,
            convert_rgb_to_intensity=False)
        T = np.linalg.inv(poses[i])
        volume.integrate(rgbd, intrinsic, T)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def fscore(pred_mesh, gt_mesh, t=5.0):
    pp = pred_mesh.sample_points_uniformly(10000)
    gp = gt_mesh.sample_points_uniformly(10000)
    d1 = np.asarray(pp.compute_point_cloud_distance(gp))
    d2 = np.asarray(gp.compute_point_cloud_distance(pp))
    pr = (d1<t).mean(); rc = (d2<t).mean()
    dist = round(float(d1.mean()),3)
    f = round(2*pr*rc/(pr+rc) if pr+rc>0 else 0, 4)
    return f, dist

seq_dir = BASE + 'cecum_t1_a'
gt_mesh = o3d.io.read_triangle_mesh(BASE + 'cecum_t1_a/coverage_mesh.obj')

models = {
    'TNet+Smooth+GTpose':   BASE + 'tnet_smooth.pth',
    'TNet+Smooth+TNetpose': BASE + 'tnet_smooth.pth',
}

print()
print('='*65)
print('So sánh GT pose vs TNet pose')
print(f"{'Method':<30} {'Dist':>8} {'F@5mm':>8} {'F@10mm':>8}")
print('-'*65)

# GT pose
enc, dec, tnet_model = load_full_model(BASE + 'tnet_smooth.pth')
gt_poses = load_poses(seq_dir)
mesh = reconstruct_with_poses(enc, dec, seq_dir, gt_poses)
f5, d5   = fscore(mesh, gt_mesh, 5.0)
f10, d10 = fscore(mesh, gt_mesh, 10.0)
print(f"{'TNet+Smooth+GTpose':<30} {d5:>8} {f5:>8} {f10:>8}")
o3d.io.write_triangle_mesh(OUT_DIR + 'smooth_gtpose.ply', mesh)

# TNet predicted pose
ds = C3VDDataset(seq_dir)
tnet_poses = accumulate_poses(tnet_model, ds, len(gt_poses))
mesh2 = reconstruct_with_poses(enc, dec, seq_dir, tnet_poses)
f5b, d5b   = fscore(mesh2, gt_mesh, 5.0)
f10b, d10b = fscore(mesh2, gt_mesh, 10.0)
print(f"{'TNet+Smooth+TNetpose':<30} {d5b:>8} {f5b:>8} {f10b:>8}")
o3d.io.write_triangle_mesh(OUT_DIR + 'smooth_tnetpose.ply', mesh2)

print('='*65)
print()
if d5b < d5:
    print('TNet pose tốt hơn GT pose → không cần GT pose!')
else:
    print(f'GT pose vẫn tốt hơn ({d5} vs {d5b})')
    print('→ Cần cải thiện pose estimation accuracy')
