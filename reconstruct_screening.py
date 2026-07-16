import sys, numpy as np, open3d as o3d, torch, cv2, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import INTRINSICS, generate_ray_map
from losses import disp_to_depth
import tifffile

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE     = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
SCREEN   = BASE + 'Screening/'
OUT_DIR  = BASE + 'output/screening/'
os.makedirs(OUT_DIR, exist_ok=True)

W_vol, H_vol = 512, 640
fx=349.0; fy=349.0; cx=237.5; cy=237.5
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    W_vol, H_vol, fx, fy, cx, cy)
ray_map = generate_ray_map(INTRINSICS)

def load_model(ckpt):
    enc = ResnetEncoder(18, pretrained=False).to(DEVICE)
    dec = DepthDecoder(enc.num_ch_enc).to(DEVICE)
    ck  = torch.load(ckpt, weights_only=False, map_location=DEVICE)
    enc.load_state_dict(ck['encoder'])
    dec.load_state_dict(ck['decoder'])
    enc.eval(); dec.eval()
    return enc, dec

def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = np.fromstring(line, dtype=float, sep=',')
            if p.size == 16:
                poses.append(p.reshape(4,4).T)  # row-major → transpose
    return poses

def reconstruct_screening(enc, dec, seq_dir, pose_file,
                           name, step=5, max_frames=1000):
    """
    step=5: lấy 1 frame mỗi 5 frames để giảm thời gian
    max_frames: giới hạn số frames xử lý
    """
    poses  = load_poses(pose_file)
    rgb_dir = os.path.join(seq_dir, 'rgb')
    frames = sorted(os.listdir(rgb_dir),
                    key=lambda x: int(x.replace('.png','')))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    count = 0
    for i, fname in enumerate(frames):
        if i >= len(poses): break
        if i % step != 0: continue  # lấy mỗi step frames
        if count >= max_frames: break

        color_path = os.path.join(rgb_dir, fname)
        cn = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
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
        count += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = os.path.join(OUT_DIR, f'{name}.ply')
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"  {name}: {count} frames → V:{len(mesh.vertices)} F:{len(mesh.triangles)}")
    return mesh

# Dùng best model — TNet+Smooth
enc, dec = load_model(BASE + 'tnet_smooth.pth')

seqs = [
    ('c0_full_t1_v1', 'pose_c0_full_t1_v1.txt'),
    ('c0_full_t2_v1', 'pose_c0_full_t2_v1.txt'),
    ('c0_full_t3_v1', 'pose_c0_full_t3_v1.txt'),
    ('c0_full_t4_v1', 'pose_c0_full_t4_v1.txt'),
]

print(f"Model: TNet+Smooth | Device: {DEVICE}")
print(f"Step: every 5 frames | Max: 1000 frames per sequence")
print()

for seq_name, pose_file in seqs:
    seq_dir   = SCREEN + seq_name
    pose_path = os.path.join(seq_dir, pose_file)
    if not os.path.exists(pose_path):
        print(f"Skip {seq_name} — pose file not found")
        continue
    print(f"Processing {seq_name}...")
    reconstruct_screening(enc, dec, seq_dir, pose_path, seq_name)

print()
print(f"Saved to: {OUT_DIR}")
