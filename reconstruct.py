import sys
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

import torch, numpy as np, cv2, open3d as o3d, os
from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import INTRINSICS, generate_ray_map, load_poses, C3VDDataset
from losses import disp_to_depth

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
CKPT    = BASE + 'tnet_final.pth'
OUT_DIR = BASE + 'output/'
os.makedirs(OUT_DIR, exist_ok=True)

SEQS = ['cecum_t1_a', 'sigmoid_t1_a', 'sigmoid_t2_a', 'sigmoid_t3_a', 'sigmoid_t3_b']

# Load model
encoder = ResnetEncoder(18, pretrained=False).to(DEVICE)
decoder = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
ckpt    = torch.load(CKPT, weights_only=False, map_location=DEVICE)
encoder.load_state_dict(ckpt['encoder'])
decoder.load_state_dict(ckpt['decoder'])
encoder.eval(); decoder.eval()
print(f"Loaded model (AbsRel={ckpt['absrel']:.4f})")

ray_map   = generate_ray_map(INTRINSICS)
W_vol, H_vol = 512, 640
fx=349.0; fy=349.0; cx=237.5; cy=237.5
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    W_vol, H_vol, fx, fy, cx, cy)

all_meshes = []
for seq in SEQS:
    seq_dir = BASE + seq
    if not os.path.exists(seq_dir):
        print(f"Skip {seq} — not found")
        continue

    ds    = C3VDDataset(seq_dir)
    poses = load_poses(seq_dir)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    print(f"\nProcessing {seq} ({len(poses)} frames)...")
    for i in range(len(poses)):
        color_path = ds._color_path(i)
        if not os.path.exists(color_path):
            continue

        color_np = cv2.cvtColor(
            cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        color_np = cv2.resize(color_np, (W_vol, H_vol))

        rgb_t = torch.from_numpy(
            color_np.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth_np = disp_to_depth(
                decoder(encoder(rgb_t))[('disp', 0)]
            ).squeeze().cpu().numpy()

        color_o3d = o3d.geometry.Image(color_np.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                   color_o3d, depth_o3d,
                   depth_scale=1.0, depth_trunc=95.0,
                   convert_rgb_to_intensity=False)

        T = np.linalg.inv(poses[i].T)
        volume.integrate(rgbd, intrinsic, T)

        if i % 100 == 0:
            print(f"  Frame {i}/{len(poses)}")

    # Lưu mesh từng sequence
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = os.path.join(OUT_DIR, f'{seq}_mesh.ply')
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"  Saved {out} | V:{len(mesh.vertices)} F:{len(mesh.triangles)}")
    all_meshes.append(mesh)

# Merge tất cả mesh
print("\nMerging all meshes...")
combined = all_meshes[0]
for m in all_meshes[1:]:
    combined += m
out_combined = OUT_DIR + 'combined_mesh.ply'
o3d.io.write_triangle_mesh(out_combined, combined)
print(f"Combined: V={len(combined.vertices)} F={len(combined.triangles)}")
print(f"Saved: {out_combined}")
