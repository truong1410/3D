import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
OUT_DIR = BASE + 'output/slam/'

W, H = 640, 512
fx=349.0; fy=349.0; cx=237.5; cy=237.5
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)

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

def estimate_relative_pose_orb(kps1, des1, depth1, kps2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None: return None, 0
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good    = [m for m in matches if m.distance < 50][:100]
    if len(good) < 8: return None, 0

    pts3d, pts2d = [], []
    for m in good:
        u1,v1 = int(kps1[m.queryIdx].pt[0]), int(kps1[m.queryIdx].pt[1])
        if 0<=u1<W and 0<=v1<H:
            d = depth1[v1,u1]
            if 1.0<d<95.0:
                pts3d.append([(u1-cx)*d/fx,(v1-cy)*d/fy,d])
                pts2d.append(kps2[m.trainIdx].pt)

    if len(pts3d) < 6: return None, 0
    pts3d = np.array(pts3d, dtype=np.float32)
    pts2d = np.array(pts2d, dtype=np.float32)
    ok,rvec,tvec,inliers = cv2.solvePnPRansac(
        pts3d,pts2d,K,None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2.0, confidence=0.99)
    if not ok or inliers is None or len(inliers)<6: return None, 0
    R,_ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T, len(inliers)

def reconstruct_sequence(enc, dec, tnet, seq_dir, T_start=None):
    """Reconstruct 1 sequence, bắt đầu từ T_start"""
    ds    = C3VDDataset(seq_dir)
    orb   = cv2.ORB_create(nfeatures=1000)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    if T_start is None: T_start = np.eye(4)
    T_accum  = T_start.copy()
    keyframes = []
    poses_out = [T_accum.copy()]

    with torch.no_grad():
        for i in range(len(ds)):
            b  = ds[i]
            t  = b['color'].unsqueeze(0).to(DEVICE)
            t1 = b['color_next'].unsqueeze(0).to(DEVICE)

            depth_np = disp_to_depth(
                dec(enc(t))[('disp',0)]
            ).squeeze().cpu().numpy()

            color_np = (b['color'].permute(1,2,0).numpy()*255
                        ).astype(np.uint8)
            gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
            kps, des = orb.detectAndCompute(gray, None)

            if i == 0:
                keyframes.append({
                    'pose':T_accum.copy(),'kps':kps,
                    'des':des,'depth':depth_np,'idx':0})
            else:
                kf = keyframes[-1]
                T_orb, n = estimate_relative_pose_orb(
                    kf['kps'],kf['des'],kf['depth'],kps,des)
                if T_orb is not None and n >= 8:
                    T_accum = kf['pose'] @ T_orb
                else:
                    vec    = tnet(t,t1)
                    T_tnet = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                    T_accum = poses_out[-1] @ np.linalg.inv(T_tnet)

            poses_out.append(T_accum.copy())
            if i % 5 == 0:
                keyframes.append({
                    'pose':T_accum.copy(),'kps':kps,
                    'des':des,'depth':depth_np,'idx':i})

            # Integrate
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0, depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic_o3d, np.linalg.inv(T_accum))

    return volume, T_accum, poses_out

# Dùng tất cả sigmoid v1 sequences — mỗi seq bắt đầu từ vị trí cuối seq trước
enc, dec, tnet = load_full_model(BASE + 'tnet_sigmoid_v1.pth')

seqs = [
    BASE + 'sigmoid_t1_a',
    BASE + 'sigmoid_t2_a',
    BASE + 'sigmoid_t3_a',
    BASE + 'sigmoid_t3_b',
]

# Dùng TSDF volume chung cho tất cả sequences
master_volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=2.0, sdf_trunc=10.0,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

T_current = np.eye(4)
total_frames = 0

for seq_dir in seqs:
    if not os.path.exists(seq_dir):
        print(f'Skip {seq_dir}')
        continue
    print(f'Processing {os.path.basename(seq_dir)}...')
    vol, T_end, _ = reconstruct_sequence(
        enc, dec, tnet, seq_dir, T_start=T_current)

    # Merge volume vào master
    # Lấy mesh từ vol và integrate lại vào master
    mesh_tmp = vol.extract_triangle_mesh()
    pcd_tmp  = mesh_tmp.sample_points_uniformly(
        min(50000, len(mesh_tmp.vertices)*3))
    print(f'  Frames done | Mesh: V={len(mesh_tmp.vertices)}')
    total_frames += 1

    # Dùng volume riêng, lưu mesh rồi merge
    out_tmp = OUT_DIR + f'{os.path.basename(seq_dir)}_slam.ply'
    mesh_tmp.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_tmp, mesh_tmp)

    T_current = T_end

print(f'\nSaved individual sequence meshes to {OUT_DIR}')
print('Files:')
for seq_dir in seqs:
    name = os.path.basename(seq_dir)
    path = OUT_DIR + f'{name}_slam.ply'
    if os.path.exists(path):
        m = o3d.io.read_triangle_mesh(path)
        print(f'  {name}_slam.ply: V={len(m.vertices)} F={len(m.triangles)}')
