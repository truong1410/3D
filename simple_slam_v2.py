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
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 640, 512
fx=349.0; fy=349.0; cx=237.5; cy=237.5
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

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

def depth_to_points3d(depth, kps):
    """Convert keypoint + depth → 3D points"""
    pts3d = []
    valid = []
    for i, kp in enumerate(kps):
        u, v = int(kp.pt[0]), int(kp.pt[1])
        if 0<=u<W and 0<=v<H:
            d = depth[v, u]
            if 1.0 < d < 95.0:
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                pts3d.append([x, y, d])
                valid.append(i)
    return np.array(pts3d, dtype=np.float32), valid

def estimate_relative_pose(kps1, des1, depth1, kps2, des2):
    """
    Estimate T_{1→2} từ ORB matching + PnP
    3D points từ frame 1, 2D points từ frame 2
    """
    orb = cv2.ORB_create(nfeatures=1000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des1 is None or des2 is None: return None, 0

    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good    = [m for m in matches if m.distance < 50][:100]

    if len(good) < 8: return None, len(good)

    # 3D từ frame 1, 2D từ frame 2
    pts3d, pts2d = [], []
    for m in good:
        u1 = int(kps1[m.queryIdx].pt[0])
        v1 = int(kps1[m.queryIdx].pt[1])
        if 0<=u1<W and 0<=v1<H:
            d = depth1[v1, u1]
            if 1.0 < d < 95.0:
                x = (u1-cx)*d/fx; y = (v1-cy)*d/fy
                pts3d.append([x, y, d])
                pts2d.append(kps2[m.trainIdx].pt)

    if len(pts3d) < 6: return None, len(good)

    pts3d = np.array(pts3d, dtype=np.float32)
    pts2d = np.array(pts2d, dtype=np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2.0, confidence=0.99)

    if not ok or inliers is None or len(inliers) < 6:
        return None, len(good)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T, len(inliers)

def reconstruct_slam(enc, dec, tnet, seq_dir, name):
    ds       = C3VDDataset(seq_dir)
    gt_poses = load_poses(seq_dir)
    orb      = cv2.ORB_create(nfeatures=1000)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # Lưu keyframes cho loop closure
    keyframes  = []  # list of {pose, kps, des, depth, idx}
    T_accum    = np.eye(4)
    poses_list = [np.eye(4)]
    gt_errors  = []
    orb_count  = 0
    tnet_count = 0
    loop_count = 0

    print(f'\n[{name}] Processing {len(ds)} frames...')

    with torch.no_grad():
        for i in range(len(ds)):
            b  = ds[i]
            t  = b['color'].unsqueeze(0).to(DEVICE)
            t1 = b['color_next'].unsqueeze(0).to(DEVICE)

            # Predict depth
            depth_np = disp_to_depth(
                dec(enc(t))[('disp',0)]
            ).squeeze().cpu().numpy()

            # RGB numpy
            color_np = (b['color'].permute(1,2,0).numpy()*255
                        ).astype(np.uint8)
            gray_curr = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
            kps_curr, des_curr = orb.detectAndCompute(gray_curr, None)

            if i == 0:
                keyframes.append({
                    'pose': np.eye(4), 'kps': kps_curr,
                    'des': des_curr, 'depth': depth_np, 'idx': 0})
                # Integrate frame 0
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np.astype(np.float32)),
                    depth_scale=1.0, depth_trunc=95.0,
                    convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic_o3d, np.eye(4))
                continue

            # ── Tracking: ORB PnP trước, fallback TNet ──────────
            kf_prev = keyframes[-1]
            T_rel_orb, n_inliers = estimate_relative_pose(
                kf_prev['kps'], kf_prev['des'], kf_prev['depth'],
                kps_curr, des_curr)

            if T_rel_orb is not None and n_inliers >= 8:
                # ORB PnP thành công
                T_accum = kf_prev['pose'] @ T_rel_orb
                orb_count += 1
            else:
                # Fallback TNet
                vec    = tnet(t, t1)
                T_tnet = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                T_accum = poses_list[-1] @ np.linalg.inv(T_tnet)
                tnet_count += 1

            # ── Loop closure mỗi 20 frames ───────────────────────
            if i % 20 == 0 and des_curr is not None and len(keyframes) > 30:
                bf_lc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                best_n, best_kf = 0, -1
                for kf_idx, kf in enumerate(keyframes[:-30]):
                    if kf['des'] is None: continue
                    m = bf_lc.match(des_curr, kf['des'])
                    n = sum(1 for x in m if x.distance < 40)
                    if n > best_n:
                        best_n, best_kf = n, kf_idx

                if best_n >= 40:
                    kf_loop = keyframes[best_kf]
                    T_lc, n_lc = estimate_relative_pose(
                        kf_loop['kps'], kf_loop['des'], kf_loop['depth'],
                        kps_curr, des_curr)
                    if T_lc is not None and n_lc >= 10:
                        # Compute corrected pose
                        T_corrected = kf_loop['pose'] @ T_lc
                        # Linear correction từ loop frame → current
                        n_corr = len(poses_list) - best_kf
                        for j in range(n_corr):
                            alpha = j / max(n_corr, 1)
                            delta = alpha * (T_corrected[:3,3] -
                                           T_accum[:3,3])
                            poses_list[best_kf+j][:3,3] += delta
                        T_accum = T_corrected
                        loop_count += 1
                        print(f'  Loop @frame {i}: '
                              f'kf={best_kf} matches={best_n}')

            poses_list.append(T_accum.copy())

            # Thêm keyframe mỗi 5 frames
            if i % 5 == 0:
                keyframes.append({
                    'pose': T_accum.copy(), 'kps': kps_curr,
                    'des': des_curr, 'depth': depth_np, 'idx': i})

            # TSDF integration
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0, depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic_o3d,
                             np.linalg.inv(T_accum))

            # Track vs GT
            err = np.linalg.norm(T_accum[:3,3] - gt_poses[i+1][:3,3])
            gt_errors.append(err)

            if (i+1) % 100 == 0:
                print(f'  Frame {i+1}/{len(ds)} | '
                      f'drift={err:.1f}mm | '
                      f'orb={orb_count} tnet={tnet_count} '
                      f'loops={loop_count}')

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = OUT_DIR + f'{name}.ply'
    o3d.io.write_triangle_mesh(out, mesh)

    print(f'\n  Summary:')
    print(f'  ORB used:  {orb_count}/{len(ds)} frames')
    print(f'  TNet used: {tnet_count}/{len(ds)} frames')
    print(f'  Loops:     {loop_count}')
    print(f'  Drift avg: {np.mean(gt_errors):.2f}mm')
    print(f'  Drift final: {gt_errors[-1]:.2f}mm')
    print(f'  Mesh: V={len(mesh.vertices)} F={len(mesh.triangles)}')
    return mesh, gt_errors

def fscore(pred_mesh, gt_mesh, t=5.0):
    pp = pred_mesh.sample_points_uniformly(10000)
    gp = gt_mesh.sample_points_uniformly(10000)
    d1 = np.asarray(pp.compute_point_cloud_distance(gp))
    d2 = np.asarray(gp.compute_point_cloud_distance(pp))
    pr = (d1<t).mean(); rc = (d2<t).mean()
    return round(2*pr*rc/(pr+rc) if pr+rc>0 else 0,4), round(float(d1.mean()),3)

# Load model
enc, dec, tnet = load_full_model(BASE + 'tnet_sigmoid_v1.pth')
seq_dir = BASE + 'sigmoid_t1_a'
gt_obj  = BASE + 'sigmoid_t1_a/coverage_mesh.obj'

mesh, errors = reconstruct_slam(enc, dec, tnet, seq_dir, 'sigmoid_slamv2')

print()
print('='*55)
if os.path.exists(gt_obj):
    gt_mesh = o3d.io.read_triangle_mesh(gt_obj)
    f5,  d5  = fscore(mesh, gt_mesh, 5.0)
    f10, d10 = fscore(mesh, gt_mesh, 10.0)
    print(f"F@5mm={f5}  F@10mm={f10}  MeanDist={d5}mm")
else:
    print(f"Surface area: {mesh.get_surface_area():.1f} mm²")
print('='*55)
