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
# Pinhole intrinsics cho feature matching
fx=349.0; fy=349.0; cx=237.5; cy=237.5
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
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

class SimpleSlam:
    """
    Lightweight SLAM:
    1. TNet predict relative pose (tracking)
    2. ORB feature matching để verify/correct pose
    3. Loop closure detection bằng feature similarity
    4. Pose graph correction khi loop detected
    """
    def __init__(self, K):
        self.K       = K
        self.orb     = cv2.ORB_create(nfeatures=1000)
        self.bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.keyframes      = []   # (pose, descriptors, keypoints, depth)
        self.poses          = [np.eye(4)]
        self.loop_threshold = 30   # min matches để detect loop
        self.loop_corrections = 0

    def extract_features(self, gray):
        kps, des = self.orb.detectAndCompute(gray, None)
        return kps, des

    def estimate_pose_orb(self, kps1, des1, depth1, kps2, des2, T_init):
        """
        Estimate pose từ ORB feature matching + PnP
        Fallback về T_init nếu không đủ matches
        """
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return T_init, 0

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good    = matches[:50]

        if len(good) < 8:
            return T_init, len(good)

        # 3D points từ frame 1 (dùng depth)
        pts3d = []
        pts2d = []
        for m in good:
            u1, v1 = kps1[m.queryIdx].pt
            u2, v2 = kps2[m.trainIdx].pt
            u1, v1 = int(u1), int(v1)
            if 0<=u1<W and 0<=v1<H:
                d = depth1[v1, u1]
                if d > 1.0 and d < 95.0:
                    x = (u1 - cx) * d / fx
                    y = (v1 - cy) * d / fy
                    pts3d.append([x, y, d])
                    pts2d.append(kps2[m.trainIdx].pt)

        if len(pts3d) < 6:
            return T_init, len(good)

        pts3d = np.array(pts3d, dtype=np.float32)
        pts2d = np.array(pts2d, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, self.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.99)

        if not success or inliers is None or len(inliers) < 6:
            return T_init, len(good)

        R, _ = cv2.Rodrigues(rvec)
        T    = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = tvec.flatten()
        return T, len(inliers)

    def detect_loop(self, des_curr, curr_idx, min_gap=30):
        """
        Detect loop closure bằng ORB descriptor matching
        Chỉ check keyframes cách hiện tại ít nhất min_gap frames
        """
        if len(self.keyframes) < min_gap:
            return -1, 0

        best_matches = 0
        best_kf_idx  = -1

        for kf_idx, kf in enumerate(self.keyframes[:-min_gap]):
            if kf['des'] is None or des_curr is None: continue
            matches = self.bf.match(des_curr, kf['des'])
            n_good  = sum(1 for m in matches if m.distance < 50)
            if n_good > best_matches:
                best_matches = n_good
                best_kf_idx  = kf_idx

        if best_matches >= self.loop_threshold:
            return best_kf_idx, best_matches
        return -1, best_matches

    def correct_drift(self, loop_kf_idx, T_loop):
        """
        Linear interpolation pose correction
        Phân phối correction từ loop frame về hiện tại
        """
        n_correct = len(self.poses) - loop_kf_idx
        if n_correct <= 0: return

        T_curr    = self.poses[-1]
        T_kf      = self.keyframes[loop_kf_idx]['pose']
        T_err     = np.linalg.inv(T_kf) @ T_loop @ T_curr

        for i in range(n_correct):
            alpha = i / n_correct
            # Linear interpolation của correction
            t_corr = alpha * T_err[:3,3]
            self.poses[loop_kf_idx + i][:3,3] += t_corr

        self.loop_corrections += 1

    def process_frame(self, color_np, depth_np, T_tnet, frame_idx):
        """Process 1 frame: tracking + loop closure"""
        gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
        kps, des = self.extract_features(gray)

        if len(self.poses) == 1:
            # Frame đầu tiên
            self.keyframes.append({
                'pose': np.eye(4), 'kps': kps,
                'des': des, 'depth': depth_np, 'idx': 0})
            return np.eye(4)

        T_prev = self.poses[-1]

        # Estimate pose từ ORB + PnP
        kf_last = self.keyframes[-1]
        T_orb, n_inliers = self.estimate_pose_orb(
            kf_last['kps'], kf_last['des'], kf_last['depth'],
            kps, des, T_tnet)

        # Chọn pose: ORB nếu đủ inliers, không thì dùng TNet
        if n_inliers >= 10:
            T_curr = T_prev @ T_orb
        else:
            T_curr = T_prev @ np.linalg.inv(T_tnet)

        # Loop closure detection mỗi 10 frames
        if frame_idx % 10 == 0 and des is not None:
            loop_idx, n_loop = self.detect_loop(des, frame_idx)
            if loop_idx >= 0:
                print(f"  Loop detected at frame {frame_idx} "
                      f"→ keyframe {loop_idx} ({n_loop} matches)")
                self.correct_drift(loop_idx, T_curr)
                T_curr = self.poses[-1]  # dùng corrected pose

        self.poses.append(T_curr)

        # Thêm vào keyframes mỗi 5 frames
        if frame_idx % 5 == 0:
            self.keyframes.append({
                'pose': T_curr, 'kps': kps,
                'des': des, 'depth': depth_np, 'idx': frame_idx})

        return T_curr

def reconstruct_slam(enc, dec, tnet, seq_dir, name):
    ds    = C3VDDataset(seq_dir)
    gt_poses = load_poses(seq_dir)

    slam   = SimpleSlam(K)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    gt_errors = []
    print(f'\nProcessing {name} ({len(ds)} frames)...')

    with torch.no_grad():
        for i in range(len(ds)):
            b  = ds[i]
            t  = b['color'].unsqueeze(0).to(DEVICE)
            t1 = b['color_next'].unsqueeze(0).to(DEVICE)

            # Predict depth
            depth_np = disp_to_depth(
                dec(enc(t))[('disp',0)]
            ).squeeze().cpu().numpy()

            # TNet relative pose
            vec   = tnet(t, t1)
            T_rel = pose_vec_to_mat(vec).squeeze().cpu().numpy()

            # SLAM process frame
            color_np = (b['color'].permute(1,2,0).numpy()*255).astype(np.uint8)
            T_curr   = slam.process_frame(color_np, depth_np, T_rel, i)

            # TSDF integration
            color_o3d = o3d.geometry.Image(color_np)
            depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0, depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic_o3d, np.linalg.inv(T_curr))

            # Track drift vs GT
            err = np.linalg.norm(T_curr[:3,3] - gt_poses[i+1][:3,3])
            gt_errors.append(err)

            if (i+1) % 100 == 0:
                print(f'  Frame {i+1}/{len(ds)} | '
                      f'drift={err:.1f}mm | '
                      f'loops={slam.loop_corrections}')

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = OUT_DIR + f'{name}.ply'
    o3d.io.write_triangle_mesh(out, mesh)

    print(f'  Loop corrections: {slam.loop_corrections}')
    print(f'  Final drift: {gt_errors[-1]:.2f}mm')
    print(f'  Avg drift:   {np.mean(gt_errors):.2f}mm')
    print(f'  Mesh: V={len(mesh.vertices)} F={len(mesh.triangles)}')
    return mesh, gt_errors

def fscore(pred_mesh, gt_mesh, t=5.0):
    pp = pred_mesh.sample_points_uniformly(10000)
    gp = gt_mesh.sample_points_uniformly(10000)
    d1 = np.asarray(pp.compute_point_cloud_distance(gp))
    d2 = np.asarray(gp.compute_point_cloud_distance(pp))
    pr = (d1<t).mean(); rc = (d2<t).mean()
    f  = round(2*pr*rc/(pr+rc) if pr+rc>0 else 0, 4)
    return f, round(float(d1.mean()),3)

# Load model train trên sigmoid v1
enc, dec, tnet = load_full_model(BASE + 'tnet_sigmoid_v1.pth')

# Test trên sigmoid_t1_a (có GT mesh không?)
seq_dir = BASE + 'sigmoid_t1_a'
gt_obj  = BASE + 'sigmoid_t1_a/coverage_mesh.obj'

mesh, errors = reconstruct_slam(enc, dec, tnet, seq_dir, 'sigmoid_slam')

print()
print('='*55)
if os.path.exists(gt_obj):
    gt_mesh = o3d.io.read_triangle_mesh(gt_obj)
    f5,  d5  = fscore(mesh, gt_mesh, 5.0)
    f10, d10 = fscore(mesh, gt_mesh, 10.0)
    print(f"F@5mm={f5}  F@10mm={f10}  MeanDist={d5}mm")
else:
    print("No GT mesh — qualitative only")
    print(f"Surface area: {mesh.get_surface_area():.1f} mm²")
print('='*55)
