import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE  = BASE + 'CV3Dv2/'
OUT_DIR = BASE + 'output/slam_v2/'
os.makedirs(OUT_DIR, exist_ok=True)

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

def estimate_pose_orb(kps1, des1, depth1, kps2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None: return None, 0
    matches = sorted(bf.match(des1,des2), key=lambda x: x.distance)
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
    pts3d = np.array(pts3d,dtype=np.float32)
    pts2d = np.array(pts2d,dtype=np.float32)
    ok,rvec,tvec,inliers = cv2.solvePnPRansac(
        pts3d,pts2d,K,None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2.0,confidence=0.99)
    if not ok or inliers is None or len(inliers)<6: return None, 0
    R,_ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T, len(inliers)

def reconstruct_one_seq(enc, dec, tnet, seq_dir, gt_poses_available=True):
    """Reconstruct 1 sequence với SLAM pose"""
    ds   = C3VDDataset(seq_dir)
    orb  = cv2.ORB_create(nfeatures=1000)
    gt_poses = load_poses(seq_dir)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_accum   = np.eye(4)
    keyframes = []
    poses_out = [np.eye(4)]
    gt_errors = []
    orb_ok = 0; tnet_ok = 0; loop_ok = 0

    bf_lc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
                    'pose':np.eye(4),'kps':kps,
                    'des':des,'depth':depth_np,'idx':0})
            else:
                # ORB PnP
                kf = keyframes[-1]
                T_orb, n = estimate_pose_orb(
                    kf['kps'],kf['des'],kf['depth'],kps,des)
                if T_orb is not None and n >= 8:
                    T_accum = kf['pose'] @ T_orb
                    orb_ok += 1
                else:
                    vec    = tnet(t,t1)
                    T_tnet = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                    T_accum = poses_out[-1] @ np.linalg.inv(T_tnet)
                    tnet_ok += 1

                # Loop closure mỗi 20 frames
                if i % 20 == 0 and des is not None and len(keyframes)>30:
                    best_n, best_kf_idx = 0, -1
                    for ki, kf_lc in enumerate(keyframes[:-30]):
                        if kf_lc['des'] is None: continue
                        m = bf_lc.match(des, kf_lc['des'])
                        n_m = sum(1 for x in m if x.distance < 40)
                        if n_m > best_n:
                            best_n, best_kf_idx = n_m, ki
                    if best_n >= 40 and best_kf_idx >= 0:
                        kf_lc = keyframes[best_kf_idx]
                        T_lc, n_lc = estimate_pose_orb(
                            kf_lc['kps'],kf_lc['des'],kf_lc['depth'],
                            kps,des)
                        if T_lc is not None and n_lc >= 10:
                            T_corrected = kf_lc['pose'] @ T_lc
                            n_corr = len(poses_out) - best_kf_idx
                            for j in range(n_corr):
                                alpha = j / max(n_corr,1)
                                delta = alpha*(T_corrected[:3,3]-
                                              T_accum[:3,3])
                                poses_out[best_kf_idx+j][:3,3] += delta
                            T_accum = T_corrected
                            loop_ok += 1

            poses_out.append(T_accum.copy())
            if i % 5 == 0:
                keyframes.append({
                    'pose':T_accum.copy(),'kps':kps,
                    'des':des,'depth':depth_np,'idx':i})

            # TSDF
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0, depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd,intrinsic_o3d,np.linalg.inv(T_accum))

            # GT error
            err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
            gt_errors.append(err)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh, gt_errors, orb_ok, tnet_ok, loop_ok

def align_and_eval(pred_mesh, gt_mesh):
    pred_pcd = pred_mesh.sample_points_uniformly(10000)
    gt_pcd   = gt_mesh.sample_points_uniformly(10000)
    pred_pts = np.asarray(pred_pcd.points)
    gt_pts   = np.asarray(gt_pcd.points)
    pred_pcd.translate(gt_pts.mean(0)-pred_pts.mean(0))
    pred_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    gt_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        pred_pcd,gt_pcd,30.0,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100))
    pred_pcd.transform(result.transformation)
    d1 = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    d2 = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    r = {'MeanDist': round(float(d1.mean()),3)}
    for t in [5.0,10.0]:
        pr=(d1<t).mean(); rc=(d2<t).mean()
        r[f'F@{t:.0f}mm']=round(2*pr*rc/(pr+rc) if pr+rc>0 else 0,4)
    return r

# Load model v2
enc, dec, tnet = load_full_model(BASE + 'tnet_v2all.pth')

# Test sequences — clean + debris
test_seqs = [
    ('c2_sigmoid_t1',   'clean'),
    ('c2_sigmoid_t2',   'clean'),
    ('c2_sigmoidv3_t1', 'debris'),
    ('c2_sigmoidv3_t2', 'debris'),
]

print()
print('='*75)
print(f"{'Sequence':<22} {'Type':<8} {'Drift':>8} {'Loops':>6} "
      f"{'MeanDist':>10} {'F@5mm':>8} {'F@10mm':>8}")
print('-'*75)

clean_f5=[];  debris_f5=[]
for seq_name, seq_type in test_seqs:
    seq_dir = V2BASE + seq_name
    gt_obj  = V2BASE + seq_name + '/coverage_mesh.obj'

    mesh, errs, orb_n, tnet_n, loop_n = reconstruct_one_seq(
        enc, dec, tnet, seq_dir)

    out = OUT_DIR + f'{seq_name}_slam.ply'
    o3d.io.write_triangle_mesh(out, mesh)

    r = align_and_eval(mesh, o3d.io.read_triangle_mesh(gt_obj))
    drift = round(errs[-1],1) if errs else 0

    print(f"{seq_name:<22} {seq_type:<8} {drift:>8} {loop_n:>6} "
          f"{r['MeanDist']:>10} {r['F@5mm']:>8} {r['F@10mm']:>8}")

    if seq_type=='clean':  clean_f5.append(r['F@5mm'])
    else:                  debris_f5.append(r['F@5mm'])

print('-'*75)
print(f"{'avg clean':<38} {round(np.mean(clean_f5),4):>26}")
print(f"{'avg debris':<38} {round(np.mean(debris_f5),4):>26}")
print('='*75)
