import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE  = BASE + 'CV3Dv2/'
OUT     = BASE + 'output/slam_debris/'

W, H = 640, 512
fx=349.0; fy=349.0; cx=237.5; cy=237.5
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)

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
    if des1 is None or des2 is None: return None, 0, 0
    matches = sorted(bf.match(des1,des2), key=lambda x:x.distance)
    good    = [m for m in matches if m.distance<50][:100]
    if len(good)<6: return None, 0, len(good)
    pts3d,pts2d=[],[]
    for m in good:
        u1,v1=int(kps1[m.queryIdx].pt[0]),int(kps1[m.queryIdx].pt[1])
        if 0<=u1<W and 0<=v1<H:
            d=depth1[v1,u1]
            if 1.0<d<95.0:
                pts3d.append([(u1-cx)*d/fx,(v1-cy)*d/fy,d])
                pts2d.append(kps2[m.trainIdx].pt)
    if len(pts3d)<5: return None,0,len(good)
    pts3d=np.array(pts3d,dtype=np.float32)
    pts2d=np.array(pts2d,dtype=np.float32)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(
        pts3d,pts2d,K,None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,confidence=0.99)
    if not ok or inliers is None or len(inliers)<4: return None,0,len(good)
    R,_=cv2.Rodrigues(rvec)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T,len(inliers),len(good)

def slam_robust(enc, dec, tnet, seq_dir,
                anchor_interval=50,
                max_frame_dist=30.0,  # max allowed translation per frame (mm)
                name='slam_robust'):
    """
    Robust SLAM với:
    1. Anchor GT pose mỗi anchor_interval frames
    2. Outlier rejection: skip frames với pose jump > max_frame_dist mm
    3. Confidence score: chỉ integrate frames có tracking tốt
    """
    ds       = C3VDDataset(seq_dir)
    gt_poses = load_poses(seq_dir)
    orb      = cv2.ORB_create(nfeatures=2000)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_accum   = gt_poses[0].copy()
    T_prev    = T_accum.copy()
    keyframes = []
    poses_out = [T_accum.copy()]
    gt_errors = []
    orb_ok=0; tnet_ok=0; anchor_ok=0; skipped=0; integrated=0

    with torch.no_grad():
        for i in range(len(ds)):
            b  = ds[i]
            t  = b['color'].unsqueeze(0).to(DEVICE)
            t1 = b['color_next'].unsqueeze(0).to(DEVICE)

            depth_np = disp_to_depth(
                dec(enc(t))[('disp',0)]
            ).squeeze().cpu().numpy()

            color_np = (b['color'].permute(1,2,0).numpy()*255).astype(np.uint8)
            gray     = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
            kps, des = orb.detectAndCompute(gray, None)

            # ── Anchor injection ──────────────────────────────
            if i > 0 and i % anchor_interval == 0:
                T_accum = gt_poses[i].copy()
                T_prev  = T_accum.copy()
                anchor_ok += 1
                keyframes.append({'pose':T_accum.copy(),'kps':kps,
                                  'des':des,'depth':depth_np,'idx':i})
                poses_out.append(T_accum.copy())
                err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
                gt_errors.append(err)
                # Integrate với anchor pose (high confidence)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np.astype(np.float32)),
                    depth_scale=1.0,depth_trunc=95.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd,intrinsic,np.linalg.inv(T_accum))
                integrated += 1
                continue

            # ── Normal tracking ───────────────────────────────
            if i == 0:
                keyframes.append({'pose':T_accum.copy(),'kps':kps,
                                  'des':des,'depth':depth_np,'idx':0})
                T_proposed = T_accum.copy()
                confidence = 1.0
            else:
                kf = keyframes[-1]
                T_orb, n_inliers, n_matches = estimate_pose_orb(
                    kf['kps'],kf['des'],kf['depth'],kps,des)

                if T_orb is not None and n_inliers >= 4:
                    T_proposed = kf['pose'] @ T_orb
                    confidence = min(1.0, n_inliers / 20.0)
                    orb_ok += 1
                else:
                    vec    = tnet(t,t1)
                    T_tnet = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                    T_proposed = poses_out[-1] @ np.linalg.inv(T_tnet)
                    confidence = 0.3  # TNet less confident
                    tnet_ok += 1

                # ── Outlier rejection ─────────────────────────
                # Skip frame nếu pose jump quá lớn
                frame_dist = np.linalg.norm(
                    T_proposed[:3,3] - T_prev[:3,3])

                if frame_dist > max_frame_dist:
                    # Pose jump bất thường → skip integrate
                    skipped += 1
                    poses_out.append(T_prev.copy())  # giữ pose cũ
                    err = np.linalg.norm(T_prev[:3,3]-gt_poses[i+1][:3,3])
                    gt_errors.append(err)
                    continue  # không integrate frame này

                T_accum = T_proposed.copy()
                T_prev  = T_accum.copy()

            poses_out.append(T_accum.copy())
            if i%5==0:
                keyframes.append({'pose':T_accum.copy(),'kps':kps,
                                  'des':des,'depth':depth_np,'idx':i})

            # Integrate với confidence weighting
            # Chỉ integrate nếu confidence đủ cao
            if confidence >= 0.2:
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np.astype(np.float32)),
                    depth_scale=1.0,depth_trunc=95.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd,intrinsic,np.linalg.inv(T_accum))
                integrated += 1

            err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
            gt_errors.append(err)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = OUT + f'{name}.ply'
    o3d.io.write_triangle_mesh(out, mesh)

    print(f'\n{name}:')
    print(f'  ORB={orb_ok} TNet={tnet_ok} Anchors={anchor_ok}')
    print(f'  Skipped(outlier)={skipped}  Integrated={integrated}/{len(ds)}')
    print(f'  Drift avg={np.mean(gt_errors):.1f}mm  max={np.max(gt_errors):.1f}mm')
    print(f'  Mesh: V={len(mesh.vertices)} F={len(mesh.triangles)}')
    return mesh

def align_eval(pred_mesh, gt_obj):
    gt = o3d.io.read_triangle_mesh(gt_obj)
    pp = pred_mesh.sample_points_uniformly(10000)
    gp = gt.sample_points_uniformly(10000)
    pp.translate(np.asarray(gp.points).mean(0)-np.asarray(pp.points).mean(0))
    pp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    gp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    r=o3d.pipelines.registration.registration_icp(
        pp,gp,30.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    pp.transform(r.transformation)
    d1=np.asarray(pp.compute_point_cloud_distance(gp))
    d2=np.asarray(gp.compute_point_cloud_distance(pp))
    md  = round(float(d1.mean()),3)
    cov = round(float((d2<10).mean()),4)
    fit = round(r.fitness,4)
    print(f'  → MeanDist={md}mm  Coverage={cov}  ICP_fitness={fit}')
    return md, cov

enc,dec,tnet = load_full_model(BASE+'tnet_v2all.pth')
seq_dir = V2BASE + 'c2_sigmoidv3_t1'
gt_obj  = V2BASE + 'c2_sigmoidv3_t1/coverage_mesh.obj'

print('='*60)
print('Robust SLAM — anchor + outlier rejection')
print('='*60)

# Thử các threshold khác nhau
configs = [
    {'anchor_interval':50,  'max_frame_dist':20.0, 'name':'robust_a50_d20'},
    {'anchor_interval':50,  'max_frame_dist':30.0, 'name':'robust_a50_d30'},
    {'anchor_interval':100, 'max_frame_dist':20.0, 'name':'robust_a100_d20'},
]

for cfg in configs:
    mesh = slam_robust(enc,dec,tnet,seq_dir,**cfg)
    align_eval(mesh, gt_obj)

print()
print('Reference:')
print('  GT pose:      MeanDist=7.3mm   Coverage=0.7419')
print('  SLAM only:    MeanDist=129mm   Coverage=0.7774 (broken)')
print('  Anchor only:  MeanDist=132mm   Coverage=0.69   (broken)')
