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
os.makedirs(OUT, exist_ok=True)

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
    if des1 is None or des2 is None: return None, 0
    matches = sorted(bf.match(des1,des2), key=lambda x:x.distance)
    good    = [m for m in matches if m.distance<50][:100]
    if len(good)<6: return None, 0
    pts3d,pts2d=[],[]
    for m in good:
        u1,v1=int(kps1[m.queryIdx].pt[0]),int(kps1[m.queryIdx].pt[1])
        if 0<=u1<W and 0<=v1<H:
            d=depth1[v1,u1]
            if 1.0<d<95.0:
                pts3d.append([(u1-cx)*d/fx,(v1-cy)*d/fy,d])
                pts2d.append(kps2[m.trainIdx].pt)
    if len(pts3d)<5: return None,0
    pts3d=np.array(pts3d,dtype=np.float32)
    pts2d=np.array(pts2d,dtype=np.float32)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(
        pts3d,pts2d,K,None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,confidence=0.99)
    if not ok or inliers is None or len(inliers)<4: return None,0
    R,_=cv2.Rodrigues(rvec)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T,len(inliers)

def slam_with_anchor(enc, dec, tnet, seq_dir,
                     anchor_interval=50,   # inject GT pose mỗi N frames
                     name='slam_anchor'):
    """
    Anchor-assisted SLAM:
    - Dùng GT pose làm anchor mỗi anchor_interval frames
    - Giữa các anchor: accumulate relative pose từ ORB/TNet như bình thường
    - Không cần GT pose liên tục, chỉ cần anchor points thưa
    """
    ds       = C3VDDataset(seq_dir)
    gt_poses = load_poses(seq_dir)
    orb      = cv2.ORB_create(nfeatures=2000)
    bf_lc    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # Bắt đầu từ GT pose frame 0 thay vì identity
    T_accum   = gt_poses[0].copy()
    keyframes = []
    poses_out = [T_accum.copy()]
    gt_errors = []
    orb_ok=0; tnet_ok=0; anchor_ok=0; loop_ok=0

    print(f'Starting from GT pose anchor: {T_accum[:3,3].round(1)}')

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
            # Mỗi anchor_interval frames: reset về GT pose
            # Simulate: trong thực tế có thể dùng fiducial markers
            # hoặc IMU integration để làm anchor
            if i > 0 and i % anchor_interval == 0:
                T_accum = gt_poses[i].copy()
                anchor_ok += 1
                # Cập nhật keyframe mới sau anchor
                keyframes.append({'pose':T_accum.copy(),'kps':kps,
                                  'des':des,'depth':depth_np,'idx':i})
                poses_out.append(T_accum.copy())
                err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
                gt_errors.append(err)
                # Tích hợp TSDF với GT pose
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np.astype(np.float32)),
                    depth_scale=1.0,depth_trunc=95.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd,intrinsic,np.linalg.inv(T_accum))
                continue

            # ── Normal SLAM tracking ──────────────────────────
            if i==0:
                keyframes.append({'pose':T_accum.copy(),'kps':kps,
                                  'des':des,'depth':depth_np,'idx':0})
            else:
                kf = keyframes[-1]
                T_orb, n = estimate_pose_orb(
                    kf['kps'],kf['des'],kf['depth'],kps,des)

                if T_orb is not None and n >= 4:
                    T_accum = kf['pose'] @ T_orb; orb_ok+=1
                else:
                    vec    = tnet(t,t1)
                    T_tnet = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                    T_accum = poses_out[-1] @ np.linalg.inv(T_tnet); tnet_ok+=1

                # Loop closure
                if i % 10 == 0 and des is not None and len(keyframes)>15:
                    best_n, best_ki = 0, -1
                    for ki, kf_lc in enumerate(keyframes[:-15]):
                        if kf_lc['des'] is None: continue
                        m   = bf_lc.match(des, kf_lc['des'])
                        n_m = sum(1 for x in m if x.distance<45)
                        if n_m > best_n: best_n,best_ki=n_m,ki
                    if best_n >= 20 and best_ki >= 0:
                        T_lc,n_lc = estimate_pose_orb(
                            keyframes[best_ki]['kps'],
                            keyframes[best_ki]['des'],
                            keyframes[best_ki]['depth'],
                            kps,des)
                        if T_lc is not None and n_lc >= 4:
                            T_cor = keyframes[best_ki]['pose'] @ T_lc
                            nc    = len(poses_out)-best_ki
                            for j in range(nc):
                                alpha=j/max(nc,1)
                                poses_out[best_ki+j][:3,3] += \
                                    alpha*(T_cor[:3,3]-T_accum[:3,3])
                            T_accum=T_cor; loop_ok+=1

            poses_out.append(T_accum.copy())
            if i%5==0:
                keyframes.append({'pose':T_accum.copy(),'kps':kps,
                                  'des':des,'depth':depth_np,'idx':i})

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0,depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            vol.integrate(rgbd,intrinsic,np.linalg.inv(T_accum))

            err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
            gt_errors.append(err)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = OUT + f'{name}.ply'
    o3d.io.write_triangle_mesh(out, mesh)

    print(f'\n{name} (anchor every {anchor_interval} frames):')
    print(f'  ORB={orb_ok} TNet={tnet_ok} Anchors={anchor_ok} Loops={loop_ok}')
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
    md   = round(float(d1.mean()),3)
    cov  = round(float((d2<10).mean()),4)
    fit  = round(r.fitness,4)
    print(f'  → MeanDist={md}mm  Coverage={cov}  ICP_fitness={fit}')
    return md, cov

enc,dec,tnet = load_full_model(BASE+'tnet_v2all.pth')
seq_dir = V2BASE + 'c2_sigmoidv3_t1'
gt_obj  = V2BASE + 'c2_sigmoidv3_t1/coverage_mesh.obj'

print('='*60)
print('Anchor-assisted SLAM — debris sequence')
print('Anchor: inject GT pose every N frames')
print('='*60)

# Thử 3 anchor intervals
for interval in [50, 100, 200]:
    mesh = slam_with_anchor(enc,dec,tnet,seq_dir,
                            anchor_interval=interval,
                            name=f'anchor_{interval}')
    align_eval(mesh, gt_obj)

print()
print('Reference:')
print(f'  GT pose:    MeanDist=7.3mm  Coverage=0.7419')
print(f'  SLAM only:  MeanDist=129mm  Coverage=0.7774  (mesh broken)')
