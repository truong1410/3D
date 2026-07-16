import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE     = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE   = BASE + 'CV3Dv2/'
SCREEN   = BASE + 'Screening/'
OUT_V2   = BASE + 'output/slam_v2/'; os.makedirs(OUT_V2, exist_ok=True)
OUT_SCR  = BASE + 'output/slam_screening/'; os.makedirs(OUT_SCR, exist_ok=True)

W, H = 640, 512
fx=349.0; fy=349.0; cx=237.5; cy=237.5
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
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

def estimate_pose_orb(kps1, des1, depth1, kps2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None: return None, 0
    matches = sorted(bf.match(des1,des2), key=lambda x: x.distance)
    good    = [m for m in matches if m.distance<50][:100]
    if len(good)<8: return None, 0
    pts3d,pts2d=[],[]
    for m in good:
        u1,v1=int(kps1[m.queryIdx].pt[0]),int(kps1[m.queryIdx].pt[1])
        if 0<=u1<W and 0<=v1<H:
            d=depth1[v1,u1]
            if 1.0<d<95.0:
                pts3d.append([(u1-cx)*d/fx,(v1-cy)*d/fy,d])
                pts2d.append(kps2[m.trainIdx].pt)
    if len(pts3d)<6: return None,0
    pts3d=np.array(pts3d,dtype=np.float32)
    pts2d=np.array(pts2d,dtype=np.float32)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(
        pts3d,pts2d,K,None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2.0,confidence=0.99)
    if not ok or inliers is None or len(inliers)<6: return None,0
    R,_=cv2.Rodrigues(rvec)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T,len(inliers)

def run_slam(enc, dec, tnet, frames_iter, n_frames, gt_poses=None):
    """
    Generic SLAM runner
    frames_iter: iterator yielding (color_np, depth_np or None)
    gt_poses: list of GT poses for drift tracking (optional)
    """
    orb    = cv2.ORB_create(nfeatures=1000)
    bf_lc  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_accum   = np.eye(4)
    keyframes = []
    poses_out = [np.eye(4)]
    gt_errors = []
    orb_ok=0; tnet_ok=0; loop_ok=0

    prev_color = None; prev_depth = None

    for i, (color_np, depth_np) in enumerate(frames_iter):
        if i % 100 == 0:
            print(f'  Frame {i}/{n_frames} | '
                  f'orb={orb_ok} tnet={tnet_ok} loops={loop_ok}',
                  flush=True)

        gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
        kps, des = orb.detectAndCompute(gray, None)

        if i == 0:
            keyframes.append({
                'pose':np.eye(4),'kps':kps,
                'des':des,'depth':depth_np,'idx':0})
        else:
            kf = keyframes[-1]
            T_orb,n = estimate_pose_orb(
                kf['kps'],kf['des'],kf['depth'],kps,des)
            if T_orb is not None and n>=8:
                T_accum = kf['pose'] @ T_orb; orb_ok+=1
            else:
                # TNet fallback
                rgb_t  = torch.from_numpy(
                    prev_color.astype(np.float32)/255.0
                ).permute(2,0,1).unsqueeze(0).to(DEVICE)
                rgb_t1 = torch.from_numpy(
                    color_np.astype(np.float32)/255.0
                ).permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    vec   = tnet(rgb_t, rgb_t1)
                    T_rel = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                T_accum = poses_out[-1] @ np.linalg.inv(T_rel); tnet_ok+=1

            # Loop closure mỗi 20 frames
            if i%20==0 and des is not None and len(keyframes)>30:
                best_n,best_ki=0,-1
                for ki,kf_lc in enumerate(keyframes[:-30]):
                    if kf_lc['des'] is None: continue
                    m   = bf_lc.match(des,kf_lc['des'])
                    n_m = sum(1 for x in m if x.distance<40)
                    if n_m>best_n: best_n,best_ki=n_m,ki
                if best_n>=40 and best_ki>=0:
                    T_lc,n_lc=estimate_pose_orb(
                        keyframes[best_ki]['kps'],
                        keyframes[best_ki]['des'],
                        keyframes[best_ki]['depth'],
                        kps,des)
                    if T_lc is not None and n_lc>=10:
                        T_cor = keyframes[best_ki]['pose'] @ T_lc
                        nc    = len(poses_out)-best_ki
                        for j in range(nc):
                            alpha=j/max(nc,1)
                            poses_out[best_ki+j][:3,3] += \
                                alpha*(T_cor[:3,3]-T_accum[:3,3])
                        T_accum=T_cor; loop_ok+=1

        poses_out.append(T_accum.copy())
        if i%5==0:
            keyframes.append({
                'pose':T_accum.copy(),'kps':kps,
                'des':des,'depth':depth_np,'idx':i})

        # TSDF integrate
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_np),
            o3d.geometry.Image(depth_np.astype(np.float32)),
            depth_scale=1.0,depth_trunc=95.0,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intrinsic_o3d,np.linalg.inv(T_accum))

        # GT drift
        if gt_poses and i+1<len(gt_poses):
            err=np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
            gt_errors.append(err)

        prev_color = color_np.copy()
        prev_depth = depth_np.copy()

    mesh=volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f'  Done | ORB={orb_ok} TNet={tnet_ok} Loops={loop_ok}')
    if gt_errors:
        print(f'  Drift avg={np.mean(gt_errors):.1f}mm '
              f'final={gt_errors[-1]:.1f}mm')
    return mesh

def frames_from_dataset(enc, dec, seq_dir):
    """Yield (color_np, depth_np) từ C3VDDataset"""
    ds = C3VDDataset(seq_dir)
    for i in range(len(ds)+1):
        cp = ds._color_path(i)
        if not os.path.exists(cp): continue
        cn = cv2.cvtColor(cv2.imread(cp), cv2.COLOR_BGR2RGB)
        cn = cv2.resize(cn, (W,H))
        rgb_t = torch.from_numpy(
            cn.astype(np.float32)/255.0
        ).permute(2,0,1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dn = disp_to_depth(
                dec(enc(rgb_t))[('disp',0)]
            ).squeeze().cpu().numpy()
        yield cn, dn

def frames_from_screening(enc, dec, seq_dir, pose_file, step=5):
    """Yield (color_np, depth_np) từ screening video"""
    poses  = []
    with open(pose_file) as f:
        for line in f:
            p=np.fromstring(line.strip(),dtype=float,sep=',')
            if p.size==16: poses.append(p.reshape(4,4).T)
    rgb_dir = os.path.join(seq_dir,'rgb')
    frames  = sorted(os.listdir(rgb_dir),
                     key=lambda x:int(x.replace('.png','')))
    for i,fname in enumerate(frames):
        if i>=len(poses): break
        if i%step!=0: continue
        cn=cv2.cvtColor(cv2.imread(os.path.join(rgb_dir,fname)),
                        cv2.COLOR_BGR2RGB)
        cn=cv2.resize(cn,(W,H))
        rgb_t=torch.from_numpy(
            cn.astype(np.float32)/255.0
        ).permute(2,0,1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dn=disp_to_depth(
                dec(enc(rgb_t))[('disp',0)]
            ).squeeze().cpu().numpy()
        yield cn,dn

def align_and_eval(pred_mesh, gt_mesh, threshold=10.0):
    pred_pcd=pred_mesh.sample_points_uniformly(10000)
    gt_pcd  =gt_mesh.sample_points_uniformly(10000)
    pred_pts=np.asarray(pred_pcd.points)
    gt_pts  =np.asarray(gt_pcd.points)
    pred_pcd.translate(gt_pts.mean(0)-pred_pts.mean(0))
    pred_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    gt_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    r=o3d.pipelines.registration.registration_icp(
        pred_pcd,gt_pcd,30.0,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100))
    pred_pcd.transform(r.transformation)
    d1=np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    d2=np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    pr=(d1<threshold).mean(); rc=(d2<threshold).mean()
    return {
        'F@10mm':   round(2*pr*rc/(pr+rc) if pr+rc>0 else 0,4),
        'Coverage': round(float(rc),4),
        'Missed':   round(float(1-rc),4),
        'MeanDist': round(float(d1.mean()),3),
        'AreaRatio':round(pred_mesh.get_surface_area()/
                          gt_mesh.get_surface_area(),3),
    }

# ── Part 1: SLAM trên v2 sequences ───────────────────────────
print('\n' + '='*60)
print('Part 1: SLAM on C3VDv2')
print('='*60)

enc, dec, tnet = load_full_model(BASE+'tnet_v2all.pth')

v2_test = [
    ('c2_sigmoid_t1',   'clean',  V2BASE+'c2_sigmoid_t1/coverage_mesh.obj'),
    ('c2_sigmoid_t2',   'clean',  V2BASE+'c2_sigmoid_t2/coverage_mesh.obj'),
    ('c2_sigmoidv3_t1', 'debris', V2BASE+'c2_sigmoidv3_t1/coverage_mesh.obj'),
    ('c2_sigmoidv3_t2', 'debris', V2BASE+'c2_sigmoidv3_t2/coverage_mesh.obj'),
]

print(f"\n{'Sequence':<22} {'Type':<8} {'F@10mm':>8} {'Coverage':>10} "
      f"{'Missed':>8} {'MeanDist':>10}")
print('-'*70)

clean_cov=[]; debris_cov=[]
for seq_name,seq_type,gt_obj in v2_test:
    seq_dir  = V2BASE+seq_name
    gt_poses = load_poses(seq_dir)
    ds       = C3VDDataset(seq_dir)
    n_frames = len(ds)+1

    print(f'{seq_name} ({seq_type}):')
    gen  = frames_from_dataset(enc, dec, seq_dir)
    mesh = run_slam(enc,dec,tnet,gen,n_frames,gt_poses)

    out=OUT_V2+f'{seq_name}_slamv3.ply'
    o3d.io.write_triangle_mesh(out,mesh)

    gt_mesh=o3d.io.read_triangle_mesh(gt_obj)
    r=align_and_eval(mesh,gt_mesh)
    print(f"{seq_name:<22} {seq_type:<8} {r['F@10mm']:>8} "
          f"{r['Coverage']:>10} {r['Missed']:>8} {r['MeanDist']:>10}")

    if seq_type=='clean': clean_cov.append(r['Coverage'])
    else:                 debris_cov.append(r['Coverage'])

print('-'*70)
print(f"avg clean coverage:  {round(np.mean(clean_cov),4)}")
print(f"avg debris coverage: {round(np.mean(debris_cov),4)}")

# ── Part 2: SLAM trên Screening videos ───────────────────────
print('\n' + '='*60)
print('Part 2: SLAM on Screening colonoscopy (real patients)')
print('='*60)

enc, dec, tnet = load_full_model(BASE+'tnet_smooth.pth')

screening_seqs = [
    ('c0_full_t1_v1', 'pose_c0_full_t1_v1.txt'),
    ('c0_full_t2_v1', 'pose_c0_full_t2_v1.txt'),
    ('c0_full_t3_v1', 'pose_c0_full_t3_v1.txt'),
    ('c0_full_t4_v1', 'pose_c0_full_t4_v1.txt'),
]

print(f"\n{'Sequence':<22} {'Frames':>8} {'Vertices':>10} "
      f"{'Triangles':>10} {'Area(mm²)':>12}")
print('-'*65)

for seq_name,pose_file in screening_seqs:
    seq_dir   = SCREEN+seq_name
    pose_path = os.path.join(seq_dir,pose_file)
    if not os.path.exists(pose_path):
        print(f'{seq_name}: pose file not found'); continue

    # Đếm frames
    frames_all = sorted(
        os.listdir(os.path.join(seq_dir,'rgb')),
        key=lambda x:int(x.replace('.png','')))
    n_use = len([f for i,f in enumerate(frames_all) if i%5==0])

    print(f'{seq_name} ({n_use} frames):')
    gen  = frames_from_screening(enc,dec,seq_dir,pose_path,step=5)
    mesh = run_slam(enc,dec,tnet,gen,n_use)

    out=OUT_SCR+f'{seq_name}_slam.ply'
    o3d.io.write_triangle_mesh(out,mesh)
    area=mesh.get_surface_area()
    print(f"{seq_name:<22} {n_use:>8} {len(mesh.vertices):>10} "
          f"{len(mesh.triangles):>10} {area:>12.1f}")

print('\nAll meshes saved!')
print(f'  V2:        {OUT_V2}')
print(f'  Screening: {OUT_SCR}')
