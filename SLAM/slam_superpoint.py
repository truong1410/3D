import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
V2BASE  = BASE + 'CV3Dv2/'
OUT     = BASE + 'output/slam_superpoint/'
os.makedirs(OUT, exist_ok=True)

W, H = 640, 512
fx=349.0; fy=349.0; cx=237.5; cy=237.5
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)

# Load SuperPoint + LightGlue
print('Loading SuperPoint + LightGlue...')
extractor = SuperPoint(max_num_keypoints=1024).eval().to(DEVICE)
matcher   = LightGlue(features='superpoint').eval().to(DEVICE)
print('Done.')

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

def extract_sp(color_np):
    """Extract SuperPoint features"""
    gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
    img_t = torch.from_numpy(
        gray.astype(np.float32)/255.0
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    with torch.no_grad():
        feats = extractor.extract(img_t)
    return feats

def match_sp(feats0, feats1):
    """Match SuperPoint features with LightGlue"""
    with torch.no_grad():
        matches_out = matcher({'image0': feats0, 'image1': feats1})
    feats0_  = rbd(feats0)
    feats1_  = rbd(feats1)
    matches_ = rbd(matches_out)
    matches  = matches_['matches']  # (N, 2)
    if matches.shape[0] < 4:
        return None, 0, 0, 0
        return None, 0
    kps0 = feats0_['keypoints'][matches[:,0]].cpu().numpy()
    kps1 = feats1_['keypoints'][matches[:,1]].cpu().numpy()
    return kps0, kps1, matches.shape[0]

def estimate_pose_sp(feats0, depth0, feats1):
    """Estimate pose from SuperPoint matches + PnP"""
    result = match_sp(feats0, feats1)
    if result is None or result[0] is None: return None, 0
    kps0, kps1, n_matches = result

    pts3d, pts2d = [], []
    for (u0,v0), (u1,v1) in zip(kps0, kps1):
        u0i, v0i = int(u0), int(v0)
        if 0<=u0i<W and 0<=v0i<H:
            d = depth0[v0i, u0i]
            if 1.0 < d < 95.0:
                pts3d.append([(u0i-cx)*d/fx, (v0i-cy)*d/fy, d])
                pts2d.append([u1, v1])

    if len(pts3d) < 5: return None, 0

    pts3d = np.array(pts3d, dtype=np.float32)
    pts2d = np.array(pts2d, dtype=np.float32)
    ok,rvec,tvec,inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2.0, confidence=0.99)

    if not ok or inliers is None or len(inliers) < 4:
        return None, 0

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T, len(inliers)

def slam_superpoint(enc, dec, tnet, seq_dir,
                    anchor_interval=50,
                    max_frame_dist=25.0,
                    name='superpoint'):
    ds       = C3VDDataset(seq_dir)
    gt_poses = load_poses(seq_dir)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_accum   = gt_poses[0].copy()
    T_prev    = T_accum.copy()
    keyframes = []  # {pose, feats, depth, idx}
    poses_out = [T_accum.copy()]
    gt_errors = []
    sp_ok=0; tnet_ok=0; anchor_ok=0; skipped=0; integrated=0

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

            # Extract SuperPoint features
            feats_curr = extract_sp(color_np)

            # ── Anchor ───────────────────────────────────────
            if i > 0 and i % anchor_interval == 0:
                T_accum = gt_poses[i].copy()
                T_prev  = T_accum.copy()
                anchor_ok += 1
                keyframes.append({'pose':T_accum.copy(),
                                  'feats':feats_curr,
                                  'depth':depth_np,'idx':i})
                poses_out.append(T_accum.copy())
                err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
                gt_errors.append(err)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np.astype(np.float32)),
                    depth_scale=1.0, depth_trunc=95.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd, intrinsic_o3d, np.linalg.inv(T_accum))
                integrated += 1
                continue

            # ── Tracking ─────────────────────────────────────
            if i == 0:
                keyframes.append({'pose':T_accum.copy(),
                                  'feats':feats_curr,
                                  'depth':depth_np,'idx':0})
                poses_out.append(T_accum.copy())
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np.astype(np.float32)),
                    depth_scale=1.0, depth_trunc=95.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd, intrinsic_o3d, np.linalg.inv(T_accum))
                integrated += 1
                err = np.linalg.norm(T_accum[:3,3]-gt_poses[1][:3,3])
                gt_errors.append(err)
                continue

            kf = keyframes[-1]

            # SuperPoint + LightGlue matching
            T_sp, n_sp = estimate_pose_sp(
                kf['feats'], kf['depth'], feats_curr)

            if T_sp is not None and n_sp >= 6:
                T_proposed = kf['pose'] @ T_sp
                sp_ok += 1
            else:
                # TNet fallback
                vec    = tnet(t, t1)
                T_tnet = pose_vec_to_mat(vec).squeeze().cpu().numpy()
                T_proposed = poses_out[-1] @ np.linalg.inv(T_tnet)
                tnet_ok += 1

            # Outlier rejection
            frame_dist = np.linalg.norm(
                T_proposed[:3,3] - T_prev[:3,3])

            if frame_dist > max_frame_dist:
                skipped += 1
                poses_out.append(T_prev.copy())
                err = np.linalg.norm(T_prev[:3,3]-gt_poses[i+1][:3,3])
                gt_errors.append(err)
                continue

            T_accum = T_proposed.copy()
            T_prev  = T_accum.copy()
            poses_out.append(T_accum.copy())

            if i % 5 == 0:
                keyframes.append({'pose':T_accum.copy(),
                                  'feats':feats_curr,
                                  'depth':depth_np,'idx':i})

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0, depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intrinsic_o3d, np.linalg.inv(T_accum))
            integrated += 1

            err = np.linalg.norm(T_accum[:3,3]-gt_poses[i+1][:3,3])
            gt_errors.append(err)

            if (i+1) % 100 == 0:
                print(f'  Frame {i+1}/{len(ds)} | '
                      f'SP={sp_ok} TNet={tnet_ok} '
                      f'Skip={skipped} Drift={err:.1f}mm')

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out  = OUT + f'{name}.ply'
    o3d.io.write_triangle_mesh(out, mesh)

    print(f'\n{name}:')
    print(f'  SuperPoint={sp_ok} TNet={tnet_ok} '
          f'Anchors={anchor_ok} Skipped={skipped}')
    print(f'  Integrated={integrated}/{len(ds)}')
    print(f'  Drift avg={np.mean(gt_errors):.1f}mm '
          f'max={np.max(gt_errors):.1f}mm')
    print(f'  Mesh: V={len(mesh.vertices)} F={len(mesh.triangles)}')
    return mesh

def align_eval(pred_mesh, gt_obj):
    gt = o3d.io.read_triangle_mesh(gt_obj)
    pp = pred_mesh.sample_points_uniformly(10000)
    gp = gt.sample_points_uniformly(10000)
    pp.translate(np.asarray(gp.points).mean(0)-np.asarray(pp.points).mean(0))
    pp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    gp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10.0,max_nn=30))
    r = o3d.pipelines.registration.registration_icp(
        pp, gp, 30.0,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100))
    pp.transform(r.transformation)
    d1 = np.asarray(pp.compute_point_cloud_distance(gp))
    d2 = np.asarray(gp.compute_point_cloud_distance(pp))
    md  = round(float(d1.mean()),3)
    cov = round(float((d2<10).mean()),4)
    fit = round(r.fitness,4)
    print(f'  → MeanDist={md}mm  Coverage={cov}  ICP_fitness={fit}')
    return md, cov

# Load models
enc, dec, tnet = load_full_model(BASE+'tnet_v2all.pth')
seq_dir = V2BASE + 'c2_sigmoidv3_t1'
gt_obj  = V2BASE + 'c2_sigmoidv3_t1/coverage_mesh.obj'

print('='*60)
print('SuperPoint + LightGlue SLAM on debris sequence')
print('='*60)

# Test 2 configs
for anchor, max_dist in [(50, 25.0), (50, 15.0)]:
    name = f'sp_a{anchor}_d{int(max_dist)}'
    mesh = slam_superpoint(enc,dec,tnet,seq_dir,
                           anchor_interval=anchor,
                           max_frame_dist=max_dist,
                           name=name)
    align_eval(mesh, gt_obj)

print()
print('Reference:')
print('  GT pose:   MeanDist=7.3mm   Coverage=0.7419')
print('  ORB SLAM:  MeanDist=129mm   Coverage=0.7774 (broken)')

def align_eval_robust(pred_mesh, gt_obj, margin=30.0):
    """Evaluation với outlier removal + GT-bbox crop"""
    gt  = o3d.io.read_triangle_mesh(gt_obj)
    pp  = pred_mesh.sample_points_uniformly(50000)
    gp  = gt.sample_points_uniformly(10000)
    gt_pts = np.asarray(gp.points)

    # Outlier removal
    pp_clean, _ = pp.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)

    # Crop to GT bbox + margin
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        gt_pts.min(0) - margin, gt_pts.max(0) + margin)
    pp_crop = pp_clean.crop(bbox)

    if len(pp_crop.points) < 100:
        print('  → Too few points after crop')
        return None, None

    pp_crop.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    gp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))

    r = o3d.pipelines.registration.registration_icp(
        pp_crop, gp, 30.0,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200))
    pp_crop.transform(r.transformation)

    d1 = np.asarray(pp_crop.compute_point_cloud_distance(gp))
    d2 = np.asarray(gp.compute_point_cloud_distance(pp_crop))
    md  = round(float(d1.mean()), 3)
    cov = round(float((d2<10).mean()), 4)
    fit = round(r.fitness, 4)
    print(f'  → MeanDist={md}mm  Coverage={cov}  '
          f'Missed={round(1-cov,4)}  ICP={fit}')
    return md, cov
