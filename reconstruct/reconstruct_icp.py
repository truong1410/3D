import sys, numpy as np, open3d as o3d, torch, cv2, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from c3vd_dataset import C3VDDataset, INTRINSICS, generate_ray_map, load_poses
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE    = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
OUT_DIR = BASE + 'output/icp/'
os.makedirs(OUT_DIR, exist_ok=True)

W_vol, H_vol = 512, 640
fx=349.0; fy=349.0; cx=237.5; cy=237.5
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
    W_vol, H_vol, fx, fy, cx, cy)
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

def depth_to_pcd(depth_np, color_np):
    """Convert depth map → point cloud"""
    color_o3d = o3d.geometry.Image(color_np.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0, depth_trunc=95.0,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic_o3d)
    return pcd

def icp_refine(source_pcd, target_pcd, T_init, max_dist=5.0):
    """
    Refine pose estimate using Point-to-Plane ICP
    source: current frame point cloud
    target: accumulated point cloud từ frames trước
    T_init: initial pose từ TNet
    """
    if len(target_pcd.points) < 100:
        return T_init, False

    # Estimate normals cho point-to-plane ICP
    source_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=max_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50))

    # Chỉ accept ICP nếu fitness đủ tốt
    if result.fitness > 0.3:
        return result.transformation, True
    return T_init, False

def reconstruct_with_icp(enc, dec, tnet, seq_dir,
                          use_icp=True, icp_interval=5):
    """
    Reconstruct với TNet pose + ICP refinement
    icp_interval: refine mỗi N frames
    """
    ds    = C3VDDataset(seq_dir)
    poses = load_poses(seq_dir)  # GT poses để compare

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0, sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_accum  = np.eye(4)  # Accumulated TNet pose
    all_pcds = []         # Lưu point clouds để ICP
    icp_count = 0
    gt_errors = []
    icp_errors = []

    with torch.no_grad():
        for i in range(len(ds)):
            b  = ds[i]
            t  = b['color'].unsqueeze(0).to(DEVICE)
            t1 = b['color_next'].unsqueeze(0).to(DEVICE)

            # Predict depth
            depth_np = disp_to_depth(
                dec(enc(t))[('disp',0)]
            ).squeeze().cpu().numpy()

            # Predict relative pose từ TNet
            vec   = tnet(t, t1)
            T_rel = pose_vec_to_mat(vec).squeeze().cpu().numpy()

            # Accumulate TNet pose
            T_accum = T_accum @ np.linalg.inv(T_rel)

            # Convert frame to point cloud
            color_np = b['color'].permute(1,2,0).numpy()
            color_np = (color_np * 255).astype(np.uint8)
            curr_pcd = depth_to_pcd(depth_np, color_np)
            curr_pcd.transform(T_accum)

            # ICP refinement mỗi icp_interval frames
            if use_icp and i > 0 and i % icp_interval == 0:
                # Build target từ accumulated point clouds
                if len(all_pcds) > 0:
                    target = all_pcds[-1]  # dùng frame trước làm target
                    T_refined, success = icp_refine(
                        curr_pcd, target, T_accum)
                    if success:
                        T_accum = T_refined
                        curr_pcd.transform(
                            np.linalg.inv(T_accum) @ T_refined)
                        icp_count += 1

            all_pcds.append(curr_pcd)
            if len(all_pcds) > 10:  # giữ 10 frames gần nhất
                all_pcds.pop(0)

            # Integrate vào TSDF
            b_color = b['color'].permute(1,2,0).numpy()
            b_color = (b_color * 255).astype(np.uint8)
            color_o3d = o3d.geometry.Image(b_color.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0, depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic_o3d, np.linalg.inv(T_accum))

            # Track error vs GT
            gt_err = np.linalg.norm(T_accum[:3,3] - poses[i+1][:3,3])
            gt_errors.append(gt_err)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"  ICP corrections applied: {icp_count}/{len(ds)//icp_interval}")
    print(f"  Final drift: {gt_errors[-1]:.2f} mm")
    print(f"  Avg drift:   {np.mean(gt_errors):.2f} mm")
    return mesh, gt_errors

def fscore(pred_mesh, gt_mesh, t=5.0):
    pp = pred_mesh.sample_points_uniformly(10000)
    gp = gt_mesh.sample_points_uniformly(10000)
    d1 = np.asarray(pp.compute_point_cloud_distance(gp))
    d2 = np.asarray(gp.compute_point_cloud_distance(pp))
    pr = (d1<t).mean(); rc = (d2<t).mean()
    dist = round(float(d1.mean()),3)
    f = round(2*pr*rc/(pr+rc) if pr+rc>0 else 0, 4)
    return f, dist

# Load model
enc, dec, tnet = load_full_model(BASE + 'tnet_smooth.pth')
seq_dir = BASE + 'cecum_t1_a'
gt_mesh = o3d.io.read_triangle_mesh(BASE + 'cecum_t1_a/coverage_mesh.obj')

print('='*60)
print('Comparing TNet pose vs TNet+ICP pose')
print('='*60)

# TNet pose only (no ICP)
print('\n1. TNet pose (no ICP):')
mesh1, errs1 = reconstruct_with_icp(
    enc, dec, tnet, seq_dir, use_icp=False)
o3d.io.write_triangle_mesh(OUT_DIR + 'tnet_noICP.ply', mesh1)
f5_1, d1_ = fscore(mesh1, gt_mesh, 5.0)
f10_1, _  = fscore(mesh1, gt_mesh, 10.0)
print(f"  MeanDist={d1_}mm  F@5mm={f5_1}  F@10mm={f10_1}")

# TNet + ICP refinement
print('\n2. TNet + ICP refinement (every 5 frames):')
mesh2, errs2 = reconstruct_with_icp(
    enc, dec, tnet, seq_dir, use_icp=True, icp_interval=5)
o3d.io.write_triangle_mesh(OUT_DIR + 'tnet_ICP5.ply', mesh2)
f5_2, d2_ = fscore(mesh2, gt_mesh, 5.0)
f10_2, _  = fscore(mesh2, gt_mesh, 10.0)
print(f"  MeanDist={d2_}mm  F@5mm={f5_2}  F@10mm={f10_2}")

# TNet + ICP refinement (every 10 frames)
print('\n3. TNet + ICP refinement (every 10 frames):')
mesh3, errs3 = reconstruct_with_icp(
    enc, dec, tnet, seq_dir, use_icp=True, icp_interval=10)
o3d.io.write_triangle_mesh(OUT_DIR + 'tnet_ICP10.ply', mesh3)
f5_3, d3_ = fscore(mesh3, gt_mesh, 5.0)
f10_3, _  = fscore(mesh3, gt_mesh, 10.0)
print(f"  MeanDist={d3_}mm  F@5mm={f5_3}  F@10mm={f10_3}")

print()
print('='*60)
print('Summary:')
print(f"{'Method':<25} {'MeanDist':>10} {'F@5mm':>8} {'F@10mm':>8}")
print('-'*60)
print(f"{'GT pose (reference)':<25} {'7.276mm':>10} {'0.2923':>8} {'0.7277':>8}")
print(f"{'TNet pose (no ICP)':<25} {str(d1_)+'mm':>10} {f5_1:>8} {f10_1:>8}")
print(f"{'TNet + ICP@5':<25} {str(d2_)+'mm':>10} {f5_2:>8} {f10_2:>8}")
print(f"{'TNet + ICP@10':<25} {str(d3_)+'mm':>10} {f5_3:>8} {f10_3:>8}")
print('='*60)
