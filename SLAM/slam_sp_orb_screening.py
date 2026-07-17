import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from losses import disp_to_depth
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE   = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
SCREEN = BASE + 'Screening/'
OUT    = BASE + 'output/slam_superpoint/'

W,H=640,512; fx=349.0; fy=349.0; cx=237.5; cy=237.5
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)
intrinsic=o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)

print('Loading models...')
enc=ResnetEncoder(18,pretrained=False).to(DEVICE)
dec=DepthDecoder(enc.num_ch_enc).to(DEVICE)
ck=torch.load(BASE+'tnet_smooth.pth',weights_only=False,map_location=DEVICE)
enc.load_state_dict(ck['encoder']); dec.load_state_dict(ck['decoder'])
enc.eval(); dec.eval()

extractor=SuperPoint(max_num_keypoints=1024).eval().to(DEVICE)
matcher  =LightGlue(features='superpoint').eval().to(DEVICE)
orb_det  =cv2.ORB_create(nfeatures=2000)
print('Done. SP+ORB hybrid — no TNet pose fallback')

def extract_sp(color_np):
    gray=cv2.cvtColor(color_np,cv2.COLOR_RGB2GRAY)
    t=torch.from_numpy(gray.astype(np.float32)/255.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): return extractor.extract(t)

def pose_sp(feats0, depth0, feats1):
    with torch.no_grad():
        m_out=matcher({'image0':feats0,'image1':feats1})
    f0=rbd(feats0); f1=rbd(feats1); m=rbd(m_out)
    matches=m['matches']
    if matches.shape[0]<4: return None,0
    kps0=f0['keypoints'][matches[:,0]].cpu().numpy()
    kps1=f1['keypoints'][matches[:,1]].cpu().numpy()
    pts3d,pts2d=[],[]
    for (u0,v0),(u1,v1) in zip(kps0,kps1):
        u0i,v0i=int(u0),int(v0)
        if 0<=u0i<W and 0<=v0i<H:
            d=depth0[v0i,u0i]
            if 1.0<d<95.0:
                pts3d.append([(u0i-cx)*d/fx,(v0i-cy)*d/fy,d])
                pts2d.append([u1,v1])
    if len(pts3d)<5: return None,0
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(
        np.array(pts3d,dtype=np.float32),
        np.array(pts2d,dtype=np.float32),
        K,None,reprojectionError=2.0,confidence=0.99)
    if not ok or inliers is None or len(inliers)<4: return None,0
    R,_=cv2.Rodrigues(rvec); T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T,len(inliers)

def pose_orb(kps0, des0, depth0, kps1, des1):
    if des0 is None or des1 is None: return None,0
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=sorted(bf.match(des0,des1),key=lambda x:x.distance)
    good=[m for m in matches if m.distance<50][:100]
    if len(good)<6: return None,0
    pts3d,pts2d=[],[]
    for m in good:
        u0i,v0i=int(kps0[m.queryIdx].pt[0]),int(kps0[m.queryIdx].pt[1])
        if 0<=u0i<W and 0<=v0i<H:
            d=depth0[v0i,u0i]
            if 1.0<d<95.0:
                pts3d.append([(u0i-cx)*d/fx,(v0i-cy)*d/fy,d])
                pts2d.append(kps1[m.trainIdx].pt)
    if len(pts3d)<5: return None,0
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(
        np.array(pts3d,dtype=np.float32),
        np.array(pts2d,dtype=np.float32),
        K,None,reprojectionError=2.0,confidence=0.99)
    if not ok or inliers is None or len(inliers)<4: return None,0
    R,_=cv2.Rodrigues(rvec); T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T,len(inliers)

def run(seq_dir, pose_file, out_name, step=5):
    poses=[]
    with open(pose_file) as f:
        for line in f:
            p=np.fromstring(line.strip(),dtype=float,sep=',')
            if p.size==16: poses.append(p.reshape(4,4).T)

    rgb_dir=os.path.join(seq_dir,'rgb')
    frames=sorted(os.listdir(rgb_dir),
                  key=lambda x:int(x.replace('.png','')))
    use_frames=[(i,f) for i,f in enumerate(frames) if i%step==0]
    n_use=len(use_frames)

    vol=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0,sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_accum=poses[0].copy() if poses else np.eye(4)
    T_prev=T_accum.copy()
    keyframes=[]
    sp_ok=0; orb_ok=0; constvel=0; skipped=0

    with torch.no_grad():
        for idx,(frame_i,fname) in enumerate(use_frames):
            if idx%100==0:
                print(f'  {out_name}: {idx}/{n_use} | '
                      f'SP={sp_ok} ORB={orb_ok} '
                      f'ConstVel={constvel} Skip={skipped}',
                      flush=True)

            cn=cv2.cvtColor(
                cv2.imread(os.path.join(rgb_dir,fname)),
                cv2.COLOR_BGR2RGB)
            cn=cv2.resize(cn,(W,H))

            t_gpu=torch.from_numpy(cn.astype(np.float32)/255.0
                ).permute(2,0,1).unsqueeze(0).to(DEVICE)
            depth_np=disp_to_depth(
                dec(enc(t_gpu))[('disp',0)]
            ).squeeze().cpu().numpy()

            feats_curr=extract_sp(cn)
            gray=cv2.cvtColor(cn,cv2.COLOR_RGB2GRAY)
            kps_curr,des_curr=orb_det.detectAndCompute(gray,None)

            if idx==0:
                keyframes.append({
                    'pose':T_accum.copy(),
                    'feats':feats_curr,'depth':depth_np,
                    'kps':kps_curr,'des':des_curr})
            else:
                kf=keyframes[-1]
                T_proposed=None

                # 1. SuperPoint primary
                T_sp,n_sp=pose_sp(kf['feats'],kf['depth'],feats_curr)
                if T_sp is not None and n_sp>=6:
                    T_proposed=kf['pose']@T_sp; sp_ok+=1

                # 2. ORB fallback (no TNet)
                if T_proposed is None:
                    T_orb,n_orb=pose_orb(
                        kf['kps'],kf['des'],kf['depth'],
                        kps_curr,des_curr)
                    if T_orb is not None and n_orb>=6:
                        T_proposed=kf['pose']@T_orb; orb_ok+=1

                # 3. Constant velocity (no TNet)
                if T_proposed is None:
                    T_proposed=T_prev.copy(); constvel+=1

                # Outlier rejection
                frame_dist=np.linalg.norm(
                    T_proposed[:3,3]-T_prev[:3,3])
                if frame_dist>60.0:
                    T_proposed=T_prev.copy(); skipped+=1

                T_accum=T_proposed.copy()
                T_prev=T_accum.copy()

            if idx%5==0:
                keyframes.append({
                    'pose':T_accum.copy(),
                    'feats':feats_curr,'depth':depth_np,
                    'kps':kps_curr,'des':des_curr})

            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(cn),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0,depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            vol.integrate(rgbd,intrinsic,np.linalg.inv(T_accum))

    mesh=vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(OUT+out_name+'.ply',mesh)
    area=mesh.get_surface_area()
    print(f'\n{out_name}:')
    print(f'  SP={sp_ok} ORB={orb_ok} ConstVel={constvel} Skip={skipped}')
    print(f'  Vertices={len(mesh.vertices):,} Area={area:,.0f}mm²')
    return len(mesh.vertices), area

seqs=[
    ('c0_full_t1_v1','pose_c0_full_t1_v1.txt','sp_orb_t1'),
    ('c0_full_t2_v1','pose_c0_full_t2_v1.txt','sp_orb_t2'),
    ('c0_full_t3_v1','pose_c0_full_t3_v1.txt','sp_orb_t3'),
    ('c0_full_t4_v1','pose_c0_full_t4_v1.txt','sp_orb_t4'),
]

print()
print('='*65)
print('SP + ORB hybrid SLAM — no TNet pose fallback')
print(f"{'Sequence':<22} {'Vertices':>10} {'Area(mm²)':>12}")
print('-'*65)

for seq_name,pose_file,out_name in seqs:
    seq_dir  =SCREEN+seq_name
    pose_path=os.path.join(seq_dir,pose_file)
    if not os.path.exists(pose_path):
        print(f'{seq_name}: not found'); continue
    print(f'\n{seq_name}:')
    v,a=run(seq_dir,pose_path,out_name,step=5)
    print(f"{seq_name:<22} {v:>10,} {a:>12,.0f}")

print('='*65)
