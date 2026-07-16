import sys, numpy as np, cv2, open3d as o3d, torch, os
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from tnet import TNet
from losses import disp_to_depth, pose_vec_to_mat
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE   = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/'
SCREEN = BASE + 'Screening/'
OUT    = BASE + 'output/slam_superpoint/'
os.makedirs(OUT, exist_ok=True)

W,H=640,512; fx=349.0; fy=349.0; cx=237.5; cy=237.5
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)
intrinsic=o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)

print('Loading models...')
enc=ResnetEncoder(18,pretrained=False).to(DEVICE)
dec=DepthDecoder(enc.num_ch_enc).to(DEVICE)
tnet=TNet().to(DEVICE)
ck=torch.load(BASE+'tnet_screening_ft.pth',weights_only=False,map_location=DEVICE)
enc.load_state_dict(ck['encoder']); dec.load_state_dict(ck['decoder'])
tnet.load_state_dict(ck['tnet'])
enc.eval(); dec.eval(); tnet.eval()

extractor=SuperPoint(max_num_keypoints=1024).eval().to(DEVICE)
matcher=LightGlue(features='superpoint').eval().to(DEVICE)
print('Done.')

def extract_sp(color_np):
    gray=cv2.cvtColor(color_np,cv2.COLOR_RGB2GRAY)
    t=torch.from_numpy(gray.astype(np.float32)/255.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): return extractor.extract(t)

def estimate_pose_sp(feats0,depth0,feats1):
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
    R,_=cv2.Rodrigues(rvec)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
    return T,len(inliers)

def run_sp_screening(seq_dir, pose_file, out_name, step=5):
    # Load GT poses for coordinate anchor (frame 0 only)
    poses=[]
    with open(pose_file) as f:
        for line in f:
            p=np.fromstring(line.strip(),dtype=float,sep=',')
            if p.size==16: poses.append(p.reshape(4,4).T)

    rgb_dir=os.path.join(seq_dir,'rgb')
    frames=sorted(os.listdir(rgb_dir),key=lambda x:int(x.replace('.png','')))
    use_frames=[(i,f) for i,f in enumerate(frames) if i%step==0]
    n_use=len(use_frames)

    vol=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2.0,sdf_trunc=10.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # Frame 0 anchor — use GT pose
    T_accum=poses[0].copy() if poses else np.eye(4)
    T_prev=T_accum.copy()
    keyframes=[]; poses_out=[T_accum.copy()]
    sp_ok=0; tnet_ok=0; skipped=0

    with torch.no_grad():
        for idx,(frame_i,fname) in enumerate(use_frames):
            if idx%100==0:
                print(f'  {out_name}: frame {idx}/{n_use} | '
                      f'SP={sp_ok} TNet={tnet_ok} Skip={skipped}',
                      flush=True)

            path=os.path.join(rgb_dir,fname)
            cn=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
            cn=cv2.resize(cn,(W,H))

            t_gpu=torch.from_numpy(cn.astype(np.float32)/255.0
                ).permute(2,0,1).unsqueeze(0).to(DEVICE)
            depth_np=disp_to_depth(dec(enc(t_gpu))[('disp',0)]
                ).squeeze().cpu().numpy()
            feats=extract_sp(cn)

            if idx==0:
                keyframes.append({'pose':T_accum.copy(),
                    'feats':feats,'depth':depth_np})
            else:
                kf=keyframes[-1]
                T_sp,n=estimate_pose_sp(kf['feats'],kf['depth'],feats)
                if T_sp is not None and n>=6:
                    T_proposed=kf['pose']@T_sp; sp_ok+=1
                else:
                    # TNet fallback
                    prev_cn=(keyframes[-1].get('color',cn))
                    t0=torch.from_numpy(prev_cn.astype(np.float32)/255.0
                        ).permute(2,0,1).unsqueeze(0).to(DEVICE)
                    t1=torch.from_numpy(cn.astype(np.float32)/255.0
                        ).permute(2,0,1).unsqueeze(0).to(DEVICE)
                    vec=tnet(t0,t1)
                    T_tnet=pose_vec_to_mat(vec).squeeze().cpu().numpy()
                    T_proposed=poses_out[-1]@np.linalg.inv(T_tnet)
                    tnet_ok+=1

                frame_dist=np.linalg.norm(T_proposed[:3,3]-T_prev[:3,3])
                if frame_dist>60.0:
                    skipped+=1; poses_out.append(T_prev.copy()); continue

                T_accum=T_proposed.copy(); T_prev=T_accum.copy()

            poses_out.append(T_accum.copy())
            if idx%5==0:
                kf_new={'pose':T_accum.copy(),'feats':feats,
                         'depth':depth_np,'color':cn.copy()}
                keyframes.append(kf_new)

            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(cn),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_scale=1.0,depth_trunc=95.0,
                convert_rgb_to_intensity=False)
            vol.integrate(rgbd,intrinsic,np.linalg.inv(T_accum))

    mesh=vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out=OUT+out_name+'.ply'
    o3d.io.write_triangle_mesh(out,mesh)
    area=mesh.get_surface_area()
    print(f'\n{out_name}: SP={sp_ok} TNet={tnet_ok} Skip={skipped}')
    print(f'  Vertices={len(mesh.vertices)} '
          f'Triangles={len(mesh.triangles)} '
          f'Area={area:.0f}mm²')
    return len(mesh.vertices), len(mesh.triangles), area

seqs=[
    ('c0_full_t1_v1','pose_c0_full_t1_v1.txt','sp_screening_t1'),
    ('c0_full_t2_v1','pose_c0_full_t2_v1.txt','sp_screening_t2'),
    ('c0_full_t3_v1','pose_c0_full_t3_v1.txt','sp_screening_t3'),
    ('c0_full_t4_v1','pose_c0_full_t4_v1.txt','sp_screening_t4'),
]

print()
print('='*65)
print('SuperPoint SLAM — Screening colonoscopy (real patients)')
print(f"{'Sequence':<22} {'Frames':>8} {'Vertices':>10} {'Area(mm²)':>12}")
print('-'*65)

for seq_name,pose_file,out_name in seqs:
    seq_dir  =SCREEN+seq_name
    pose_path=os.path.join(seq_dir,pose_file)
    if not os.path.exists(pose_path):
        print(f'{seq_name}: pose not found'); continue
    rgb_dir=os.path.join(seq_dir,'rgb')
    n_frames=len([f for i,f in enumerate(
        sorted(os.listdir(rgb_dir))) if i%5==0])
    print(f'\n{seq_name} ({n_frames} frames):')
    v,t,a=run_sp_screening(seq_dir,pose_path,out_name,step=5)
    print(f"{seq_name:<22} {n_frames:>8} {v:>10,} {a:>12,.0f}")

print('='*65)
print('Saved to:', OUT)
