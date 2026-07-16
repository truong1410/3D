import os
import numpy as np
import cv2
import tifffile
import torch
from torch.utils.data import Dataset

INTRINSICS = {
    "width":  1350, "height": 1080,
    "cx": 678.544839263292, "cy": 542.975887548343,
    "a0": 769.243600037458, "a1": 0.0,
    "a2": -0.000812770624150226,
    "a3": 6.25674244578925e-07,
    "a4": -1.19662182144280e-09,
    "c": 0.999986882249990,
    "d": 0.00288273829525059,
    "e": -0.00296316513429569
}

def generate_ray_map(intrinsics):
    ix, iy = np.meshgrid(
        np.arange(intrinsics['width']),
        np.arange(intrinsics['height']))
    uvp_x = ix - intrinsics['cx']
    uvp_y = iy - intrinsics['cy']
    uvp = np.stack([uvp_x, uvp_y], axis=-1)
    stretchMat = np.array([
        [intrinsics["c"], intrinsics["d"]],
        [intrinsics["e"], 1.0]])
    inv_stretch = np.linalg.inv(stretchMat)
    uvpp = np.einsum('ij,...j->...i', inv_stretch, uvp)
    rho = np.sqrt(uvpp[..., 0]**2 + uvpp[..., 1]**2)
    z = (intrinsics['a0'] +
         intrinsics['a2'] * rho**2 +
         intrinsics['a3'] * rho**3 +
         intrinsics['a4'] * rho**4)
    rays = np.stack([uvpp[..., 0], uvpp[..., 1], z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return rays / norms

def load_poses(data_dir):
    """
    Load camera-to-world poses.
    C3VD stores row-major matrices → cần transpose để ra column-major
    standard: T[i] là camera-to-world 4x4 column-major
    """
    poses = []
    with open(os.path.join(data_dir, 'pose.txt')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pose = np.fromstring(line, dtype=float, sep=',')
            if pose.size == 16:
                # Row-major → transpose → column-major
                T = pose.reshape(4, 4).T
                poses.append(T)
    return poses

def detect_version(data_dir):
    """
    v1: rgb ở root, tên N_color.png
    v2: rgb ở thư mục con rgb/NNNN.png
    """
    if os.path.isdir(os.path.join(data_dir, 'rgb')):
        return 'v2'
    return 'v1'

class C3VDDataset(Dataset):
    def __init__(self, data_dir, size=(640, 512)):
        self.data_dir = data_dir
        self.size     = size
        self.poses    = load_poses(data_dir)
        self.n_frames = len(self.poses)
        self.ray_map  = generate_ray_map(INTRINSICS)
        self.version  = detect_version(data_dir)
        print(f"Dataset: {os.path.basename(data_dir)} | "
              f"Frames: {self.n_frames} | Version: {self.version}")

    def __len__(self):
        return self.n_frames - 1

    def _color_path(self, idx):
        if self.version == 'v2':
            return os.path.join(self.data_dir, 'rgb', f'{idx:04d}.png')
        p = os.path.join(self.data_dir, f'{idx}_color.png')
        if os.path.exists(p):
            return p
        return os.path.join(self.data_dir, f'{idx:04d}_color.png')

    def _depth_path(self, idx):
        if self.version == 'v2':
            return os.path.join(self.data_dir, 'depth',
                                f'{idx:04d}_depth.tiff')
        return os.path.join(self.data_dir, f'{idx:04d}_depth.tiff')

    def __getitem__(self, idx):
        return {
            'color':      self._load_color(idx),
            'color_next': self._load_color(idx + 1),
            'depth_gt':   self._load_depth(idx),
            'idx':        idx
        }

    def _load_color(self, idx):
        img = cv2.imread(self._color_path(idx))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size)
        return torch.from_numpy(
            img.astype(np.float32) / 255.0).permute(2, 0, 1)

    def _load_depth(self, idx):
        depth = tifffile.imread(self._depth_path(idx)).astype(np.float32)
        depth = np.where((depth == 0) | (depth == 65535), np.nan, depth)
        depth = depth / 65535.0 * 100.0
        depth = cv2.resize(depth, self.size,
                           interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(depth).unsqueeze(0)
