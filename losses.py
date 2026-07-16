import torch
import kornia
import torch.nn.functional as F
import numpy as np

def pose_vec_to_mat(vec):
    B = vec.shape[0]
    rot = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
              vec[:, 3:])
    T = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(vec.device)
    T[:, :3, :3] = rot
    T[:, :3,  3] = vec[:, :3]
    return T

def photometric_loss(img, img_warped):
    l1   = (img - img_warped).abs().mean()
    ssim = kornia.losses.ssim_loss(img, img_warped, window_size=11)
    return 0.1 * l1 + 0.9 * ssim

def _resize_ray_map(ray_map, H, W, B, device):
    ray = torch.from_numpy(ray_map).float().to(device)
    ray = ray.permute(2, 0, 1).unsqueeze(0)
    ray = F.interpolate(ray, size=(H, W), mode='bilinear', align_corners=True)
    ray = ray.squeeze(0).permute(1, 2, 0)
    return ray.unsqueeze(0).repeat(B, 1, 1, 1)

def warp_frame(img_t1, depth_t, T, ray_map):
    B, _, H, W = img_t1.shape
    device = img_t1.device

    ray = _resize_ray_map(ray_map, H, W, B, device)
    d   = depth_t[:, 0]
    rx, ry, rz = ray[...,0], ray[...,1], ray[...,2]
    x3d = (d * rx) / rz
    y3d = (d * ry) / rz
    z3d = d
    ones = torch.ones_like(z3d)
    pts  = torch.stack([x3d, y3d, z3d, ones], dim=-1).reshape(B, -1, 4)

    pts_t1 = torch.bmm(pts, T.transpose(1, 2))
    X = pts_t1[:, :, 0].reshape(B, H, W)
    Y = pts_t1[:, :, 1].reshape(B, H, W)
    Z = pts_t1[:, :, 2].reshape(B, H, W).clamp(min=1e-6)

    # Omnidirectional re-projection (matches c3vd_dataset ray_map)
    # Convert 3D point to angle theta, then to image coordinates
    a0 =  769.243600037458
    a2 = -0.000812770624150226
    a3 =  6.25674244578925e-07
    a4 = -1.19662182144280e-09
    c  =  0.999986882249990
    d  =  0.00288273829525059
    e  = -0.00296316513429569
    cx_o = 678.544839263292
    cy_o = 542.975887548343

    # Scale to 640x512 from original 1350x1080
    scale_x = 640.0 / 1350.0
    scale_y = 512.0 / 1080.0
    cx_s = cx_o * scale_x
    cy_s = cy_o * scale_y

    # theta = angle from optical axis
    r2d  = torch.sqrt(X**2 + Y**2).clamp(min=1e-8)
    theta = torch.atan2(r2d, Z.clamp(min=1e-6))

    # rho = polynomial projection
    rho = (a0 * theta +
           a2 * theta**3 +
           a3 * theta**4 +
           a4 * theta**5)

    # Image coordinates with affine correction
    phi   = torch.atan2(Y, X.clamp(min=1e-8))
    u_raw = rho * torch.cos(phi)
    v_raw = rho * torch.sin(phi)
    u = (c * u_raw + d * v_raw + cx_o) * scale_x
    v = (e * u_raw +     v_raw + cy_o) * scale_y

    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid   = torch.stack([u_norm, v_norm], dim=-1)

    return F.grid_sample(img_t1, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)

def depth_consistency_loss(depth_t, depth_t1, T, ray_map):
    B, _, H, W = depth_t.shape
    device = depth_t.device

    ray = _resize_ray_map(ray_map, H, W, B, device)
    d   = depth_t[:, 0]
    rx, ry, rz = ray[...,0], ray[...,1], ray[...,2]
    x3d = (d * rx) / rz
    y3d = (d * ry) / rz
    z3d = d
    ones = torch.ones_like(z3d)
    pts  = torch.stack([x3d, y3d, z3d, ones], dim=-1).reshape(B, -1, 4)

    pts_t1 = torch.bmm(pts, T.transpose(1, 2))
    z_proj = pts_t1[:, :, 2].reshape(B, 1, H, W)

    valid = (z_proj > 0) & ~torch.isnan(depth_t) & ~torch.isnan(z_proj)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=device)

    diff  = (depth_t - z_proj).abs()
    denom = depth_t + z_proj + 1e-7
    return (diff / denom)[valid].mean()

def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled   = min_disp + (max_disp - min_disp) * disp
    return 1.0 / scaled

def smoothness_loss(depth, image):
    """
    Edge-aware smoothness — depth smooth ở vùng ảnh đồng đều
    depth có thể discontinuous ở edges
    """
    grad_d_x = (depth[:,:,:,:-1] - depth[:,:,:,1:]).abs()
    grad_d_y = (depth[:,:,:-1,:] - depth[:,:,1:,:]).abs()
    grad_i_x = (image[:,:,:,:-1] - image[:,:,:,1:]).abs().mean(1, keepdim=True)
    grad_i_y = (image[:,:,:-1,:] - image[:,:,1:,:]).abs().mean(1, keepdim=True)
    smooth_x  = grad_d_x * torch.exp(-grad_i_x)
    smooth_y  = grad_d_y * torch.exp(-grad_i_y)
    return (smooth_x.mean() + smooth_y.mean()) / 2
