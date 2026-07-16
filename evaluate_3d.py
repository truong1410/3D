import open3d as o3d
import numpy as np
import os

DATA_DIR = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/cecum_t1_a/'
PRED_PLY = '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/output/colon_3d.ply'
GT_OBJ   = os.path.join(DATA_DIR, 'coverage_mesh.obj')

# Load predicted mesh
pred_mesh = o3d.io.read_triangle_mesh(PRED_PLY)
pred_pcd  = pred_mesh.sample_points_uniformly(number_of_points=10000)

# Load GT mesh
gt_mesh = o3d.io.read_triangle_mesh(GT_OBJ)
gt_pcd  = gt_mesh.sample_points_uniformly(number_of_points=10000)

# Tính cloud-to-cloud distance
dists = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
dists = dists[~np.isnan(dists)]

print("=== Reconstruction Metrics ===")
print(f"Mean distance:  {dists.mean():.4f} mm")
print(f"Std  distance:  {dists.std():.4f} mm")
print(f"Median:         {np.median(dists):.4f} mm")
