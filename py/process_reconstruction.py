import json
import numpy as np
import open3d as o3d
import os
import csv
from shapely.geometry import MultiPoint, Point

def process_reconstruction_v12(json_path):
    # 1. Setup Directories
    json_dir = os.path.dirname(os.path.abspath(json_path))
    output_dir = os.path.join(json_dir, "house_analysis_v12")
    debug_dir = os.path.join(output_dir, "individual_houses")
    for d in [output_dir, debug_dir]:
        if not os.path.exists(d): os.makedirs(d)
    
    # Load and Level Data
    with open(json_path, 'r') as f:
        data = json.load(f)
    pts = np.array([v['coordinates'] for v in data[0]['points'].values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Statistical Outlier Removal (Cleans floating noise)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    
    # Leveling to Ground Zero (Z=0)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=2000)
    [a, b, c, d_val] = plane_model
    original_offset_xy = np.mean(pts[inliers, :2], axis=0)
    pcd.translate((-original_offset_xy[0], -original_offset_xy[1], 0))
    target_norm = np.array([0, 0, 1])
    plane_norm = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    v = np.cross(plane_norm, target_norm); s = np.linalg.norm(v); c_val = np.dot(plane_norm, target_norm)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2))
    pcd.rotate(rotation, center=(0,0,0))
    pcd.translate((0, 0, -np.median(np.asarray(pcd.points)[inliers, 2])))
    if np.median(np.asarray(pcd.points)[:, 2]) < 0:
        pcd.rotate(np.array([[1,0,0],[0,1,0],[0,0,-1]]), center=(0,0,0))

    # 2. Adaptive Segmentation & Filtering
    obj_idx = np.where(np.asarray(pcd.points)[:, 2] > 0.5)[0] # Everything > 0.5m above ground
    objects_pcd = pcd.select_by_index(obj_idx)
    
    # Primary clustering: find the general islands
    labels = np.array(objects_pcd.cluster_dbscan(eps=3.5, min_points=40))
    house_list = []
    
    if labels.size > 0 and labels.max() >= 0:
        for i in range(labels.max() + 1):
            cluster_pcd = objects_pcd.select_by_index(np.where(labels == i)[0])
            c_pts = np.asarray(cluster_pcd.points)
            
            # PROBLEM 1: Merged Houses (If Area is massive, re-cluster tightly)
            hull = MultiPoint(c_pts[:, :2]).convex_hull
            if hull.area > 350: # If cluster is too large for one house
                sub_labels = np.array(cluster_pcd.cluster_dbscan(eps=1.5, min_points=20))
                sub_clusters = [cluster_pcd.select_by_index(np.where(sub_labels == k)[0]) 
                                for k in range(sub_labels.max() + 1)]
            else:
                sub_clusters = [cluster_pcd]

            for sc_pcd in sub_clusters:
                sc_pts = np.asarray(sc_pcd.points)
                if len(sc_pts) < 10: continue
                
                # PROBLEM 2 & 3: Cars/Bushes/Fences (Filter by Height Profile)
                sc_hull = MultiPoint(sc_pts[:, :2]).convex_hull
                area = sc_hull.area
                max_h = sc_pts[:, 2].max()
                avg_h = sc_pts[:, 2].mean()
                
                # GEOMETRIC RULES:
                # - Houses must be > 2.2m tall (excludes most cars/fences)
                # - Houses must have a footprint > 35m2 (excludes cars/bushes)
                # - Houses usually have a gap between min and max height > 1.5m
                if area > 35 and max_h > 2.5 and (max_h - sc_pts[:, 2].min()) > 1.5:
                    
                    # ID Generation
                    local_centroid = np.mean(sc_pts[:, :2], axis=0)
                    global_x = local_centroid[0] + original_offset_xy[0]
                    global_y = local_centroid[1] + original_offset_xy[1]
                    unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
                    
                    # Generate Synthetic Floor
                    min_x, min_y, max_x, max_y = sc_hull.bounds
                    grid_points = [[x, y, 0.0] for x in np.arange(min_x, max_x, 0.6) 
                                   for y in np.arange(min_y, max_y, 0.6) if sc_hull.contains(Point(x, y))]
                    floor_pcd = o3d.geometry.PointCloud()
                    if grid_points:
                        floor_pcd.points = o3d.utility.Vector3dVector(np.array(grid_points))
                        floor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                    
                    o3d.io.write_point_cloud(os.path.join(debug_dir, f"{unique_id}.ply"), sc_pcd + floor_pcd)
                    house_list.append({"house_ID": unique_id, "Area_m2": round(area, 2), "volume_m3": round(area * avg_h, 2)})

    # Save Results
    csv_path = os.path.join(output_dir, "house_measurements.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_m2", "volume_m3"])
        writer.writeheader()
        writer.writerows(house_list)
    
    print(f"Refinement complete. Identified {len(house_list)} unique buildings.")

process_reconstruction_v12('data/ElmA60H90-24/reconstruction.json')