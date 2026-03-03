import json
import numpy as np
import open3d as o3d
import os
import csv
from shapely.geometry import MultiPoint, Point

def process_reconstruction_v11(json_path):
    """
    Processes OpenSfM reconstruction.json to identify houses, calculate area/volume,
    and generate a synthetic floor for visualization. 
    Units are moved to CSV column headers.
    """
    # 1. Setup Directories
    json_dir = os.path.dirname(os.path.abspath(json_path))
    output_dir = os.path.join(json_dir, "house_analysis_v11")
    debug_dir = os.path.join(output_dir, "individual_houses")
    
    for d in [output_dir, debug_dir]:
        if not os.path.exists(d): 
            os.makedirs(d)
    
    # Load Data from OpenSfM
    with open(json_path, 'r') as f:
        data = json.load(f)
    pts = np.array([v['coordinates'] for v in data[0]['points'].values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 2. Outlier Removal & Leveling (Ensures accurate Z=0 ground)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=2000)
    [a, b, c, d_val] = plane_model
    
    # Global Centering to (0,0,0) to avoid large-coordinate jitter
    original_offset_xy = np.mean(pts[inliers, :2], axis=0)
    pcd.translate((-original_offset_xy[0], -original_offset_xy[1], 0))

    # Leveling math: align detected ground plane to world Up [0,0,1]
    target_norm = np.array([0, 0, 1])
    plane_norm = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    v = np.cross(plane_norm, target_norm)
    s = np.linalg.norm(v)
    c_val = np.dot(plane_norm, target_norm)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2))
    pcd.rotate(rotation, center=(0,0,0))

    # Normalize Z so median ground point is exactly 0.0
    ground_z = np.median(np.asarray(pcd.points)[inliers, 2])
    pcd.translate((0, 0, -ground_z))
    
    # Flip Z if the model is inverted (common with some DJI orientations)
    if np.median(np.asarray(pcd.points)[:, 2]) < 0:
        pcd.rotate(np.array([[1,0,0],[0,1,0],[0,0,-1]]), center=(0,0,0))

    # 3. Clustering (Identify individual buildings)
    obj_idx = np.where(np.asarray(pcd.points)[:, 2] > 0.5)[0]
    objects_pcd = pcd.select_by_index(obj_idx)
    # eps=3.5 connects sparse roof segments; min_points=40 filters out small noise
    labels = np.array(objects_pcd.cluster_dbscan(eps=3.5, min_points=40))

    house_list = []
    
    if labels.size > 0 and labels.max() >= 0:
        for i in range(labels.max() + 1):
            cluster_pcd = objects_pcd.select_by_index(np.where(labels == i)[0])
            c_pts = np.asarray(cluster_pcd.points)
            
            # Unique ID Generation based on Global Lat/Long Fingerprint
            local_centroid = np.mean(c_pts[:, :2], axis=0)
            global_x = local_centroid[0] + original_offset_xy[0]
            global_y = local_centroid[1] + original_offset_xy[1]
            unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
            
            # Area and Prism Volume Math
            points_2d = c_pts[:, :2]
            if len(points_2d) < 3: continue
            hull = MultiPoint(points_2d).convex_hull
            area = hull.area
            avg_height = np.mean(c_pts[:, 2])
            volume = area * avg_height

            # Filters for realistic residential buildings
            if area > 30 and avg_height > 2.5:
                # Generate Synthetic Floor at Z=0 for visualization
                min_x, min_y, max_x, max_y = hull.bounds
                grid_points = []
                for x in np.arange(min_x, max_x, 0.5):
                    for y in np.arange(min_y, max_y, 0.5):
                        if hull.contains(Point(x, y)):
                            grid_points.append([x, y, 0.0])
                
                floor_pcd = o3d.geometry.PointCloud()
                if grid_points:
                    floor_pcd.points = o3d.utility.Vector3dVector(np.array(grid_points))
                    floor_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray floor
                
                # Combine and Save House Model
                combined_house = cluster_pcd + floor_pcd
                o3d.io.write_point_cloud(os.path.join(debug_dir, f"{unique_id}.ply"), combined_house)
                
                # Store data (numeric only for cells)
                house_list.append({
                    "house_ID": unique_id,
                    "Area_m2": round(area, 2),
                    "volume_m3": round(volume, 2)
                })

    # 4. Save CSV with Units in Headers
    csv_path = os.path.join(output_dir, "house_measurements.csv")
    headers = ["house_ID", "Area_m2", "volume_m3"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(house_list)
    
    print(f"Success! Processed {len(house_list)} unique houses.")
    print(f"CSV saved with unit headers to: {csv_path}")

# Run the process
process_reconstruction_v11('data/ElmA60H90/reconstruction.json')