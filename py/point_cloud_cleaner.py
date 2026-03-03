import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_tool

def process_isler_advanced(input_json, 
                           use_scaling=True, 
                           scaling_method="manual_height", 
                           ref_val=5.5, 
                           pixel_gsd=None):
    """
    scaling_method options: 
    - "manual_height": ref_val is the height in cm.
    - "pixel_width": ref_val is unused, pixel_gsd is cm per pixel.
    """
    if not os.path.exists(input_json):
        print(f"❌ File not found: {input_json}")
        return
        
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    reconstruction = data[0]
    points = reconstruction['points']
    shots = reconstruction['shots']
    camera_data = reconstruction['cameras']

    # --- PART 1: TRUE COORDINATE ALIGNMENT ---
    true_cam_locs = []
    for shot_id, shot_data in shots.items():
        rot_vec = np.array(shot_data['rotation'])
        t_vec = np.array(shot_data['translation'])
        r_mat = R_tool.from_rotvec(rot_vec).as_matrix()
        true_cam_locs.append(-r_mat.T @ t_vec)
    
    model_focus = np.mean(true_cam_locs, axis=0)
    pt_coords = np.array([p['coordinates'] for p in points.values()])
    pt_colors = np.array([p['color'] for p in points.values()])
    
    # Clean noise (using our successful 0.3 cutoff logic)
    distances = np.linalg.norm(pt_coords - model_focus, axis=1)
    cutoff = np.median(distances) + (0.3 * np.std(distances))
    valid_mask = distances < cutoff
    
    clean_coords = pt_coords[valid_mask]
    clean_colors = pt_colors[valid_mask]

    # --- PART 2: GROUND ISOLATION ---
    z_vals = clean_coords[:, 2]
    # Find the desk surface (the largest peak in the bottom of the model)
    hist, bin_edges = np.histogram(z_vals, bins=100)
    ground_bin_idx = np.argmax(hist[:30]) 
    ground_z = bin_edges[ground_bin_idx]
    
    # House points are everything significantly above the desk
    house_mask = z_vals > (ground_z + 0.05) 
    house_coords = clean_coords[house_mask]
    house_colors = clean_colors[house_mask]

    # --- PART 3: SCALING LOGIC ---
    scale_factor = 1.0
    if use_scaling:
        dims_units = np.max(house_coords, axis=0) - np.min(house_coords, axis=0)
        
        if scaling_method == "manual_height":
            # Scale based on the 5.5cm house height
            scale_factor = ref_val / dims_units[2]
            
        elif scaling_method == "pixel_width" and pixel_gsd:
            # Scale based on Pixel Width (GSD). 
            # We use the camera focal length to relate pixels to world units.
            cam_id = list(camera_data.keys())[0]
            focal_pixels = camera_data[cam_id]['focal'] * camera_data[cam_id]['width']
            # Scale Factor = (GSD * Focal) / Avg_Depth_from_Cameras
            avg_depth = np.mean(distances[valid_mask])
            scale_factor = (pixel_gsd * focal_pixels) / avg_depth

    # --- PART 4: BOUNDING BOX & EXPORT ---
    h_min = np.min(house_coords, axis=0)
    h_max = np.max(house_coords, axis=0)
    scaled_dims = (h_max - h_min) * scale_factor

    # Save House PLY
    export_ply(house_coords, house_colors, input_json.replace('reconstruction.json', 'house_only.ply'))
    # Save Bounding Box PLY (for visual check)
    export_box(h_min, h_max, input_json.replace('reconstruction.json', 'house_bbox.ply'))

    print("="*40)
    print(f"📏 ISLERS ANALYSIS: {scaling_method.upper()}")
    print("="*40)
    print(f"Scale Factor: {scale_factor:.4f} cm/unit")
    print(f"Footprint:    {scaled_dims[0]:.2f} x {scaled_dims[1]:.2f} cm")
    print(f"True Height:  {scaled_dims[2]:.2f} cm")
    print(f"Total Volume: {scaled_dims[0]*scaled_dims[1]*scaled_dims[2]:.2f} cm³")
    print("="*40)

def export_ply(coords, colors, path):
    header = ["ply", "format ascii 1.0", f"element vertex {len(coords)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, 'w') as f:
        f.write("\n".join(header) + "\n")
        for i in range(len(coords)):
            f.write(f"{coords[i,0]} {coords[i,1]} {coords[i,2]} {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")

def export_box(min_pt, max_pt, path):
    # Creates 8 corners of the bounding box
    c = [ [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
          [max_pt[0], max_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
          [min_pt[0], min_pt[1], max_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
          [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]] ]
    header = ["ply", "format ascii 1.0", "element vertex 8", "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, 'w') as f:
        f.write("\n".join(header) + "\n")
        for corner in c: f.write(f"{corner[0]} {corner[1]} {corner[2]} 255 0 0\n")

# RUN: Change scaling_method to "pixel_width" and provide pixel_gsd if preferred
process_isler_advanced('/Users/willicon/Desktop/OpenSfM/data/phc_8_across/reconstruction.json', 
                       use_scaling=True, 
                       scaling_method="manual_height", 
                       ref_val=5.5)
'''
import json
import numpy as np
import os

def analyze_model_structure(input_json):
    if not os.path.exists(input_json):
        print(f"❌ File not found at: {input_json}")
        return

    with open(input_json, 'r') as f:
        data = json.load(f)
    
    reconstruction = data[0]
    points = reconstruction['points']
    shots = reconstruction['shots']

    # 1. Point Cloud Stats
    pt_coords = np.array([p['coordinates'] for p in points.values()])
    pt_mean = np.mean(pt_coords, axis=0)
    pt_std = np.std(pt_coords, axis=0)
    pt_min = np.min(pt_coords, axis=0)
    pt_max = np.max(pt_coords, axis=0)

    # 2. Camera Stats
    cam_coords = np.array([s['translation'] for s in shots.values()])
    cam_mean = np.mean(cam_coords, axis=0)
    cam_min = np.min(cam_coords, axis=0)
    cam_max = np.max(cam_coords, axis=0)

    # 3. Distance Analysis (Point vs. Camera Center)
    dists_to_cam_center = np.linalg.norm(pt_coords - cam_mean, axis=1)
    
    print("="*40)
    print("📊 OPENSFM MODEL DIAGNOSTIC REPORT")
    print("="*40)
    print(f"Total Points:  {len(points)}")
    print(f"Total Cameras: {len(shots)}")
    print("\n--- COORDINATE RANGE ---")
    print(f"Points Min: {pt_min}")
    print(f"Points Max: {pt_max}")
    print(f"Cameras Min: {cam_min}")
    print(f"Cameras Max: {cam_max}")
    
    print("\n--- DISTANCE FROM CAMERA HUB ---")
    print(f"Mean Dist:   {np.mean(dists_to_cam_center):.4f}")
    print(f"Median Dist: {np.median(dists_to_cam_center):.4f}")
    print(f"Std Dev:     {np.std(dists_to_cam_center):.4f}")
    print(f"90th P-tile: {np.percentile(dists_to_cam_center, 90):.4f}")
    print(f"Max Dist:    {np.max(dists_to_cam_center):.4f}")

    # 4. Find the "Outlier Gap"
    # Let's see if there is a massive jump in distance
    sorted_dists = np.sort(dists_to_cam_center)
    gaps = np.diff(sorted_dists)
    max_gap_idx = np.argmax(gaps)
    print("\n--- OUTLIER GAP ANALYSIS ---")
    print(f"Largest jump in distance: {gaps[max_gap_idx]:.4f}")
    print(f"Occurs at distance:       {sorted_dists[max_gap_idx]:.4f}")
    
    # Save a small sample for me to see
    print("\n[Action] Saving diagnostic_data.csv for coordinate verification...")
    with open('diagnostic_data.csv', 'w') as f:
        f.write("type,x,y,z\n")
        for i, c in enumerate(cam_coords):
            f.write(f"camera_{i},{c[0]},{c[1]},{c[2]}\n")
        # Sample 10 random points
        sample_indices = np.random.choice(len(pt_coords), 10)
        for i, idx in enumerate(sample_indices):
            p = pt_coords[idx]
            f.write(f"point_{i},{p[0]},{p[1]},{p[2]}\n")

# Run it
analyze_model_structure('/Users/willicon/Desktop/OpenSfM/data/phc_8_across/reconstruction.json')
'''