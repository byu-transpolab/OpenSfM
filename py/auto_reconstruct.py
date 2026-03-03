import os
import subprocess
import json
import glob
import re
import sys
import shutil

"""
Run using the following

python py/auto_reconstruct.py data/foldername

python py/auto_reconstruct.py data/ElmA60H90
"""

def run_command(command):
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {' '.join(command)}")
        sys.exit(1)

def organize_folders(project_path):
    """Creates 'images' folder and moves JPGs into it."""
    images_dir = os.path.join(project_path, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created folder: {images_dir}")

    # Move all JPG/jpg files from project root to images/
    image_files = glob.glob(os.path.join(project_path, '*.[jJ][pP][gG]'))
    for img in image_files:
        shutil.move(img, os.path.join(images_dir, os.path.basename(img)))
    
    print(f"Moved {len(image_files)} images into {images_dir}")

def inject_mrk_data(project_path, mrk_file):
    """Parses MRK and updates OpenSfM EXIF JSONs."""
    print(f"Injecting RTK data from {mrk_file}...")
    exif_path = os.path.join(project_path, 'exif')
    
    mrk_data = {}
    with open(mrk_file, 'r') as f:
        for line in f:
            lat_match = re.search(r"([-+]?\d*\.\d+|\d+),Lat", line)
            lon_match = re.search(r"([-+]?\d*\.\d+|\d+),Lon", line)
            alt_match = re.search(r"([-+]?\d*\.\d+|\d+),Ellh", line)
            idx_match = re.match(r"^(\d+)", line.strip())
            
            if lat_match and lon_match and alt_match and idx_match:
                mrk_data[idx_match.group(1)] = {
                    "lat": float(lat_match.group(1)),
                    "lon": float(lon_match.group(1)),
                    "alt": float(alt_match.group(1))
                }

    json_files = sorted(glob.glob(os.path.join(exif_path, "*.json")))
    for i, json_file in enumerate(json_files, start=1):
        idx_str = str(i)
        if idx_str in mrk_data:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            data['gps']['latitude'] = mrk_data[idx_str]['lat']
            data['gps']['longitude'] = mrk_data[idx_str]['lon']
            data['gps']['altitude'] = mrk_data[idx_str]['alt']
            data['gps']['dop'] = 0.01  # RTK Precision
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
    print(f"Successfully updated {len(json_files)} metadata files.")

def main(project_path):
    # Ensure the path is absolute for safety
    project_path = os.path.abspath(project_path)
    
    # 1. Organize images
    organize_folders(project_path)

    # 2. Find MRK
    mrk_files = glob.glob(os.path.join(project_path, "*.MRK"))
    if not mrk_files:
        print("Error: No .MRK file found in the project directory!")
        return
    mrk_file = mrk_files[0]

    # 3. OpenSfM Pipeline
    opensfm_bin = "bin/opensfm"
    
    run_command([opensfm_bin, "extract_metadata", project_path])
    inject_mrk_data(project_path, mrk_file)
    run_command([opensfm_bin, "detect_features", project_path])
    run_command([opensfm_bin, "match_features", project_path])
    run_command([opensfm_bin, "create_tracks", project_path])
    run_command([opensfm_bin, "reconstruct", project_path])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_reconstruct.py <project_path>")
    else:
        main(sys.argv[1])