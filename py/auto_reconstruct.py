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

python py/auto_reconstruct.py data/WillowA60H90-20
"""

def run_command(command, project_path):
    """Executes OpenSfM commands with an explicit PYTHONPATH to avoid ImportErrors."""
    # Add the current OpenSfM directory to the environment path
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"Error executing {' '.join(command)}")
        sys.exit(1)

def organize_folders(project_path):
    """Creates 'images' folder and moves JPGs into it."""
    images_dir = os.path.join(project_path, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    image_files = glob.glob(os.path.join(project_path, '*.[jJ][pP][gG]'))
    for img in image_files:
        shutil.move(img, os.path.join(images_dir, os.path.basename(img)))
    print(f"Moved {len(image_files)} images into {images_dir}")

def inject_mrk_data(project_path, mrk_file):
    """Parses MRK and updates OpenSfM EXIF JSONs with RTK precision."""
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
                    "lat": float(lat_match.group(1)), "lon": float(lon_match.group(1)), "alt": float(alt_match.group(1))
                }
    
    json_files = sorted(glob.glob(os.path.join(exif_path, "*.json")))
    for i, json_file in enumerate(json_files, start=1):
        if str(i) in mrk_data:
            with open(json_file, 'r') as f: data = json.load(f)
            data['gps'].update({'latitude': mrk_data[str(i)]['lat'], 
                                'longitude': mrk_data[str(i)]['lon'], 
                                'altitude': mrk_data[str(i)]['alt'], 
                                'dop': 0.01})
            with open(json_file, 'w') as f: json.dump(data, f, indent=4)
    print(f"Successfully injected RTK data into {len(json_files)} files.")

def main(project_path):
    project_path = os.path.abspath(project_path)
    organize_folders(project_path)
    mrk_file = glob.glob(os.path.join(project_path, "*.MRK"))[0]
    
    opensfm_bin = "bin/opensfm"
    steps = ["extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct"]
    
    for step in steps:
        if step == "detect_features": # Inject MRK after metadata is generated but before features
            inject_mrk_data(project_path, mrk_file)
        run_command([opensfm_bin, step, project_path], project_path)

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python auto_reconstruct.py <path>")
    else: main(sys.argv[1])