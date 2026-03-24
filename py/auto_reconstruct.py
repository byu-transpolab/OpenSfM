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

python py/auto_reconstruct.py data/ElmA60H90P22/RGB
bin/opensfm export_ply data/ElmA60H90P22/Green
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
    """Creates 'images' folder and moves JPGs and TIFs into it."""
    images_dir = os.path.join(project_path, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Grab all files in the directory
    all_files = os.listdir(project_path)
    valid_extensions = ('.jpg', '.jpeg', '.tif', '.tiff')
    
    image_files = [
        os.path.join(project_path, f) for f in all_files
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(project_path, f))
    ]

    for img in image_files:
        # Prevent moving errors if the image is somehow already in the images folder
        if os.path.dirname(img) != images_dir:
            shutil.move(img, os.path.join(images_dir, os.path.basename(img)))
            
    print(f"Moved {len(image_files)} images into {images_dir}")

def inject_mrk_data(project_path, mrk_files):
    """Parses MRK files and updates JSONs, using prefixes (F1_, F2_) to prevent sequence collisions."""
    exif_path = os.path.join(project_path, 'exif')
    
    # mrk_data structure: { "Prefix": { Sequence_Number: { lat, lon, alt } } }
    mrk_data = {}
    
    # 1. Parse all MRK Files into prefix-separated dictionaries
    for mrk_file in mrk_files:
        filename = os.path.basename(mrk_file)
        
        # Check if the file has an F1_ or F2_ prefix. If not, assign it to 'DEFAULT'
        prefix_match = re.match(r"^(F\d)_", filename)
        prefix = prefix_match.group(1) if prefix_match else "DEFAULT"
        
        if prefix not in mrk_data:
            mrk_data[prefix] = {}
            
        with open(mrk_file, 'r') as f:
            for line in f:
                lat_match = re.search(r"([-+]?\d*\.\d+|\d+),Lat", line)
                lon_match = re.search(r"([-+]?\d*\.\d+|\d+),Lon", line)
                alt_match = re.search(r"([-+]?\d*\.\d+|\d+),Ellh", line)
                idx_match = re.match(r"^(\d+)", line.strip()) 
                
                if lat_match and lon_match and alt_match and idx_match:
                    seq_num = int(idx_match.group(1))
                    
                    # Store the GPS data under the specific prefix namespace
                    mrk_data[prefix][seq_num] = {
                        "lat": float(lat_match.group(1)), 
                        "lon": float(lon_match.group(1)), 
                        "alt": float(alt_match.group(1))
                    }
    
    # 2. Match GPS data to the correct JSON file based on BOTH prefix and sequence number
    json_files = glob.glob(os.path.join(exif_path, "*.json"))
    injected_count = 0
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        
        # Identify the JSON file's prefix and sequence number
        prefix_match = re.match(r"^(F\d)_", filename)
        prefix = prefix_match.group(1) if prefix_match else "DEFAULT"
        seq_match = re.search(r"_(\d{4})_", filename)
        
        if seq_match:
            seq_num = int(seq_match.group(1)) 
            
            # Ensure both the prefix dictionary exists AND the sequence number is in it
            if prefix in mrk_data and seq_num in mrk_data[prefix]:
                with open(json_file, 'r') as f: 
                    data = json.load(f)
                
                if 'gps' not in data:
                    data['gps'] = {}
                    
                data['gps'].update({
                    'latitude': mrk_data[prefix][seq_num]['lat'], 
                    'longitude': mrk_data[prefix][seq_num]['lon'], 
                    'altitude': mrk_data[prefix][seq_num]['alt'], 
                    'dop': 0.01 
                })
                
                with open(json_file, 'w') as f: 
                    json.dump(data, f, indent=4)
                
                injected_count += 1
                
    print(f"Successfully injected precise RTK data into {injected_count} out of {len(json_files)} metadata files.")
    
def main(project_path):
    project_path = os.path.abspath(project_path)
    organize_folders(project_path)
    
    # Safely check for the MRK file instead of hard-indexing [0]
    mrk_files = glob.glob(os.path.join(project_path, "*.MRK"))
    mrk_file = mrk_files[0] if mrk_files else None
    
    opensfm_bin = "bin/opensfm"
    steps = ["extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct"]
    
    for step in steps:
        if step == "detect_features": # Inject MRK after metadata is generated but before features
            if mrk_file:
                inject_mrk_data(project_path, mrk_file)
            else:
                print("\n--- NOTICE: No .MRK file found. Relying on default image EXIF data for GPS coordinates. ---\n")
        run_command([opensfm_bin, step, project_path], project_path)

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python auto_reconstruct.py <path>")
    else: main(sys.argv[1])