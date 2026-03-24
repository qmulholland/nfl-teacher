import os
import glob
from roboflow import Roboflow

# --- CONFIG ---
# Replace YOUR_PROJECT_ID with the name in your browser URL 
API_KEY = "ZF8A0eDRO2YdYCu0di2k"
WORKSPACE_ID = "quinns-workspace"
PROJECT_ID = "YOUR_PROJECT_ID" 
IMAGE_FOLDER = "/Users/quinnmulholland/Desktop/NFL_AI/Defense"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

# Find all PNGs in your defense folders
image_files = glob.glob(os.path.join(IMAGE_FOLDER, "**/*.png"), recursive=True)

print(f"Found {len(image_files)} images. Preparing upload to {PROJECT_ID}...")

for img_path in image_files:
    # Captures the folder name as the tag
    folder_tag = img_path.split(os.sep)[-2]
    
    print(f"Uploading: {os.path.basename(img_path)} | Tag: {folder_tag}")
    
    project.upload(
        image_path=img_path,
        tag_names=[folder_tag],
        split="train"
    )

print("\nAll snaps are now in your Roboflow project.")