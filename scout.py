import os
import cv2
import glob
import time
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=API_KEY
)


IMAGE_FOLDER = "/Users/quinnmulholland/Desktop/NFL_AI/Defense"
OUTPUT_FOLDER = "/Users/quinnmulholland/Desktop/NFL_AI/Scouting_Results"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def clean(text):
    return str(text).lower().replace("_", "").replace(" ", "").strip()

print("Initializing Scouting Audit...")
time.sleep(2)

stats = {}
image_files = glob.glob(os.path.join(IMAGE_FOLDER, "**/*.png"), recursive=True)

print(f"Analyzing {len(image_files)} snaps. Press Ctrl+C to stop.\n")

for i, img_path in enumerate(image_files):
    path_parts = img_path.split(os.sep)
    actual_defense = path_parts[-2] 
    filename = path_parts[-1]

    if actual_defense not in stats:
        stats[actual_defense] = {'total': 0, 'correct': 0}
    stats[actual_defense]['total'] += 1

    try:
        # Force lower confidence to ensure we get some detections
        result = client.run_workflow(
            workspace_name="quinns-workspace",
            workflow_id="custom-workflow",
            images={"image": img_path},
            parameters={"confidence": 0.2}
        )
        data = result[0]
        
        # Pull the specific output you mapped in the Roboflow UI
        predicted_defense = data.get('defense_type', "Unknown")
        
        # Handle cases where AI returns a dictionary of data instead of a string
        if isinstance(predicted_defense, dict):
            predicted_defense = "Detection Error"

        # Match Logic
        is_correct = clean(predicted_defense) == clean(actual_defense)
        if is_correct:
            stats[actual_defense]['correct'] += 1
        
        status = "✅" if is_correct else "❌"
        print(f"[{i+1}/{len(image_files)}] {status} Folder: {actual_defense} | AI: {predicted_defense}")

        # Visual Save for your Portfolio
        raw_video = data.get('final_video')
        if raw_video and hasattr(raw_video, 'numpy_image'):
            annotated_img = raw_video.numpy_image
        else:
            annotated_img = cv2.imread(img_path)
            
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(annotated_img, f"AI PREDICT: {predicted_defense}", (20, 100), 1, 1.5, color, 2)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"scout_{filename}"), annotated_img)

    except Exception as e:
        print(f"Error on {filename}: {e}")

# --- FINAL SUMMARY REPORT ---
print("\n" + "="*60)
print(f"{'DEFENSIVE FORMATION':<30} | {'SNAPS':<6} | {'ACCURACY':<10}")
print("-" * 60)
for formation, d in sorted(stats.items()):
    acc = (d['correct'] / d['total']) * 100 if d['total'] > 0 else 0
    print(f"{formation:<30} | {d['total']:<6} | {acc:>8.1f}%")
print("="*60)