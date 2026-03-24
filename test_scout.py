import os
import cv2
import glob
import random
import time
from inference_sdk import InferenceHTTPClient

# --- SETUP ---
client = InferenceHTTPClient(
    api_url="http://localhost:9001", 
    api_key="ZF8A0eDRO2YdYCu0di2k"
)
MODEL_ID = "defense_classification_dots/1"
IMAGE_DIR = "/Users/quinnmulholland/Desktop/NFL_AI/Defense"

all_images = glob.glob(os.path.join(IMAGE_DIR, "**/*.png"), recursive=True)

total = 0
correct = 0

print("Starting Live Dashboard... Press 'q' in the window to quit.")

while True:
    test_image_path = random.choice(all_images)
    actual_label = os.path.basename(os.path.dirname(test_image_path))
    
    # 1. Inference
    start_time = time.perf_counter()
    result = client.infer(test_image_path, model_id=MODEL_ID)
    latency = (time.perf_counter() - start_time) * 1000
    
    predictions = result.get('predictions', [])
    if not predictions: continue
    
    pred = predictions[0]
    top_guess = pred.get('class') or pred.get('class_name')
    conf = pred.get('confidence', 0) * 100
    
    # 2. Score Tracking
    total += 1
    is_correct = top_guess.lower().replace("_", " ") == actual_label.lower().replace("_", " ")
    if is_correct:
        correct += 1
    
    accuracy = (correct / total) * 100

    # 3. Visual Rendering (OpenCV)
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (800, 800)) # Larger for the dashboard
    
    # Color coding: Green for correct, Red for wrong
    status_color = (0, 200, 0) if is_correct else (0, 0, 255) # BGR
    
    # Draw Background Header
    cv2.rectangle(img, (0, 0), (800, 100), (40, 40, 40), -1)
    
    # Text Overlays
    cv2.putText(img, f"AI GUESS: {top_guess}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
    cv2.putText(img, f"ACTUAL:   {actual_label}", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Accuracy Stats Box
    cv2.rectangle(img, (550, 10), (790, 90), (60, 60, 60), -1)
    cv2.putText(img, f"ACCURACY: {accuracy:.1f}%", (560, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"SCORE: {correct}/{total}", (560, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("NFL Defensive Scout", img)
    
    # Pause for 1 second per image so you can read it
    if cv2.waitKey(1200) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()