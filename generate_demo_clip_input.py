"""
Generate Synthetic Test Image for Zero-Shot CLIP Object Detector & Tagging Engine
Day 28 - 30-Day Computer Vision Challenge
"""

import os
import cv2
import numpy as np

def create_synthetic_clip_demo(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "demo_clip_test_image.jpg")

    # Create 640x480 RGB image with 4 distinct objects
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 230

    # Background subtle wood grain texture pattern
    for y in range(h):
        for x in range(w):
            val = int(210 + 15 * np.sin(y / 20.0))
            img[y, x] = [val - 20, val - 10, val]

    # Object 1: "vintage red mug" (Top Left ROI: 60,60 -> 220,220)
    cv2.rectangle(img, (70, 70), (210, 210), (30, 30, 200), -1)
    cv2.ellipse(img, (220, 140), (25, 45), 0, 0, 360, (30, 30, 200), 12) # Mug handle
    cv2.putText(img, "MUG", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Object 2: "yellow sports ball" (Top Right ROI: 420,60 -> 580,220)
    cv2.circle(img, (500, 140), 70, (20, 220, 255), -1)
    cv2.circle(img, (500, 140), 70, (0, 150, 200), 4)
    cv2.putText(img, "BALL", (465, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Object 3: "metallic wrench tool" (Bottom Left ROI: 60,280 -> 260,420)
    cv2.rectangle(img, (80, 340), (240, 370), (180, 180, 180), -1)
    cv2.circle(img, (90, 355), 30, (160, 160, 160), -1)
    cv2.circle(img, (90, 355), 15, (230, 230, 230), -1)
    cv2.putText(img, "TOOL", (130, 363), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Object 4: "green plant pot" (Bottom Right ROI: 420,260 -> 580,440)
    cv2.rectangle(img, (450, 330), (550, 430), (50, 100, 180), -1)
    cv2.ellipse(img, (500, 300), (45, 30), 0, 0, 360, (30, 180, 50), -1) # Leaves
    cv2.putText(img, "PLANT", (465, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(img_path, img)
    print(f"[SUCCESS] Synthetic CLIP test image saved: {img_path} ({w}x{h})")
    return img_path

if __name__ == "__main__":
    create_synthetic_clip_demo()
