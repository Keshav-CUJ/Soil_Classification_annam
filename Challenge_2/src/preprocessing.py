import cv2
import os
from pathlib import Path

def apply_clahe(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the channels and convert back to BGR
    merged = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_img

def apply_clahe_to_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png','.webp','.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping invalid image: {filename}")
                continue

            enhanced_img = apply_clahe(img)
            cv2.imwrite(output_path, enhanced_img)

# Example usage:

apply_clahe_to_folder("../data/data/train", "../data/data/clahe_soil_images")


import shutil
import os

# === Paths ===
source_folder = "../data/data/train"
destination_folder = "../data/data/clahe_soil_images"

# Make sure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# === List of image filenames to copy ===
images_to_copy = ['img_ed9ba5bd.gif', 'img_923fc79a.gif']  # Replace with actual filenames

# === Copy Loop ===
for image_name in images_to_copy:
    src_path = os.path.join(source_folder, image_name)
    dst_path = os.path.join(destination_folder, image_name)
    shutil.copy(src_path, dst_path)

print("Images copied successfully.")



