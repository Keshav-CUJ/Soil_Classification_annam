import os
from PIL import Image

def resize_with_padding(img, size=224, fill_color=(0, 0, 0)):
    old_size = img.size  # (width, height)
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", (size, size), fill_color)
    new_img.paste(img, ((size - new_size[0]) // 2,
                        (size - new_size[1]) // 2))
    return new_img

def process_folder(input_folder, output_folder, size=224):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png','.webp','.gif')):
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                resized_img = resize_with_padding(img, size=size)
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Example usage
input_folder = "../data/data/train"
output_folder = "../data/data/resized_images_224"
process_folder(input_folder, output_folder)


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

apply_clahe_to_folder("../data/data/resized_images_224", "../data/data/clahe_soil_images")


import shutil
import os

# === Paths ===
source_folder = "../data/data/resized_images_224"
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



import pandas as pd
import os
import cv2
from pathlib import Path
import shutil  # for copying files

def save_images_to_label_folders(csv_path, image_folder, output_folder):
    df = pd.read_csv(csv_path)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Create a folder for each label
    labels = df['soil_type'].unique()
    for label in labels:
        Path(os.path.join(output_folder, label)).mkdir(exist_ok=True)

    for _, row in df.iterrows():
        img_name = row['image_id']
        label = row['soil_type']

        src_path = os.path.join(image_folder, img_name)
        dst_path = os.path.join(output_folder, label, img_name)

        if os.path.exists(src_path):
            # Option 1: Copy original file
            shutil.copy(src_path, dst_path)

            # Option 2: If you want to save a processed image (e.g., CLAHE applied), you would read/process and cv2.imwrite here instead.

        else:
            print(f"Warning: {src_path} does not exist.")

# Example usage:
save_images_to_label_folders("../data/data/train_labels.csv", "../data/data/clahe_soil_images", "../data/data/labeled_soil_images")


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np
from tqdm import tqdm

# ===== Paths =====
input_root = "../data/data/labeled_soil_images" # original dataset path (class subfolders)
output_root = "../data/data/Aug_for_train"  # augmented image output path
os.makedirs(output_root, exist_ok=True)

# ===== Data Augmentation Pipeline =====
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1),
    
])

# ===== Augment Per Class Folder =====
for class_name in os.listdir(input_root):
    class_input_path = os.path.join(input_root, class_name)
    class_output_path = os.path.join(output_root, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    images = [f for f in os.listdir(class_input_path) if f.endswith(('.jpg', '.png', 'webm','.gif','jpeg'))]
    print(f"\n🔹 Class: {class_name}")
    print(f"Original images: {len(images)}")
    count = 0

    for img_file in tqdm(images):
        img_path = os.path.join(class_input_path, img_file)
        img = load_img(img_path, target_size=(224, 224))  # Resize here
        img_arr = img_to_array(img) / 255.0  # Normalize to [0,1]
        img_tensor = tf.expand_dims(img_arr, axis=0)  # Shape: (1, H, W, C)

        # Save original (optional)
        array_to_img(img_arr).save(os.path.join(class_output_path, f"orig_{img_file}"))

        # Save 3 augmentations per image
        for i in range(1):
            aug_img = data_augmentation(img_tensor, training=True)[0].numpy()
            array_to_img(aug_img).save(os.path.join(class_output_path, f"aug_{i}_{img_file}"))
            count += 1

    print(f"Total images after augmentation: {count + len(images)}")
