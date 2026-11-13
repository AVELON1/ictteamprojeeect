import os
import json
import cv2
import numpy as np
from src.augment import generate_synthetic_variants

def load_classes(classes_path):
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    sorted_codes = sorted(list(classes.keys()))
    code_to_idx = {code: i for i, code in enumerate(sorted_codes)}
    idx_to_name = {i: f"{code} ({classes[code]})" for i, code in enumerate(sorted_codes)}
    return sorted_codes, code_to_idx, idx_to_name

def load_dataset(data_dir, code_to_idx, img_size=(64, 64)):
    """Loads images and labels from a structured directory (dir/<class_code>/<image.png>)."""
    images = []
    labels = []
    
    for code, idx in code_to_idx.items():
        class_folder = os.path.join(data_dir, code)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(class_folder, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                # OpenCV loads BGR, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(idx)
                
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

def create_augmented_dataset(images, labels, copies_per_image=3):
    """Expands dataset by generating OpenCV synthetic variations."""
    aug_images = list(images)
    aug_labels = list(labels)
    
    for img, lbl in zip(images, labels):
        variants = generate_synthetic_variants(img, num_variants=copies_per_image)
        for var in variants:
            aug_images.append(var)
            aug_labels.append(lbl)
            
    # Shuffle
    indices = np.random.permutation(len(aug_images))
    aug_images = np.array(aug_images, dtype=np.uint8)[indices]
    aug_labels = np.array(aug_labels, dtype=np.int32)[indices]
    return aug_images, aug_labels
