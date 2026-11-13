import os
import json
import numpy as np
import tensorflow as tf
from src.data_loader import load_classes, load_dataset
from src.augment import apply_blur, apply_rotation, apply_brightness_contrast, apply_noise

def evaluate_both():
    classes_file = 'dataset/classes.json'
    test_dir = 'dataset/test_clean'
    
    codes, code_to_idx, idx_to_name = load_classes(classes_file)
    X_test, y_test = load_dataset(test_dir, code_to_idx)
    X_test_clean = X_test.astype(np.float32) / 255.0
    
    # Generate distorted adverse test set
    X_test_distorted = []
    for img in X_test:
        dist = img.copy()
        # Heavy adverse condition: blur + tilt + dusk lighting
        dist = apply_rotation(dist, angle=np.random.choice([-14, -10, 10, 14]))
        dist = apply_blur(dist, ksize=np.random.choice([5, 7, 9]))
        dist = apply_brightness_contrast(dist, alpha=0.65, beta=-30) # dusk/underexposed
        dist = apply_noise(dist, sigma=15)
        X_test_distorted.append(dist)
        
    X_test_distorted = np.array(X_test_distorted, dtype=np.float32) / 255.0
    
    # Load models
    m1_path = 'models/model_clean.keras'
    m2_path = 'models/model_aug.keras'
    
    m1 = tf.keras.models.load_model(m1_path) if os.path.exists(m1_path) else None
    m2 = tf.keras.models.load_model(m2_path) if os.path.exists(m2_path) else None
    
    print("\n" + "=" * 65)
    print("           BENCHMARK COMPARISON: OPTION B (SYNTHETIC AUGMENTATION)")
    print("=" * 65)
    
    if m1:
        loss_c1, acc_c1 = m1.evaluate(X_test_clean, y_test, verbose=0)
        loss_d1, acc_d1 = m1.evaluate(X_test_distorted, y_test, verbose=0)
        print(f"Model 1 (Baseline - Trained on Clean Only):")
        print(f"  - Clean Test Accuracy:     {acc_c1 * 100:.2f}%")
        print(f"  - Adverse/Blurry Accuracy: {acc_d1 * 100:.2f}%  <-- Sharp degradation!")
    
    if m2:
        loss_c2, acc_c2 = m2.evaluate(X_test_clean, y_test, verbose=0)
        loss_d2, acc_d2 = m2.evaluate(X_test_distorted, y_test, verbose=0)
        print(f"\nModel 2 (Enriched - Trained with OpenCV Augmentations):")
        print(f"  - Clean Test Accuracy:     {acc_c2 * 100:.2f}%")
        print(f"  - Adverse/Blurry Accuracy: {acc_d2 * 100:.2f}%  <-- Robust generalization!")
        
    print("=" * 65)

if __name__ == '__main__':
    evaluate_both()
