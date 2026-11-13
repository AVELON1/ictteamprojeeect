import os
import json
import numpy as np
import tensorflow as tf
from src.data_loader import load_classes, load_dataset, create_augmented_dataset
from src.model import build_traffic_sign_model

def train():
    os.makedirs('models', exist_ok=True)
    classes_file = 'dataset/classes.json'
    train_dir = 'dataset/train_clean'
    test_dir = 'dataset/test_clean'
    
    codes, code_to_idx, idx_to_name = load_classes(classes_file)
    print("Loading clean training images...")
    X_train, y_train = load_dataset(train_dir, code_to_idx)
    X_test, y_test = load_dataset(test_dir, code_to_idx)
    
    print(f"Original clean train samples: {len(X_train)}")
    print("Generating synthetic OpenCV augmentations (blur, rotate ±15 deg, lighting changes)...")
    X_train_aug, y_train_aug = create_augmented_dataset(X_train, y_train, copies_per_image=3)
    print(f"Total augmented train samples: {len(X_train_aug)}")
    
    X_train_norm = X_train_aug.astype(np.float32) / 255.0
    X_test_norm = X_test.astype(np.float32) / 255.0
    
    model = build_traffic_sign_model(num_classes=len(codes))
    print("Training Model 2 (Option B - Enriched with OpenCV Augmentation)...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    history = model.fit(
        X_train_norm, y_train_aug,
        validation_data=(X_test_norm, y_test),
        epochs=12,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
    print(f"\n>>> Model 2 (Augmented Robust) Test Accuracy on clean set: {test_acc * 100:.2f}%")
    
    save_path = 'models/model_aug.keras'
    model.save(save_path)
    print(f"Saved robust model to {save_path}")

if __name__ == '__main__':
    train()
