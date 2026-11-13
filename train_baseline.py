import os
import json
import numpy as np
import tensorflow as tf
from src.data_loader import load_classes, load_dataset
from src.model import build_traffic_sign_model

def train():
    os.makedirs('models', exist_ok=True)
    classes_file = 'dataset/classes.json'
    train_dir = 'dataset/train_clean'
    test_dir = 'dataset/test_clean'
    
    codes, code_to_idx, idx_to_name = load_classes(classes_file)
    with open('models/classes_map.json', 'w', encoding='utf-8') as f:
        json.dump({'code_to_idx': code_to_idx, 'idx_to_name': idx_to_name}, f, indent=2, ensure_ascii=False)
        
    print(f"Loaded {len(codes)} classes: {codes}")
    print("Loading clean training images...")
    X_train, y_train = load_dataset(train_dir, code_to_idx)
    X_test, y_test = load_dataset(test_dir, code_to_idx)
    print(f"X_train shape: {X_train.shape}, y_train: {len(y_train)}")
    print(f"X_test shape: {X_test.shape}, y_test: {len(y_test)}")
    
    # Normalize to [0, 1]
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_test_norm = X_test.astype(np.float32) / 255.0
    
    model = build_traffic_sign_model(num_classes=len(codes))
    print("Training Model 1 (Baseline - Clean only)...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    history = model.fit(
        X_train_norm, y_train,
        validation_data=(X_test_norm, y_test),
        epochs=12,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
    print(f"\n>>> Model 1 (Baseline Clean) Test Accuracy: {test_acc * 100:.2f}%")
    
    save_path = 'models/model_clean.keras'
    model.save(save_path)
    print(f"Saved baseline model to {save_path}")

if __name__ == '__main__':
    train()
