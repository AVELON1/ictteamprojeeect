import os
import json
import numpy as np
import tensorflow as tf
from src.data_loader import load_classes, load_dataset
from src.model import build_traffic_sign_model

def train_strict_clean():
    """
    Trains Model 1 strictly on a small clean subset (15-20 samples per class)
    WITHOUT any normalization tricks or noise tolerance.
    This creates the classic realistic baseline: high accuracy on pristine images (95%+),
    but fragile and easily confused when blur/rotation is applied!
    """
    classes_file = 'dataset/classes.json'
    train_dir = 'dataset/train_clean'
    test_dir = 'dataset/test_clean'
    
    codes, code_to_idx, idx_to_name = load_classes(classes_file)
    X_train_full, y_train_full = load_dataset(train_dir, code_to_idx)
    X_test, y_test = load_dataset(test_dir, code_to_idx)
    
    # Take only clean samples without augmentation
    # Subsample to 25 samples per class so it does not overfit to general invariance
    sub_X = []
    sub_y = []
    for c_idx in range(len(codes)):
        idxs = np.where(y_train_full == c_idx)[0][:25]
        sub_X.append(X_train_full[idxs])
        sub_y.append(y_train_full[idxs])
    X_train_sub = np.concatenate(sub_X, axis=0)
    y_train_sub = np.concatenate(sub_y, axis=0)
    
    print(f"Model 1 (Pure Pristine): {len(X_train_sub)} clean samples.")
    
    X_train_norm = X_train_sub.astype(np.float32) / 255.0
    X_test_norm = X_test.astype(np.float32) / 255.0
    
    # Lean CNN without aggressive dropout to simulate standard basic model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(codes), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_norm, y_train_sub, epochs=15, batch_size=16, verbose=1)
    
    model.save('models/model_clean.keras')
    print("Saved true baseline model to models/model_clean.keras")

if __name__ == '__main__':
    train_strict_clean()
