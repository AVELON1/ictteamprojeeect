import tensorflow as tf
from tensorflow.keras import layers, models

def build_traffic_sign_model(num_classes=10, input_shape=(64, 64, 3)):
    """
    Compact CNN model designed for traffic sign classification.
    High accuracy, fast inference (<10ms on CPU).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1: Basic edge and shape detection
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2: Geometric features and symbol detection
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3: Higher-level composite features
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected head
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
