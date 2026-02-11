import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import get_data_generators
from model import build_model

# Constants
DATA_DIR = os.path.join(os.getcwd(), 'raw_dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 # Adjust based on time availability
LEARNING_RATE = 0.0001

def train():
    print("--- Waste Classification Training ---")
    
    # Check data
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    # Load Data
    train_gen, val_gen = get_data_generators(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    num_classes = len(train_gen.class_indices)
    
    # Build Model
    model = build_model(num_classes=num_classes, img_size=IMG_SIZE, learning_rate=LEARNING_RATE)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'waste_classifier.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    # Train
    print(f"Starting training for {EPOCHS} epochs...")
    try:
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[checkpoint, early_stopping, lr_reducer]
        )
        
        # Explicit final save
        model.save('waste_classifier_final.keras')
        print("Training complete. Models saved: 'waste_classifier.keras' (best) and 'waste_classifier_final.keras' (final).")
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    train()
