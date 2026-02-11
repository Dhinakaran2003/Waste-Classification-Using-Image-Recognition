import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(num_classes=10, img_size=(224, 224), learning_rate=0.0001):
    """
    Builds a Transfer Learning model using MobileNetV2.
    """
    # Load base model without top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    
    # Freeze the base model to preserve pre-trained knowledge
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
    print("Model built successfully.")
