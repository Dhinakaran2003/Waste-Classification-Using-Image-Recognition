import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_data_generators

def apply_gradcam(model, img_array, intensity=0.5, res=224):
    """
    Computes Grad-CAM heatmap.
    """
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name or 'relu' in layer.name:
            last_conv_layer = layer
            break
            
    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def evaluate():
    model_file = 'waste_classifier.keras'
    if not os.path.exists(model_file):
        model_file = 'waste_classifier_final.keras'
        
    if not os.path.exists(model_file):
        print("Error: No model file found. Please train the model first.")
        return

    print(f"Loading model: {model_file}")
    model = tf.keras.models.load_model(model_file)
    
    DATA_DIR = os.path.join(os.getcwd(), 'raw_dataset')
    _, val_gen = get_data_generators(DATA_DIR)
    
    print("Generating predictions...")
    Y_pred = model.predict(val_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(val_gen.classes, y_pred, target_names=list(val_gen.class_indices.keys())))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(val_gen.classes, y_pred)
    print(cm)

if __name__ == "__main__":
    evaluate()
