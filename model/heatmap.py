import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras import backend as K
from PIL import Image

def generate_heatmap(model, img_array, pred_index=None):
    """
    Generate a heatmap using Grad-CAM technique
    
    Args:
        model: Pre-trained model
        img_array: Preprocessed image array
        pred_index: Index of the predicted class (if None, uses the class with highest score)
    
    Returns:
        Heatmap overlaid on the original image
    """
    # Expand dimensions to match model input requirements
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Get the last convolutional layer
    # (You may need to adjust this based on your model architecture)
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )
    
    # Create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)
    
    # Compute the gradient of the top predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_tensor)
        tape.watch(last_conv_layer_output)
        
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by corresponding gradients
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    # Weight each channel in the feature map array by importance towards the predicted class
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # Average over all channels to get heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Resize heatmap to match original image dimensions
    original_img_shape = img_array.shape[:2]
    heatmap = cv2.resize(heatmap, (original_img_shape[1], original_img_shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    img_array_rgb = np.uint8(255 * img_array)
    if len(img_array_rgb.shape) == 2:  # If grayscale, convert to RGB
        img_array_rgb = cv2.cvtColor(img_array_rgb, cv2.COLOR_GRAY2RGB)
    
    superimposed_img = heatmap * 0.4 + img_array_rgb
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return superimposed_img, heatmap

def visualize_ogham_attention(model_path, image_path, letter_classes=None, output_path=None):
    """
    Load an Ogham image, predict its class, and visualize the model's attention
    
    Args:
        model_path: Path to the saved model
        image_path: Path to the Ogham letter image
        letter_classes: List of Ogham letter classes in prediction order
        output_path: Path to save the visualization (if None, will display instead)
    """
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale if not already
    
    # Resize to match model input size
    input_shape = model.input_shape[1:3]  # (height, width)
    img = img.resize(input_shape)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Make prediction
    pred = model.predict(np.expand_dims(img_array, axis=0))
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    # Get class name if available
    class_name = f"Class {pred_class}"
    if letter_classes and pred_class < len(letter_classes):
        class_name = letter_classes[pred_class]
    
    # Generate heatmap
    superimposed_img, raw_heatmap = generate_heatmap(model, img_array, pred_class)
    
    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Ogham Letter')
    axes[0].axis('off')
    
    
    # Superimposed
    axes[1].imshow(superimposed_img)
    axes[1].set_title(f'Predicted: {class_name} ({confidence:.2f})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return superimposed_img

# Example usage
if __name__ == "__main__":
    # Replace these with your actual paths and class names
    MODEL_PATH = "best_ogham_hybrid_model.h5"
    IMAGE_PATH = "TODO"
    
    # Optional: List of Ogham letter classes in order of model's output
    # Example: OGHAM_CLASSES = ["Beith", "Luis", "Fearn", "Saille", ...]
    OGHAM_CLASSES = None  # Replace with your actual class names if available
    
    visualize_ogham_attention(MODEL_PATH, IMAGE_PATH, OGHAM_CLASSES)