import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import glob


def train_hybrid_model():
    # Load the CSV file
    print(os.getcwd())
    csv_path = "/Users/tolabowenmaccurtain/Desktop/ISE/Block-7/AI/Project/synthesis/output/labels.csv"
    df = pd.read_csv(csv_path)

    # Map 'img_letter' to numerical labels
    unique_letters = df["img_letter"].unique()
    letter_to_label = {letter: i for i, letter in enumerate(unique_letters)}
    label_to_letter = {i: letter for letter, i in letter_to_label.items()}

    df["label"] = df["img_letter"].map(letter_to_label)

    # Load images and labels
    image_dir = "/Users/tolabowenmaccurtain/Desktop/ISE/Block-7/AI/Project/synthesis/output/temp-letters-dir"
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['img_no']:05}.png")
        if os.path.exists(img_path):
            img = load_img(img_path, color_mode="grayscale", target_size=(128, 128))
            img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(row["label"])

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(unique_letters))

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the hybrid CNN-RNN model
    input_shape = (128, 128, 1)

    # CNN Component
    inputs = Input(shape=input_shape)

    # CNN layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Reshape for RNN
    # After 3 max pooling layers (2x2), the 128x128 image becomes (128/8)x(128/8)x128 = 16x16x128
    x = Reshape((16, 16*128))(x)

    # RNN layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(unique_letters), activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Add callbacks for better training
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_ogham_hybrid_model.h5', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # We'll use early stopping to determine the actual number of epochs
        batch_size=32,
        callbacks=callbacks
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')

    # Evaluate the model
    model = load_model('best_ogham_hybrid_model.h5')  # Load the best model saved by ModelCheckpoint
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

    # Save the mapping for prediction
    with open('label_to_letter.json', 'w') as f:
        json.dump({str(k): v for k, v in label_to_letter.items()}, f)

    return model, label_to_letter


# Predict on a new image
def predict_image(image_path, model_path="best_ogham_hybrid_model.h5"):

    model = load_model(model_path)

    # Load label mapping
    with open('label_to_letter.json', 'r') as f:
        label_to_letter = json.load(f)
        label_to_letter = {int(k): v for k, v in label_to_letter.items()}

    # Preprocess the image
    img = load_img(image_path, color_mode="grayscale", target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]

    # Get top 3 predictions for ambiguous cases
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_predictions = [(label_to_letter[idx], float(prediction[0][idx])) for idx in top_indices]

    plt.imshow(img[0, :, :, 0], cmap="gray")
    plt.title(f"Predicted: {label_to_letter[predicted_label]} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()


    return {
        'predicted': label_to_letter[predicted_label],
        'confidence': float(confidence),
        'top_predictions': top_predictions
    }

# Function to run inference on a set of images
def batch_predict(image_dir, pattern="*.png"):
    model = load_model("best_ogham_hybrid_model.h5")
    results = []

    for img_path in glob.glob(os.path.join(image_dir, pattern)):
        result = predict_image(img_path, model)
        results.append({
            'image': os.path.basename(img_path),
            'prediction': result
        })

    return results




if __name__ == "__main__":
    # Choose one of these training functions:

    #Option 1: Train the basic hybrid model
    #model, label_mapping = train_hybrid_model()

    # OR

    # Option 2: Train with data augmentation (recommended for better results)
    # model, label_mapping = train_with_augmentation()


    # print("Training complete! Model saved.")
    test_image_path = ("/Users/tolabowenmaccurtain/Desktop/ISE/Block-7/AI/Project/model/Test(g).png")  # Update this with the actual image path
    result = predict_image(test_image_path)
    print(f"Predicted: {result['predicted']}, Confidence: {result['confidence']:.2f}")