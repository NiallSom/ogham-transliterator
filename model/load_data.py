import os
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

def train_model():
    # # Load the CSV file
    csv_path = "output/labels.csv"
    df = pd.read_csv(csv_path)

    # Map 'img_letter' to numerical labels
    unique_letters = df["img_letter"].unique()
    letter_to_label = {letter: i for i, letter in enumerate(unique_letters)}
    label_to_letter = {i: letter for letter, i in letter_to_label.items()}

    df["label"] = df["img_letter"].map(letter_to_label)

    # Load images and labels
    image_dir = "output/temp-letters-dir"
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['img_no']:05}.png")
        if os.path.exists(img_path):
            img = load_img(img_path, color_mode="grayscale", target_size=(128, 128))  # Resize to 128x128
            img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(row["label"])

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(unique_letters))

    # Split into training and validation sets
    print(labels)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


    # Define the model
    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Fully connected layers
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),  # Dropout to prevent overfitting
        Dense(len(unique_letters), activation="softmax")  # Output layer for Ogham characters
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Print model summary
    model.summary()


    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # Adjust the number of epochs
        batch_size=32  # Adjust the batch size
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

    model.save("ogham_letter_classifier.h5")
    return label_to_letter

# Predict on a new image
def predict_image(image_path,label_to_letter):
    # Load the saved model
    model = load_model("ogham_letter_classifier.h5")
    img = load_img(image_path, color_mode="grayscale", target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return label_to_letter[predicted_label]

def pre_process_image(image_path):
    base_name = os.path.splitext(image_path)[0]
    ext = os.path.splitext(image_path)[1]
    output_path = f"{base_name}_pre-processed{ext}"

    hand_drawn_image = Image.open(image_path)
    hand_drawn_image = hand_drawn_image.convert("L")  # Convert to grayscale
    hand_drawn_image = hand_drawn_image.resize((300, 300), Image.LANCZOS)
    image_array = np.array(hand_drawn_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension (for grayscale)
    # Apply thresholding
    hand_drawn_image = hand_drawn_image.point(lambda p: 255 if p > 128 else 0)
    # Save the processed image
    hand_drawn_image.save(output_path)

    return output_path
