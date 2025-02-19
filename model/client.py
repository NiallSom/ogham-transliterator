from load_data import *

csv_path = "output/labels.csv"
df = pd.read_csv(csv_path)

train_model()
# Map 'img_letter' to numerical labels
unique_letters = df["img_letter"].unique()
letter_to_label = {letter: i for i, letter in enumerate(unique_letters)}
label_to_letter = {i: letter for letter, i in letter_to_label.items()}
image_path = "TODO"  # Replace with your actual image path
preprocessed_path = pre_process_image(image_path)
predicted_letter = predict_image(preprocessed_path, label_to_letter)
print(f"Predicted Letter: {predicted_letter}")
