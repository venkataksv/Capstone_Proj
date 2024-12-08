import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path to dataset
dataset_path = "fruits-360_extended/fruits-360/fruits-360/Training"

images = []
labels = []


# load the dataset

for label_folder in os.listdir(dataset_path):
    label_folder_path = os.path.join(dataset_path, label_folder)
    for image_name in os.listdir(label_folder_path):
        image_path = os.path.join(label_folder_path, image_name)
        # Load the image and resize to a fixed size (e.g., 100x100)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (100, 100))
        images.append(image)
        labels.append(label_folder)

images = np.array(images)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)



def predict_fruit(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = np.expand_dims(image, axis=0) / 255.0

    prediction = model.predict(image)
    metrics = model.compile_metrics()
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    return predicted_label


# load model later
model = tf.keras.models.load_model('fruit_classifier.h5')

# Example usage
fruit = predict_fruit(r'fruits-360_extended/fruits-360/fruits-360/Test/Apple 6/r0_3_100.jpg')

print(f'This is a {fruit}')
