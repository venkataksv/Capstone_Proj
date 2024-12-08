import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
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

# Normalize images
images = images / 255.0

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Build a CNN model 

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# save the model 

model.save('fruit_classifier.h5')
