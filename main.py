import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
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


class FruitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Fruit Recognizer")
        self.model = model

        self.canvas = tk.Label(root)
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT)

        camera_btn = tk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        camera_btn.pack(side=tk.LEFT)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            img = ImageTk.PhotoImage(image)
            self.canvas.configure(image=img)
            self.canvas.image = img

            prediction = self.predict_fruit(file_path)
            self.display_prediction(prediction)

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Camera Feed - Press 'q' to Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

                # Save and process the frame
                image_path = 'captured_image.jpg'
                cv2.imwrite(image_path, frame)

                prediction = self.predict_fruit(image_path)
                self.display_prediction(prediction)
                break

    def predict_fruit(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (100, 100))
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = self.model.predict(image)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        return predicted_label

    def display_prediction(self, prediction):
        tk.messagebox.showinfo("Prediction", f'This is a {prediction}')

if __name__ == "__main__":
    root = tk.Tk()
    model = tf.keras.models.load_model('fruit_classifier.h5')
    app = FruitRecognizerApp(root, model)
    root.mainloop()