import numpy as np
import cv2
import argparse
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/model.h5")

# Load class names (IMPORTANT FIX)
classes = np.load("models/classes.npy", allow_pickle=True)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    print(f"Prediction: {classes[class_index]}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    predict_image(args.image)