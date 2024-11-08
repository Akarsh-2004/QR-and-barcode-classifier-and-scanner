import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from pyzbar.pyzbar import decode

# Load the trained model
model = load_model(r'C:\project Machine Learning\model.h5')  # Change this path as needed
IMG_SIZE = (128, 128)

def preprocess_image(img_path):
    """Preprocess the image for prediction"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    if img is None:
        print(f"Error: Image not found or unable to load at {img_path}.")
        return None
    img = cv2.resize(img, IMG_SIZE)  # Resize to the model input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=-1)  # Add the channel dimension
    img = np.expand_dims(img, axis=0)  # Add the batch dimension
    return img

def predict_single_image(img_path):
    """Predict QR or Barcode for a single image and read its content"""
    img = preprocess_image(img_path)
    if img is None:
        return  # Exit if the image could not be loaded
    
    prediction = model.predict(img)
    
    # Assuming binary classification: 0 for QR code, 1 for Barcode
    label = 0 if prediction[0][0] < 0.5 else 1  # Adjust threshold if necessary
    label_name = 'QR Code' if label == 0 else 'Barcode'
    
    # Print only the prediction label
    print(f"Prediction: {label_name}")

    # Decode the content
    decoded_content = decode_image(img_path)
    if decoded_content:
        print(f"Decoded content: {decoded_content}")
    else:
        print("No content could be decoded.")



def decode_image(img_path):
    """Decode the QR code or barcode from the image"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image not found or unable to load at {img_path}.")
        return None
    decoded_objects = decode(img)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')  # Return the first decoded object content
    return None

def test_on_multiple_images(test_dir):
    """Test the model on multiple images in a directory"""
    for filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, filename)
        if os.path.isfile(img_path):  # Ensure it's a file, not a subdirectory
            print(f"Testing image: {filename}")
            predict_single_image(img_path)

# Prompt the user to enter the path of the test image
single_test_image = input("Enter the path of the image file (e.g., C:\\path\\to\\image.jpg): ")

# Check if the file exists
if os.path.exists(single_test_image):
    print("Testing the provided image:")
    predict_single_image(single_test_image)
else:
    print(f"Error: The image file does not exist at {single_test_image}.")
