import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
from PIL import Image, ImageOps  # For image resizing and processing with PIL
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Firebase
cred = credentials.Certificate(r"/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/smart-crop-management-app-firebase-adminsdk-7n77g-ebb8f0fc25.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-crop-management-app-default-rtdb.firebaseio.com/'
})

# Load the model
model_path = "/Users/sriganesan/Downloads/converted_keras/keras_model.h5"  # Path to the new Keras model
model = load_model(model_path, compile=False)

# Load the labels
with open("/Users/sriganesan/Downloads/converted_keras/labels.txt", "r") as f:
    class_names = f.readlines()

# Define constants
img_size = (224, 224)  # Input size for the new model

# Function to predict animal using the new model
def predict_animal_pil(image):
    # Resize the image to (224, 224) and normalize
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from OpenCV to PIL format
    image = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predict the class and confidence
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]
    return class_name, confidence

# Function to update Firebase with detection result
def update_firebase_detection(animal, detected):
    if animal == "peacock":
        db.reference('/detection/peacock_detected').set(detected)
        db.reference('/status/laser_deterrent').set(detected)
    elif animal == "elephant":
        db.reference('/detection/elephant_detected').set(detected)
        db.reference('/status/sprinkler').set(detected)

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Main loop for fetching and displaying frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam")
        break

    # Predict the animal in the frame
    animal, confidence = predict_animal_pil(frame)

    # Update Firebase if peacock or elephant is detected with confidence > 50%
    if animal == "1 peacock" and confidence > 0.5:
        update_firebase_detection("peacock", True)
    else:
        update_firebase_detection("peacock", False)  # Reset if not detected

    if animal == "0 elephant" and confidence > 0.5:
        update_firebase_detection("elephant", True)
    else:
        update_firebase_detection("elephant", False)  # Reset if not detected

    # Display the prediction result on the frame
    cv2.putText(frame, f"Animal: {animal} ({confidence * 100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with OpenCV
    cv2.imshow("Animal Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()