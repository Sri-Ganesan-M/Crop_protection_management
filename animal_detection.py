import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps  # For image resizing and processing with PIL
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Firebase
cred = credentials.Certificate(r"")
firebase_admin.initialize_app(cred, {
    'databaseURL': ''
})

# Load the model
model_path = ""  # Path to the new Keras model
model = load_model(model_path, compile=False)

# Load the labels
with open("", "r") as f:
    class_names = f.readlines()

# Define constants
img_size = (224, 224)  # Input size for the new model
esp32_url = ""  # ESP32-CAM IP URL (to be updated dynamically from Firebase)

# Firebase listener to update IP address
def update_ip_address(event):
    global esp32_url
    ip_address = event.data
    esp32_url = f"http://{ip_address}/capture"
    print(f"Updated ESP32-CAM IP address: {esp32_url}")

# Attach listener to the IP address node
db.reference('/IP/address').listen(update_ip_address)

# Define the virtual boundary (centered rectangle)
frame_width = 640  # Width of the frame (if known)
frame_height = 480  # Height of the frame (if known)
boundary_top_left = (frame_width // 4, frame_height // 4)
boundary_bottom_right = (3 * frame_width // 4, 3 * frame_height // 4)

# Function to check if animal is inside the boundary
def is_inside_boundary(center_x, center_y):
    return (boundary_top_left[0] <= center_x <= boundary_bottom_right[0] and
            boundary_top_left[1] <= center_y <= boundary_bottom_right[1])

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
        db.reference('/status/sound').set(detected)

# Main loop for fetching and displaying frames
while True:
    if esp32_url:  # Only proceed if the IP address is set
        try:
            # Fetch image from ESP32-CAM
            response = requests.get(esp32_url)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Check if the frame was decoded successfully
                if frame is None:
                    print("Failed to decode image from ESP32-CAM")
                    continue

                # Predict the animal in the frame
                animal, confidence = predict_animal_pil(frame)

                # Define center coordinates of the animal detection (for simplicity)
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

                # Check if the detection is inside the boundary
                inside_boundary = is_inside_boundary(center_x, center_y)

                # Update Firebase if peacock or elephant is detected with confidence > 50%
                if animal == "1 peacock" and confidence > 0.5:
                    update_firebase_detection("peacock", True)
                else:
                    update_firebase_detection("peacock", False)  # Reset if not detected

                if animal == "0 elephant" and confidence > 0.5:
                    update_firebase_detection("elephant", True)
                else:
                    update_firebase_detection("elephant", False)  # Reset if not detected

                # Draw the virtual boundary
                cv2.rectangle(frame, boundary_top_left, boundary_bottom_right, (255, 0, 0), 2)

                # Display result with different colors based on boundary check
                text_color = (0, 255, 0) if inside_boundary else (0, 0, 255)
                cv2.putText(frame, f"Animal: {animal} ({confidence * 100:.2f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                # Show the frame with OpenCV
                cv2.imshow("Animal Detection", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ESP32-CAM: {e}")

        except cv2.error as e:
            print(f"OpenCV error: {e}")

    else:
        print("Waiting for ESP32-CAM IP address...")
        cv2.waitKey(1000)  # Wait before retrying to reduce CPU usage

# Close all windows safely
cv2.destroyAllWindows()