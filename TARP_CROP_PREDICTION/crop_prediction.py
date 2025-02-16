# import firebase_admin
# from firebase_admin import credentials
# import pandas as pd
# import requests
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import time
# from google.cloud import firestore
# import google.generativeai as genai
# # Initialize Firebase Realtime Database
# cred = credentials.Certificate(r"/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/smart-crop-management-app-firebase-adminsdk-7n77g-ebb8f0fc25.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://smart-crop-management-app-default-rtdb.firebaseio.com/'
# })

# from firebase_admin import db

# # Load dataset and train model
# file_path = r'/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/Modified_Crop_recommendation.csv'
# crop_data_modified = pd.read_csv(file_path)
# X = crop_data_modified[['Temperature', 'Humidity', 'Rainfall', 'soil_moisture']]
# y = crop_data_modified['Crop']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train RandomForest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# accuracy = accuracy_score(y_test, model.predict(X_test))
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# thresholds_df = pd.read_csv(r'/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/crop_thresholds.csv')

# # Function to fetch rainfall data from OpenWeatherMap API
# def fetch_rainfall(lat, lon):
#     url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation"
#     response = requests.get(url)
#     data = response.json()
    
#     # Extract hourly precipitation (rainfall) data
#     if 'hourly' in data:
#         rainfall = data['hourly'].get('precipitation', [0])  # Get hourly rainfall data, default to 0 if not available
#         total_rainfall = sum(rainfall)  # Sum the hourly rainfall for total rainfall
#     else:
#         total_rainfall = 0  # Default to 0 if no rainfall data is available
    
#     return total_rainfall
# def get_thresholds(crop_type):
#     crop_thresholds = thresholds_df[thresholds_df['Crop'] == crop_type]
#     if not crop_thresholds.empty:
#         return {
#             'Temperature': (crop_thresholds['Min'].values[0], crop_thresholds['Max'].values[0]),
#             'Humidity': (crop_thresholds['Min'].values[1], crop_thresholds['Max'].values[1]),
#             'Rainfall': (crop_thresholds['Min'].values[2], crop_thresholds['Max'].values[2]),
#             'soil_moisture': (crop_thresholds['Min'].values[3], crop_thresholds['Max'].values[3])
#         }
#     return None
# def evaluate_parameter(value, param_range):
#     min_val, max_val = param_range
#     if value < min_val:
#         return -1
#     elif value > max_val:
#         return +1
#     else:
#         return 0
# def evaluate_hardcoded_criteria(humidity):
#     # Example: Hardcoded humidity criteria
#     hardcoded_humidity_min = 60
#     hardcoded_humidity_max = 80
#     if humidity < hardcoded_humidity_min:
#         return -1
#     elif humidity > hardcoded_humidity_max:
#         return +1
#     else:
#         return 0
# def evaluate_hardcoded_temperature(temperature):
#     # Hardcoded temperature criteria
#     hardcoded_temp_min = 18  # Minimum temperature for crops
#     hardcoded_temp_max = 35  # Maximum temperature for crops
#     if temperature < hardcoded_temp_min:
#         return -1
#     elif temperature > hardcoded_temp_max:
#         return +1
#     else:
#         return 0
# def evaluate_hardcoded_rainfall(rainfall):
#     # Hardcoded rainfall criteria (in mm)
#     hardcoded_rainfall_min = 100  # Minimum rainfall for crops (in mm)
#     hardcoded_rainfall_max = 300  # Maximum rainfall for crops (in mm)
#     if rainfall < hardcoded_rainfall_min:
#         return -1
#     elif rainfall > hardcoded_rainfall_max:
#         return +1
#     else:
#         return 0
# def evaluate_hardcoded_soil_moisture(soil_moisture):
#     # Hardcoded soil moisture criteria (percentage)
#     hardcoded_soil_moisture_min = 20  # Minimum soil moisture (in %)
#     hardcoded_soil_moisture_max = 60  # Maximum soil moisture (in %)
#     if soil_moisture < hardcoded_soil_moisture_min:
#         return -1
#     elif soil_moisture > hardcoded_soil_moisture_max:
#         return +1
#     else:
#         return 0
# # Function to get advice from Gemini API based on environmental conditions
# def get_farmer_advice(temperature, humidity, rainfall, soil_moisture):
#     genai.configure(api_key='AIzaSyCEYbDu9Bfoq4cKkm_IjRgyWa97h4nnnUU')
#     model = genai.GenerativeModel('gemini-pro')
#     prompt = f"Given the temperature: {temperature}, humidity: {humidity}, rainfall: {rainfall}, and soil moisture: {soil_moisture}, provide advice for farmers in 100 words as points."
#     response = model.generate_content(prompt)
#     advice = response.text.replace('> *', '').replace('*', '').strip()
#     return advice

# # Function to collect data, predict crop, and send advice to Firebase Realtime Database
# def predict_and_update_firebase(lat, lon):
#     # Fetch environmental data from Firebase Realtime Database
#     ref = db.reference("sensorData")
#     ref2=db.reference("detection")
#     data = ref.get()
#     detection_data=ref2.get()
#     if data:
#         temperature = data.get("temperature")
#         humidity = data.get("humidity")
#         soil_moisture = data.get("soilMoisture")
#         crop_planted=data.get("crop_planted")
#     else:
#         print("No farm data found.")
#         return

#     # Fetch rainfall from OpenWeatherMap
#     rainfall = fetch_rainfall(lat, lon)
#     ref.update({"rainfall": rainfall}) 
#     # Predict crop
#     predicted_crop = model.predict([[temperature, humidity, rainfall, soil_moisture]])[0]
#     thresholds = get_thresholds(crop_planted)
#     # Get advice for the farmer
#     advice = get_farmer_advice(temperature, humidity, rainfall, soil_moisture)
#     if thresholds:
#         temp_status = evaluate_parameter(temperature, thresholds['Temperature'])
#         humidity_status = evaluate_parameter(humidity, thresholds['Humidity'])
#         rainfall_status = evaluate_parameter(rainfall, thresholds['Rainfall'])
#         soil_moisture_status = evaluate_parameter(soil_moisture, thresholds['soil_moisture'])

#         # Evaluate hardcoded criteria
#         hardcoded_humidity_status = evaluate_hardcoded_criteria(humidity)
#         hardcoded_temp_status = evaluate_hardcoded_temperature(temperature)
#         hardcoded_rainfall_status = evaluate_hardcoded_rainfall(rainfall)
#         hardcoded_soil_moisture_status = evaluate_hardcoded_soil_moisture(soil_moisture)

#         # Print out status for each parameter
#         print(f"Temperature Status: {temp_status}")
#         print(f"Humidity Status: {humidity_status} (Hardcoded Status: {hardcoded_humidity_status})")
#         print(f"Rainfall Status: {rainfall_status} (Hardcoded Status: {hardcoded_rainfall_status})")
#         print(f"Soil Moisture Status: {soil_moisture_status} (Hardcoded Status: {hardcoded_soil_moisture_status})")

#         # Update Firebase with results
#         db.reference("detection").set({
#             "recommended_crop": predicted_crop,
#             "temperature_status": temp_status,
#             "humidity_status": humidity_status,
#             "hardcoded_humidity_status": hardcoded_humidity_status,
#             "rainfall_status": rainfall_status,
#             "hardcoded_rainfall_status": hardcoded_rainfall_status,
#             "soil_moisture_status": soil_moisture_status,
#             "hardcoded_soil_moisture_status": hardcoded_soil_moisture_status,
#             "timestamp": firestore.SERVER_TIMESTAMP
#         })
#         print("Data has been updated in Firebase.")
#     else:
#         print(f"No thresholds available for the predicted crop: {predicted_crop}")


#     # Update Firebase Realtime Database with prediction and advice
#     results_ref = db.reference("predictionResults")
#     results_ref.set({
#         "recommended_crop": predicted_crop,
#         "advice": advice,
#         "timestamp": time.time()
#     })
#     print("Prediction and advice have been updated in Firebase Realtime Database in predictionResults.")

# # Set up OpenWeatherMap API key and location details
# latitude = 13.067439  # Replace with farm's latitude
# longitude = 80.237617  # Replace with farm's longitude
# # Continuously monitor for changes in Firebase Realtime Database
# def monitor_firebase_and_predict():
#     previous_data = None
#     while True:
#         # Check current data in Firebase
#         ref = db.reference("sensorData")
#         current_data = ref.get()

#         # Only proceed if data has changed
#         if current_data != previous_data:
#             print("Detected change in farm conditions, processing new data.")
#             predict_and_update_firebase(latitude, longitude)
#             previous_data = current_data
        
#         # Wait for a few seconds before checking again
#         time.sleep(5)

# # Start monitoring Firebase Realtime Database and predict crop
# monitor_firebase_and_predict()
import firebase_admin
from firebase_admin import credentials
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
from google.cloud import firestore
import google.generativeai as genai
from datetime import datetime
crop_planted=""
# Initialize Firebase Realtime Database
cred = credentials.Certificate(r"/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/smart-crop-management-app-firebase-adminsdk-7n77g-ebb8f0fc25.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-crop-management-app-default-rtdb.firebaseio.com/'
})

from firebase_admin import db

# Load dataset and train model
file_path = r'/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/Modified_Crop_recommendation.csv'
crop_data_modified = pd.read_csv(file_path)
X = crop_data_modified[['Temperature', 'Humidity', 'Rainfall', 'soil_moisture']]
y = crop_data_modified['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

thresholds_df = pd.read_csv(r'/Users/sriganesan/DATA/trap_project/TARP_CROP_PREDICTION/crop_thresholds.csv')

# Function to fetch rainfall data from OpenWeatherMap API
def fetch_rainfall(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation"
    response = requests.get(url)
    data = response.json()
    
    # Extract hourly precipitation (rainfall) data
    if 'hourly' in data:
        rainfall = data['hourly'].get('precipitation', [0])  # Get hourly rainfall data, default to 0 if not available
        total_rainfall = sum(rainfall)  # Sum the hourly rainfall for total rainfall
    else:
        total_rainfall = 0  # Default to 0 if no rainfall data is available
    
    return total_rainfall

def get_thresholds(crop_type):
    crop_thresholds = thresholds_df[thresholds_df['Crop'] == crop_type]
    if not crop_thresholds.empty:
        return {
            'Temperature': (crop_thresholds['Min'].values[0], crop_thresholds['Max'].values[0]),
            'Humidity': (crop_thresholds['Min'].values[1], crop_thresholds['Max'].values[1]),
            'Rainfall': (crop_thresholds['Min'].values[2], crop_thresholds['Max'].values[2]),
            'soil_moisture': (crop_thresholds['Min'].values[3], crop_thresholds['Max'].values[3])
        }
    return None

def evaluate_parameter(value, param_range):
    min_val, max_val = param_range
    if value < min_val:
        return -1
    elif value > max_val:
        return 1
    else:
        return 0

def evaluate_hardcoded_criteria(humidity):
    # Example: Hardcoded humidity criteria
    hardcoded_humidity_min = 60
    hardcoded_humidity_max = 80
    if humidity < hardcoded_humidity_min:
        return -1
    elif humidity > hardcoded_humidity_max:
        return +1
    else:
        return 0

def evaluate_hardcoded_temperature(temperature):
    # Hardcoded temperature criteria
    hardcoded_temp_min = 18  # Minimum temperature for crops
    hardcoded_temp_max = 35  # Maximum temperature for crops
    if temperature < hardcoded_temp_min:
        return -1
    elif temperature > hardcoded_temp_max:
        return +1
    else:
        return 0

def evaluate_hardcoded_rainfall(rainfall):
    # Hardcoded rainfall criteria (in mm)
    hardcoded_rainfall_min = 100  # Minimum rainfall for crops (in mm)
    hardcoded_rainfall_max = 300  # Maximum rainfall for crops (in mm)
    if rainfall < hardcoded_rainfall_min:
        return -1
    elif rainfall > hardcoded_rainfall_max:
        return +1
    else:
        return 0

def evaluate_hardcoded_soil_moisture(soil_moisture):
    # Hardcoded soil moisture criteria (percentage)
    hardcoded_soil_moisture_min = 20  # Minimum soil moisture (in %)
    hardcoded_soil_moisture_max = 60  # Maximum soil moisture (in %)
    if soil_moisture < hardcoded_soil_moisture_min:
        return -1
    elif soil_moisture > hardcoded_soil_moisture_max:
        return 1
    else:
        return 0

# Function to get advice from Gemini API based on environmental conditions
def get_farmer_advice(temperature, humidity, rainfall, soil_moisture):
    genai.configure(api_key='AIzaSyCEYbDu9Bfoq4cKkm_IjRgyWa97h4nnnUU')
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Given the temperature: {temperature}, humidity: {humidity}, rainfall: {rainfall}, and soil moisture: {soil_moisture} for the crop {crop_planted}, provide advice for farmers in 100 words as points."
    response = model.generate_content(prompt)
    advice = response.text.replace('> *', '').replace('*', '').strip()
    return advice

# Function to collect data, predict crop, and send advice to Firebase Realtime Database
def predict_and_update_firebase(lat, lon):
    # Fetch environmental data from Firebase Realtime Database
    ref = db.reference("sensorData")
    ref2 = db.reference("detection")
    data = ref.get()
    detection_data = ref2.get()
    
    if data:
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        soil_moisture = data.get("soilMoisture", 0)
        crop_planted = data.get("crop_planted", "")
    else:
        print("No farm data found.")
        return

    # Fetch rainfall from OpenWeatherMap
    rainfall = fetch_rainfall(lat, lon)
    ref.update({"rainfall": rainfall}) 
    
    # Predict crop
    predicted_crop = model.predict([[temperature, humidity, rainfall, soil_moisture]])[0]
    thresholds = get_thresholds(crop_planted)
    
    # Get advice for the farmer
    advice = get_farmer_advice(temperature, humidity, rainfall, soil_moisture)
    if thresholds:
        temp_status = evaluate_parameter(temperature, thresholds['Temperature'])
        humidity_status = evaluate_parameter(humidity, thresholds['Humidity'])
        rainfall_status = evaluate_parameter(rainfall, thresholds['Rainfall'])
        soil_moisture_status = evaluate_parameter(soil_moisture, thresholds['soil_moisture'])

        # Evaluate hardcoded criteria
        hardcoded_humidity_status = evaluate_hardcoded_criteria(humidity)
        hardcoded_temp_status = evaluate_hardcoded_temperature(temperature)
        hardcoded_rainfall_status = evaluate_hardcoded_rainfall(rainfall)
        hardcoded_soil_moisture_status = evaluate_hardcoded_soil_moisture(soil_moisture)
        if(soil_moisture_status==-1):
            db.reference("status/irrigation").set(bool(1))
        else:
             db.reference("status/irrigation").set(bool(0))
        
        # Update Firebase with results
        timestamp = datetime.now().isoformat()  # Use ISO format for timestamp
        db.reference("plant_alerts").set({
            # "temperature_status": temp_status,
            # "humidity_status": humidity_status,
            "hardcoded_humidity_status": humidity_status,
            "hardcoded_temp_status": temp_status,
            # "rainfall_status": rainfall_status,
            "hardcoded_rainfall_status": rainfall_status,
            # "soil_moisture_status": soil_moisture_status,
            "hardcoded_soil_moisture_status": soil_moisture_status,
            "timestamp": timestamp  # Use ISO formatted timestamp
        })
        print("Data has been updated in Firebase.")
    else:
        print(f"No thresholds available for the predicted crop: {predicted_crop}")

    # Update Firebase Realtime Database with prediction and advice
    results_ref = db.reference("predictionResults")
    results_ref.set({
        "recommended_crop": predicted_crop,
        "advice": advice,
        "timestamp": timestamp
    })
    print("Prediction and advice sent to Firebase.")

# Real-time listener to detect changes in Firebase
def listen_for_changes():
    ref = db.reference("sensorData")
    ref.listen(lambda event: predict_and_update_firebase(12.9716, 77.5946))

# Start listening for data changes
listen_for_changes()