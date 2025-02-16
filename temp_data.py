import pandas as pd
import requests
from google.cloud import firestore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset for model training
file_path = r'C:\Users\abhi1\Desktop\Machine Learning\TARP_CROP_PREDICTION\Modified_Crop_recommendation.csv'
crop_data_modified = pd.read_csv(file_path)
X = crop_data_modified[['Temperature', 'Humidity', 'Rainfall', 'soil_moisture']]
y = crop_data_modified['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Load threshold CSV file for min and max values per crop
thresholds_df = pd.read_csv(r'C:\Users\abhi1\Desktop\Machine Learning\TARP_CROP_PREDICTION\crop_thresholds.csv')

# Function to fetch rainfall data
def fetch_rainfall(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation"
    response = requests.get(url)
    data = response.json()
    rainfall = data['hourly'].get('precipitation', [0]) if 'hourly' in data else [0]
    total_rainfall = sum(rainfall)
    return total_rainfall

# Function to get threshold for a crop
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

# Evaluation function to compare value against thresholds
def evaluate_parameter(value, param_range):
    min_val, max_val = param_range
    if value < min_val:
        return -1
    elif value > max_val:
        return +1
    else:
        return 0

# Hardcoded criteria evaluation functions

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
        return +1
    else:
        return 0

# Function to predict and update Firebase
def predict_and_update_firebase(lat, lon):
    doc_ref = db.collection("farm_conditions").document("latest")
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        soil_moisture = data.get("soil_moisture", 0)
    else:
        print("No farm data found.")
        return

    # Fetch rainfall data
    rainfall = fetch_rainfall(lat, lon)

    # Predict crop
    predicted_crop = best_rf.predict([[temperature, humidity, rainfall, soil_moisture]])[0]

    # Get thresholds for the predicted crop
    thresholds = get_thresholds(predicted_crop)

    # Check if thresholds were found and evaluate parameters
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

        # Print out status for each parameter
        print(f"Temperature Status: {temp_status}")
        print(f"Humidity Status: {humidity_status} (Hardcoded Status: {hardcoded_humidity_status})")
        print(f"Rainfall Status: {rainfall_status} (Hardcoded Status: {hardcoded_rainfall_status})")
        print(f"Soil Moisture Status: {soil_moisture_status} (Hardcoded Status: {hardcoded_soil_moisture_status})")

        # Update Firebase with results
        db.collection("farm_conditions").document("latest").set({
            "recommended_crop": predicted_crop,
            "temperature_status": temp_status,
            "humidity_status": humidity_status,
            "hardcoded_humidity_status": hardcoded_humidity_status,
            "rainfall_status": rainfall_status,
            "hardcoded_rainfall_status": hardcoded_rainfall_status,
            "soil_moisture_status": soil_moisture_status,
            "hardcoded_soil_moisture_status": hardcoded_soil_moisture_status,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print("Data has been updated in Firebase.")
    else:
        print(f"No thresholds available for the predicted crop: {predicted_crop}")

# Run prediction and update
latitude, longitude = 13.067439, 80.237617
predict_and_update_firebase(latitude, longitude)