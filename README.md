# 🌾 Crop Prediction & Animal Protection System

## 📝 Introduction
This project integrates **Crop Prediction & Management** with an **AI-powered Animal Protection System** to assist farmers in optimizing their crop yield while preventing animal intrusions.
- The **Crop Prediction System** leverages real-time weather and soil data to recommend suitable crops.
- The **Animal Protection System** uses AI-based image recognition to detect animals like elephants and peacocks, sending real-time alerts.
- Both systems store and process data using **Firebase**.

## 🚀 Features
### 🌱 Crop Prediction & Management
- **Real-time Data Processing:** Uses **OpenWeatherMap API** and sensor data (humidity, temperature, rainfall, soil moisture).
- **Threshold-Based Analysis:** Compares live data with pre-defined crop thresholds to provide feedback (-1, 0, +1).
- **Firebase Integration:** Stores crop recommendations and advice.

### 🦌 Animal Protection System
- **AI-based Animal Detection:** Identifies elephants and peacocks using a deep learning model trained on the **TARP dataset**.
- **Live Camera Feed Analysis:** Uses **OpenCV** and **TensorFlow/PyTorch** for real-time object detection.
- **Push Notifications:** Sends alerts when an animal is detected.
- **Firebase Integration:** Stores detection logs and images.

## 🛠️ Technologies Used
- **Python** for data processing and machine learning.
- **Flutter** for the mobile application interface.
- **Firebase** for real-time database storage.
- **OpenCV & TensorFlow/PyTorch** for image recognition.
- **OpenWeatherMap API** for live weather data.

## 📥 Installation & Setup
### 🔹 Clone the Repository
```sh
git clone https://github.com/your-repo/Crop_protection_management.git
```
### 🔹 Install Dependencies
```sh
pip install opencv-python tensorflow firebase-admin requests
```
### 🔹 Set Up Firebase
- Place `google-services.json` (for Android) or `GoogleService-Info.plist` (for iOS) in the appropriate folder.
- Configure Firebase Realtime Database rules.

### 🔹 Start the Crop Prediction System
```sh
python crop_prediction.py
```

### 🔹 Start the Animal Detection System
```sh
python animal_detection.py
```

## 📌 How It Works
### 🌱 Crop Prediction
1. The system collects real-time sensor and weather data.
2. It compares data against predefined crop thresholds.
3. The system recommends the best crops and stores results in **Firebase**.

### 🦌 Animal Protection
1. The system processes live video feed using a deep learning model.
2. If an animal is detected, an alert is sent via push notification.
3. Detection logs are stored in **Firebase**.

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
📧 **Email:** abhi11.sbsm@gmail.com  
🔗 **GitHub:** [Crop_Protection_Management](https://github.com/Dracerxy/Crop_protection_Management)

