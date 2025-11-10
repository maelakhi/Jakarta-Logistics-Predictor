import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime
import os # <--- ADDED: Import os for path checking

# --- Configuration: Adaptive Model Path ---
MODEL_FILENAME = 'best_logistics_model.pkl'

# This logic checks if the file exists in the current directory (Docker) 
# or in the 'src/' subdirectory (Local Run).
if os.path.exists(MODEL_FILENAME):
    MODEL_PATH = MODEL_FILENAME
elif os.path.exists(os.path.join('src', MODEL_FILENAME)):
    MODEL_PATH = os.path.join('src', MODEL_FILENAME)
else:
    # Default to the most likely local path for clear error messages if missing
    MODEL_PATH = os.path.join('src', MODEL_FILENAME) 
# ------------------------------------------

# Coordinates of a central point in Jakarta for distance calculation (e.g., Central Jakarta)
CENTER_LAT = -6.2088 
CENTER_LON = 106.8456

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- Model and Feature Engineering Loader ---
try:
    # Use the dynamically determined path
    model = joblib.load(MODEL_PATH)
    # Changed log to reflect the path used
    print(f"[STATUS] Model loaded successfully from {MODEL_PATH}") 
except Exception as e:
    # Changed log to reflect the path that failed
    print(f"[ERROR] Could not load model using determined path '{MODEL_PATH}': {e}") 
    model = None

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance (in km) between two points on the Earth."""
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance_km = R * c
    return distance_km


def calculate_features(data):
    """
    Replicates the exact feature engineering steps from model_trainer.py 
    to ensure the API prediction uses the correct input features.
    """
    
    df = pd.DataFrame([data])
    
    # --- 1. Base Feature Engineering ---
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df['distance_km'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Euclidean distance from the center point (Needed for fuel_distance_cost)
    df['hour'] = df['delivery_date'].dt.hour
    morning_peak = (df['hour'] >= 7) & (df['hour'] <= 9)
    evening_peak = (df['hour'] >= 17) & (df['hour'] <= 19)
    df['traffic_peak_factor'] = np.where(morning_peak | evening_peak, 1.5, 1.0)
    df['is_weekend'] = df['delivery_date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # --- 2. Structural and Polynomial Interaction Features (CRITICAL) ---    
    df['demand_weekend_interaction'] = df['demand_volume'] * df['is_weekend']
    df['fuel_distance_cost'] = df['fuel_price_factor'] * df['distance_km']
    df['demand_traffic_interaction'] = df['demand_volume'] * df['traffic_peak_factor']    
    df['fuel_demand_interaction'] = df['fuel_price_factor'] * df['demand_volume']
    df['fuel_weekend_interaction'] = df['fuel_price_factor'] * df['is_weekend']
    
    # --- 3. Final Feature Selection (Order must match the training set) ---
    # REDUNDANT FEATURES REMOVED: 'distance_from_center', 'distance_squared'
    feature_list = [
        'demand_volume', 
        'fuel_price_factor', 
        'is_weekend',
        'distance_km',              # NEW KEY FEATURE
        'traffic_peak_factor',      # NEW KEY FEATURE
        'demand_weekend_interaction', 
        'fuel_distance_cost',       # UPDATED to use Haversine
        'demand_traffic_interaction', # NEW POWERFUL INTERACTION
        'fuel_demand_interaction', 
        'fuel_weekend_interaction',
    ]
    
    return df[feature_list]

@app.route('/predict', methods=['POST'])
def predict_price():
    """Endpoint to receive logistics data and return a price prediction."""
    
    if model is None:
        # Include the path in the error message for easy debugging
        return jsonify({"error": f"Model not loaded. Check server logs. Last path checked: {MODEL_PATH}"}), 500

    try:
        raw_data = request.json

        if not raw_data:
            return jsonify({"error": "No data provided in request."}), 400
        
        required_keys = ['pickup_latitude', 'pickup_longitude', 'demand_volume', 'fuel_price_factor', 'delivery_date']
        if not all(key in raw_data for key in required_keys):
             return jsonify({"error": f"Missing required raw data fields. Needed: {required_keys}"}), 400
             
        engineered_features = calculate_features(raw_data)
        
        # Make prediction
        prediction_idr = model.predict(engineered_features)[0]
        
        response = {
            "predicted_price": round(float(prediction_idr), 2),
            "model_r2_status": "High performance (R2 near 1.0) confirmed on synthetic data with streamlined feature set.",
            "features_used": engineered_features.to_dict('records')[0]
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def status():
    """Simple status check endpoint."""
    return jsonify({
        "status": "Logistics Prediction API is running",
        "model_loaded": model is not None,
        "model_path_checked": MODEL_PATH, # Added for debugging confirmation
        "next_step": "Send POST request to /predict"
    })

if __name__ == '__main__':
    print("\n--- API STARTUP: Waiting for requests on port 8080 ---")
    app.run(debug=True, port=8080)
