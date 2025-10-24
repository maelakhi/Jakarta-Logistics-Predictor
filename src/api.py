import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime

# --- Configuration ---
MODEL_FILE = 'src/best_logistics_model.pkl'

# Coordinates of a central point in Jakarta for distance calculation (e.g., Central Jakarta)
CENTER_LAT = -6.2088 
CENTER_LON = 106.8456

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- Model and Feature Engineering Loader ---
try:
    model = joblib.load(MODEL_FILE)
    print(f"[STATUS] Model loaded successfully from {MODEL_FILE}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None

def calculate_features(data):
    """
    Replicates the exact feature engineering steps from model_trainer.py 
    to ensure the API prediction uses the correct input features.
    """
    
    df = pd.DataFrame([data])
    
    # --- 1. Base Feature Engineering ---
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df['day_of_week'] = df['delivery_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Euclidean distance from the center point (Needed for fuel_distance_cost)
    df['distance_from_center'] = np.sqrt(
        (df['pickup_latitude'] - CENTER_LAT)**2 + 
        (df['pickup_longitude'] - CENTER_LON)**2
    )

    # --- 2. Structural and Polynomial Interaction Features (CRITICAL) ---
    df['distance_squared'] = df['distance_from_center'] ** 2
    
    df['demand_weekend_interaction'] = df['demand_volume'] * df['is_weekend']
    # NOTE: fuel_distance_cost IS KEPT because it was highly important
    df['fuel_distance_cost'] = df['fuel_price_factor'] * df['distance_from_center'] 
    df['fuel_demand_interaction'] = df['fuel_price_factor'] * df['demand_volume']
    df['fuel_weekend_interaction'] = df['fuel_price_factor'] * df['is_weekend']
    df['fuel_latitude_interaction'] = df['fuel_price_factor'] * df['pickup_latitude']
    
    # --- 3. Final Feature Selection (Order must match the training set) ---
    # REDUNDANT FEATURES REMOVED: 'distance_from_center', 'distance_squared'
    feature_list = [
        'demand_volume', 
        'fuel_price_factor', 
        'is_weekend',
        'demand_weekend_interaction', 
        'fuel_distance_cost', 
        'fuel_demand_interaction', 
        'fuel_weekend_interaction',
        'fuel_latitude_interaction'
    ]
    
    return df[feature_list]

@app.route('/predict', methods=['POST'])
def predict_price():
    """Endpoint to receive logistics data and return a price prediction."""
    
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

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
        "next_step": "Send POST request to /predict"
    })

if __name__ == '__main__':
    print("\n--- API STARTUP: Waiting for requests on port 8080 ---")
    app.run(debug=True, port=8080)