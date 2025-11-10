import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration ---
NUM_RECORDS = 10000
START_DATE = datetime(2024, 10, 1)

# --- Utility Function: Calculate Haversine Distance (simplified for numpy) ---
# This function is used to create a strong, realistic signal in the final price calculation.
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance_km = R * c
    return distance_km

# 1. Generate Random/Sequential Data
dates = pd.date_range(start=START_DATE, periods=NUM_RECORDS, freq='H')

# Simulate Geo-Location in the Greater Jakarta area (approximate ranges)
# NOTE: The ranges for pickup and dropoff are kept the same to simulate delivery within the area.
pickup_latitudes = np.random.uniform(-6.3, -6.1, NUM_RECORDS)
pickup_longitudes = np.random.uniform(106.7, 107.0, NUM_RECORDS)
dropoff_latitudes = np.random.uniform(-6.3, -6.1, NUM_RECORDS)
dropoff_longitudes = np.random.uniform(106.7, 107.0, NUM_RECORDS)

# Simulate Demand
demand = np.random.randint(50, 500, NUM_RECORDS)
demand[::7] = (demand[::7] * 1.5).astype(int) # Simulate higher demand on a periodic basis

# Simulate Fuel Price Factor 
fuel_factor = 1.0 + (np.random.rand(NUM_RECORDS) * 0.1)

# 2. Assemble the INITIAL DataFrame
data = pd.DataFrame({
    'transaction_id': range(1, NUM_RECORDS + 1),
    'delivery_date': dates,
    'pickup_latitude': pickup_latitudes,
    'pickup_longitude': pickup_longitudes,
    'dropoff_latitude': dropoff_latitudes, # NEW COLUMN
    'dropoff_longitude': dropoff_longitudes, # NEW COLUMN
    'demand_volume': demand,
    'fuel_price_factor': fuel_factor
})

# --- NEW: Create Helper Features for Price Calculation ---
data['hour'] = data['delivery_date'].dt.hour
data['day_of_week'] = data['delivery_date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # 5=Sat, 6=Sun
data['traffic_peak_factor'] = np.where(
    ((data['hour'] >= 7) & (data['hour'] <= 9)) | ((data['hour'] >= 17) & (data['hour'] <= 19)), 
    1.5, 
    1.0
)
# Calculate the distance in kilometers
data['distance_km'] = haversine(
    data['pickup_latitude'], data['pickup_longitude'], 
    data['dropoff_latitude'], data['dropoff_longitude']
)
# --------------------------------------------------------


# 3. Create the Target Variable (Final Price) - INJECTING STRONG SIGNAL

BASE_PRICE = 45000 
PRICE_PER_KM = 3500 # Strong correlation with distance
DEMAND_PREMIUM_PER_UNIT = 85
WEEKEND_SURCHARGE = 12000

# A. Base Distance Cost
distance_cost = data['distance_km'] * PRICE_PER_KM

# B. Demand Influence
demand_premium = data['demand_volume'] * DEMAND_PREMIUM_PER_UNIT 

# C. Time/Day Surcharge
time_surcharge = (data['is_weekend'] * WEEKEND_SURCHARGE) + (data['traffic_peak_factor'] - 1.0) * 15000 

# D. Fuel Multiplier
fuel_multiplier = data['fuel_price_factor'] * 1.05

# E. Final Calculation: 
final_price = (BASE_PRICE + distance_cost + demand_premium + time_surcharge) * fuel_multiplier
# Add small, normally distributed noise for realism
data['final_price_idr'] = final_price + np.random.normal(0, 500, NUM_RECORDS)
# Ensure price is positive
data['final_price_idr'] = np.maximum(data['final_price_idr'], 10000)
# ----------------------------------------------------------------------


# 4. Finalize the DataFrame for CSV Save
# ONLY include raw data in the final CSV/SQL load. Derived features are dropped.
columns_to_keep = [
    'transaction_id', 
    'delivery_date', 
    'pickup_latitude', 
    'pickup_longitude',
    'dropoff_latitude', 
    'dropoff_longitude', # NEW
    'demand_volume', 
    'fuel_price_factor', 
    'final_price_idr'
]
data = data[columns_to_keep]

# 5. Data Cleaning and Save
data.dropna(inplace=True) 

data.to_csv('data/logistic_data.csv', index=False)
print("Data simulation and cleaning complete. Ready for SQL load.")
print(f"Total records saved: {len(data)}")
print("\n--- Columns in saved CSV ---")
print(data.columns.tolist())
