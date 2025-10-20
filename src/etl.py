import pandas as pd
import numpy as np
from datetime import datetime

# 1. Define the number of data points
NUM_RECORDS = 10000

# 2. Generate Random/Sequential Data

# --- CORRECTED CODE FOR DATES ---
start_date = datetime(2024, 10, 1) # Start date for simulation
dates = pd.date_range(start=start_date, periods=NUM_RECORDS, freq='H')
# ---------------------------------

# Simulate Geo-Location in the Greater Jakarta area (approximate ranges)
latitudes = np.random.uniform(-6.3, -6.1, NUM_RECORDS)
longitudes = np.random.uniform(106.7, 107.0, NUM_RECORDS)

# Simulate Demand
demand = np.random.randint(50, 500, NUM_RECORDS)
# Simulate higher demand on a weekend day (multiplying every 7th record by 1.5)
demand[::7] = (demand[::7] * 1.5).astype(int)

# Simulate Fuel Price Factor 
fuel_factor = 1.0 + (np.random.rand(NUM_RECORDS) * 0.1)

# 3. Assemble the INITIAL DataFrame to create helper columns
# We must assemble the DataFrame first to use its properties (like dayofweek)
data = pd.DataFrame({
    'transaction_id': range(1, NUM_RECORDS + 1),
    'delivery_date': dates,
    'pickup_latitude': latitudes,
    'pickup_longitude': longitudes,
    'demand_volume': demand,
    'fuel_price_factor': fuel_factor
})

# --- NEW: Create Helper Features for Price Calculation ---
data['day_of_week'] = data['delivery_date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # 5=Sat, 6=Sun
# --------------------------------------------------------


# 4. Create the Target Variable (Final Price) - INJECTING STRONG SIGNAL
# The price is a function of multiple factors, ensuring a strong R^2 potential

BASE_PRICE = 45000 # Minimum fixed cost

# A. Demand Influence: A strong, positive correlation
demand_premium = data['demand_volume'] * 85 

# B. Weekend Surcharge: Adds a flat fee for weekend delivery
weekend_surcharge = data['is_weekend'] * 12000

# C. Geo-Influence: Simulate a cost gradient (e.g., further from a central point -6.2, 106.8)
# This uses latitude to simulate distance/difficulty.
# Higher latitude values (closer to 0) are less expensive, lower (more negative) are more expensive.
# Factor creates a range of about -15000 to +15000 based on location
geo_influence = (data['pickup_latitude'] + 6.2) * 150000 

# D. Fuel/Cost Multiplier: Multiplies the total cost
fuel_multiplier = data['fuel_price_factor'] * 1.05

# E. Final Calculation: Base cost + premiums, multiplied by the fuel factor, plus noise
final_price = (BASE_PRICE + demand_premium + weekend_surcharge + geo_influence) * fuel_multiplier
# Add small, normally distributed noise for realism
data['final_price_idr'] = final_price + np.random.normal(0, 500, NUM_RECORDS)
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------
# FEATURE ENGINEERING: Adding Traffic Stress Index
# ----------------------------------------------------------------------

# 1. Extract the hour from the delivery_date column
# The 'dt.hour' accessor works because 'delivery_date' was created using pd.date_range
data['hour'] = data['delivery_date'].dt.hour

# 2. Define the conditions for peak traffic (e.g., 7-9 AM and 5-7 PM)
# This simulates the real friction of Jakarta traffic
morning_peak = (data['hour'] >= 7) & (data['hour'] <= 9)
evening_peak = (data['hour'] >= 17) & (data['hour'] <= 19)

# 3. Create the new feature column (traffic_peak_factor)
# np.where assigns 1.5 (higher cost/friction) during peak times, and 1.0 otherwise.
data['traffic_peak_factor'] = np.where(morning_peak | evening_peak, 1.5, 1.0)

# 4. (Optional but recommended) Verify the new feature was created
print("New Traffic Peak Feature Head:")
print(data[['delivery_date', 'hour', 'traffic_peak_factor']].head(10))

# 5. Drop the temporary 'hour' column (we don't need it in the final table)
data.drop(columns=['hour'], inplace=True)


# 5. Finalize the DataFrame (Drop temporary columns)
# Drop the helper columns before saving the CSV, as the model will re-engineer them later.
data = data.drop(columns=['day_of_week', 'is_weekend'])


# 6. Data Cleaning and Save
data = data[data['final_price_idr'] > 0] 
data.dropna(inplace=True) 

data.to_csv('data/logistic_data.csv', index=False)
print("Data simulation and cleaning complete. Ready for SQL load.")