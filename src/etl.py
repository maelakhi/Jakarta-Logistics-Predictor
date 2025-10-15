import pandas as pd
import numpy as np
from datetime import datetime # Still needed for defining the start date

# 1. Define the number of data points
NUM_RECORDS = 10000

# 2. Generate Random/Sequential Data

# --- CORRECTED CODE FOR DATES ---
start_date = datetime(2024, 10, 1) # Start date for simulation
# Generate hourly data for 10,000 data points (approx 416 days of historical data)
dates = pd.date_range(start=start_date, periods=NUM_RECORDS, freq='H')
# ---------------------------------

# Simulate Geo-Location in the Greater Jakarta area (approximate ranges)
# Lat: -6.3 to -6.1; Long: 106.7 to 107.0 (Simulating different pick-up spots)
latitudes = np.random.uniform(-6.3, -6.1, NUM_RECORDS)
longitudes = np.random.uniform(106.7, 107.0, NUM_RECORDS)

# Simulate Demand: Higher volume on certain days (e.g., weekends)
demand = np.random.randint(50, 500, NUM_RECORDS)
# Simulate higher demand on a weekend day (multiplying every 7th record by 1.5)
demand[::7] = (demand[::7] * 1.5).astype(int)

# Simulate Fuel Price Factor (Introduce volatility based on time/government factor [1])
fuel_factor = 1.0 + (np.random.rand(NUM_RECORDS) * 0.1)

# 3. Create the Target Variable (Final Price)
# The price is a function of base demand, location, and the fuel factor
final_price = 10000 + (demand * 5) + (fuel_factor * 2000) + (np.random.normal(0, 1000, NUM_RECORDS))

# 4. Assemble the DataFrame and Clean
data = pd.DataFrame({
    'transaction_id': range(1, NUM_RECORDS + 1),
    'delivery_date': dates,
    'pickup_latitude': latitudes,
    'pickup_longitude': longitudes,
    'demand_volume': demand,
    'fuel_price_factor': fuel_factor,
    'final_price_idr': final_price
})

# Example of Data Cleaning (Non-Negotiable ETL step [2]):
# Ensure all prices are positive and remove extreme outliers
data = data[data['final_price_idr'] > 0] 
data.dropna(inplace=True) 

# Save the data to a CSV file in your 'data/' folder for ETL loading
data.to_csv('data/simulated_logistics.csv', index=False)
print("Data simulation and cleaning complete. Ready for SQL load on Day 4.")