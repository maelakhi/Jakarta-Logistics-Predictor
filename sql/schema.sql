-- Use the database created for the project
USE jakarta_logistics_db;

-- Drop the table if it already exists to ensure a clean start
DROP TABLE IF EXISTS logistics_transactions;
TRUNCATE TABLE logistics_transactions;

-- Create the main transaction table matching the CSV column names and data types

CREATE TABLE logistics_transactions (
    -- 1. transaction_id: Primary Key
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- 2. delivery_date: Date type (Diperlukan untuk fitur Traffic/Weekend)
    delivery_date DATE NOT NULL,
    
    -- 3. pickup_latitude: Decimal for GPS coordinates (Lokasi Awal)
    pickup_latitude DECIMAL(9, 6) NOT NULL,
    
    -- 4. pickup_longitude: Decimal for GPS coordinates (Lokasi Awal)
    pickup_longitude DECIMAL(9, 6) NOT NULL,

    -- 5. dropoff_latitude: Decimal for GPS coordinates (Lokasi Tujuan - BARU)
    dropoff_latitude DECIMAL(9, 6) NOT NULL,
    
    -- 6. dropoff_longitude: Decimal for GPS coordinates (Lokasi Tujuan - BARU)
    dropoff_longitude DECIMAL(9, 6) NOT NULL,

    -- 7. demand_volume: Integer representing demand (Surge Pricing Factor)
    demand_volume INT NOT NULL,
    
    -- 8. fuel_price_factor: Decimal for cost driver (Macroeconomic Risk Factor)
    fuel_price_factor DECIMAL(5, 2) NOT NULL,
    
    -- 9. final_price_idr: Decimal for the target variable (Harga Akhir)
    final_price_idr DECIMAL(15, 2) NOT NULL
);


ALTER TABLE logistics_transactions
ADD COLUMN traffic_peak_factor DECIMAL(2, 1) DEFAULT 1.0 NOT NULL;

