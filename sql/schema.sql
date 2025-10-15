-- Use the database created for the project
USE jakarta_logistics_db;

-- Drop the table if it already exists to ensure a clean start
DROP TABLE IF EXISTS logistics_transactions;

-- Create the main transaction table matching the CSV column names and data types

CREATE TABLE logistics_transactions (
    -- 1. transaction_id: Primary Key
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- 2. delivery_date: Date type
    delivery_date DATE NOT NULL,
    
    -- 3. pickup_latitude: Decimal for GPS coordinates
    pickup_latitude DECIMAL(9, 6) NOT NULL,
    
    -- 4. pickup_longitude: Decimal for GPS coordinates
    pickup_longitude DECIMAL(9, 6) NOT NULL,
    
    -- 5. demand_volume: Integer representing demand (e.g., number of packages)
    demand_volume INT NOT NULL,
    
    -- 6. fuel_price_factor: Decimal for cost driver
    fuel_price_factor DECIMAL(5, 2) NOT NULL,
    
    -- 7. final_price_idr: Decimal for the target variable (in Indonesian Rupiah)
    final_price_idr DECIMAL(15, 2) NOT NULL
);