CREATE DATABASE IF NOT EXISTS jakarta_logistics_db;

-- Use the database created for the project
USE jakarta_logistics_db;

-- Drop the table if it already exists to ensure a clean start
DROP TABLE IF EXISTS logistics_transactions;

-- Create the main transaction table for logistics data
CREATE TABLE logistics_transactions (
    -- 1. Primary Key: Unique identifier for each transaction
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- 2. Geo-spatial data (TEXT or VARCHAR is easiest for lat,long strings)
    -- Consider using a specialized spatial data type (POINT) later for advanced queries
    pickup_lat_long VARCHAR(50) NOT NULL COMMENT 'Origin coordinates (e.g., "34.0522,-118.2437")',

    -- 3. Delivery Date: Essential for time-series analysis
    delivery_date DATE NOT NULL,
    
    -- 4. Cost/Driver Factor: A float to simulate external cost factors
    cost_driver_factor DECIMAL(5, 2) NOT NULL COMMENT 'Simulated fuel/govt regulation factor',
    
    -- 5. Final Price: The target variable (what you want to predict)
    final_price DECIMAL(10, 2) NOT NULL,
    
    -- Example of another useful non-predicted column
    distance_km INT
);

