import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from pathlib import Path


# --- Configuration ---
DB_HOST = "localhost"
DB_DATABASE = "jakarta_logistics_db"
DB_USER = "root"
DB_PASSWORD = "200801agis" # Use the password you set during installation

# FIX: Define the project root and then construct the path
# 1. Get the current directory of the script (src/)
script_dir = Path(__file__).parent 

# 2. Go up one level (..) to the project root and then into the data folder
# This creates a Path object that correctly points to the CSV.
CSV_FILE_PATH = script_dir.parent / "data" / "logistic_data.csv"

TABLE_NAME = "logistics_transactions"

def load_csv_to_mysql():
    print("Starting data load process...")
    
    # 1. Read the CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded {len(df)} rows from {CSV_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        return
    
    # 2. Connect to the MySQL Database
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_DATABASE,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        if conn.is_connected():
            print("Successfully connected to MySQL database.")
            
            # 3. Create a cursor object for executing SQL queries
            cursor = conn.cursor()
            
            # 4. Convert DataFrame rows into SQL INSERT statements
            # The 'mysql.connector' library doesn't have a direct 'to_sql' like SQLAlchemy,
            # so we'll use a direct INSERT approach.
            
            # Ensure the table is empty before inserting new data (optional cleanup)
            cursor.execute(f"DELETE FROM {TABLE_NAME}")
            
            # Prepare the INSERT query template
            cols = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            insert_query = f"INSERT INTO {TABLE_NAME} ({cols}) VALUES ({placeholders})"
            
            # Prepare data: convert DataFrame rows to a list of tuples
            data_to_insert = [tuple(row) for row in df.values]
            
            # Execute the batch insert
            cursor.executemany(insert_query, data_to_insert)
            
            # Commit the changes to the database
            conn.commit()
            print(f"Data insertion complete. {cursor.rowcount} rows loaded into {TABLE_NAME}.")
            
    except Error as e:
        print(f"Error connecting to or loading data into MySQL: {e}")
        
    finally:
        # 5. Close the database connection
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection closed.")

if __name__ == "__main__":
    load_csv_to_mysql()