import pandas as pd
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# --- Configuration ---
DB_HOST = "localhost"
DB_DATABASE = "jakarta_logistics_db"
DB_USER = "root"
DB_PASSWORD = "200801agis" 

TABLE_NAME = "logistics_transactions"
MODEL_FILE = "src/best_logistics_model.pkl"

# Coordinates of a central point in Jakarta for distance calculation (e.g., Central Jakarta)
CENTER_LAT = -6.2088 
CENTER_LON = 106.8456

def get_logistics_data():
    """Connects to MySQL, executes a query, and returns the results as a Pandas DataFrame."""
    
    sql_query = f"SELECT * FROM {DB_DATABASE}.{TABLE_NAME}"
    conn = None 
    
    try:
        # Establish connection to the MySQL database
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_DATABASE,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        if conn.is_connected():
            print("[STATUS] Successfully connected to MySQL database.")
            df = pd.read_sql(sql_query, conn)
            print(f"[STATUS] Data loaded into DataFrame. Total rows: {len(df)}")
            return df
        
    except Error as e:
        print(f"[ERROR] Error reading data from MySQL: {e}")
        return None
        
    finally:
        if conn and conn.is_connected():
            conn.close()

def prepare_data(df):
    """Performs feature engineering and splits data for training."""
    
    # 1. Convert date column to datetime object
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    
    # 2. Feature Engineering 
    df['day_of_week'] = df['delivery_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # --- BREAKING THE LEAKAGE: TRANSFORMING LOCATION DATA ---
    # Create a non-linear feature: Euclidean distance from the center point
    df['distance_from_center'] = np.sqrt(
        (df['pickup_latitude'] - CENTER_LAT)**2 + 
        (df['pickup_longitude'] - CENTER_LON)**2
    )
    # --------------------------------------------------------

    # 3. Define Features (X) - Using the new distance feature instead of raw coordinates
    X = df[[
        'demand_volume', 
        'fuel_price_factor', 
        'is_weekend',
        'distance_from_center', # NEW Leakage-free geo feature
    ]]
    
    # 4. Define Target (y)
    y = df['final_price_idr']

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    return X_train, X_test, y_train, y_test, X.columns

def compare_models(X_train, X_test, y_train, y_test):
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = []
    print("\n--- Training and Evaluating Models ---")

    for name, model in models.items():
        print(f"Training {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='R2 Score', ascending=False)
    
    print("\n" + "="*50)
    print("        MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(results_df.to_string(float_format="%.2f"))
    print("="*50)
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n[RESULT] The best performing model is the {best_model_name}.")
    
    return models[best_model_name], results_df

if __name__ == "__main__":
    
    print("--- STARTING MODEL TRAINING PIPELINE ---")
    
    logistics_df = get_logistics_data()
    
    if logistics_df is not None:
        
        X_train, X_test, y_train, y_test, features = prepare_data(logistics_df)
        
        best_model, results_df = compare_models(X_train, X_test, y_train, y_test)
        
        if isinstance(best_model, LinearRegression):
            print("\n--- Best Model Coefficients (Linear Regression) ---")
            coefficients = pd.Series(best_model.coef_, index=features)
            print(coefficients.sort_values(ascending=False))

        try:
            joblib.dump(best_model, MODEL_FILE)
            print(f"\n[INFO] Best model ({best_model.__class__.__name__}) saved to {MODEL_FILE} for API deployment.")
        except Exception as e:
            print(f"[ERROR] Could not save model: {e}")
            
    print("\n--- END OF PIPELINE ---")
