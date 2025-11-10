import pandas as pd
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from pathlib import Path

# --- Configuration ---
DB_HOST = "localhost"
DB_DATABASE = "jakarta_logistics_db"
DB_USER = "root"
DB_PASSWORD = "200801agis" 

TABLE_NAME = "logistics_transactions"
MODEL_FILE = Path("src/best_logistics_model.pkl") # Use Path for robust file handling

# NOTE: CENTER_LAT/LON is removed as we now use Haversine Distance

# --- Utility Function: Calculate Haversine Distance (Used for Feature Engineering) ---
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
# ----------------------------------------------------

def get_logistics_data():
    """Connects to MySQL, executes a query, and returns the results as a Pandas DataFrame."""
    sql_query = f"SELECT * FROM {DB_DATABASE}.{TABLE_NAME}"
    conn = None 
    
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_DATABASE,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        if conn and conn.is_connected():
            print("[STATUS] Successfully connected to MySQL database.")
            # Ensure all columns, including new drop-off coords, are loaded
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
    """Performs feature engineering using new coordinate columns and splits data."""
    
    # 1. Convert date column to datetime object
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    
    # --- NEW CRITICAL FEATURE ENGINEERING ---
    
    # 2. FEATURE 1: Calculate Distance (Haversine) between Pickup and Drop-off
    print("-> Engineering Haversine Distance...")
    df['distance_km'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # 3. FEATURE 2: Traffic Peak Factor (Time-based feature)
    print("-> Engineering Traffic Peak Factor...")
    df['hour'] = df['delivery_date'].dt.hour
    morning_peak = (df['hour'] >= 7) & (df['hour'] <= 9)
    evening_peak = (df['hour'] >= 17) & (df['hour'] <= 19)
    df['traffic_peak_factor'] = np.where(morning_peak | evening_peak, 1.5, 1.0)
    df['is_weekend'] = df['delivery_date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    
    # --- UPDATED INTERACTION FEATURES ---
    
    # 4. Interaction Features (Using the new distance and traffic features)
    # This feature set is now much stronger!
    df['demand_weekend_interaction'] = df['demand_volume'] * df['is_weekend']
    # Replacing the old 'fuel_distance_cost' with the Haversine distance
    df['fuel_distance_cost'] = df['fuel_price_factor'] * df['distance_km']
    # New powerful interaction: Demand is amplified during peak traffic
    df['demand_traffic_interaction'] = df['demand_volume'] * df['traffic_peak_factor']
    
    df['fuel_demand_interaction'] = df['fuel_price_factor'] * df['demand_volume']
    df['fuel_weekend_interaction'] = df['fuel_price_factor'] * df['is_weekend']
    
    # 5. Define Features (X) and Target (y)
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
    
    # Drop columns not needed for modeling
    df.drop(columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'hour'], 
            errors='ignore', inplace=True)
            
    X = df[feature_list]
    y = df['final_price_idr']

    # 6. Data Split (Train, Validation, Test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=69
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=69 
    )
    
    print(f"[STATUS] Data Split: {len(X_train)} Train, {len(X_val)} Validation, {len(X_test)} Test. Total Features: {len(feature_list)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()

def evaluate_model(model, X_data, y_true):
    """Helper function to calculate and return model metrics."""
    y_pred = model.predict(X_data)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def compare_models(X_train, X_val, y_train, y_val):
    
    models = {
        "Linear Regression": LinearRegression(), 
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    print("\n--- Training and Evaluating Initial Models (on Validation Set) ---")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        mae, rmse, r2 = evaluate_model(model, X_val, y_val) 
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score (Val)': r2
        })

    results_df = pd.DataFrame(results).sort_values(by='R2 Score (Val)', ascending=False)
    
    print("\n" + "="*50)
    print("       INITIAL MODEL PERFORMANCE COMPARISON (Validation Set)")
    print("="*50)
    print(results_df.to_string(float_format="%.4f")) 
    print("="*50)
    
    best_initial_model_name = results_df.iloc[0]['Model']
    print(f"\n[RESULT] Best initial model: {best_initial_model_name}.")
    
    return models, results_df

def tune_gradient_boosting(X_train, y_train, X_val, y_val):
    """Performs Grid Search Cross-Validation for Hyperparameter Tuning on GBR."""
    print("\n--- Starting Hyperparameter Tuning (Grid Search) for Gradient Boosting ---")
    
    param_grid = {
        'n_estimators': [100, 200],  
        'learning_rate': [0.1, 0.15], 
        'max_depth': [3, 4] 
    }
    
    gbr = GradientBoostingRegressor(random_state=42)
    
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, 
                               scoring='r2', cv=3, verbose=0, n_jobs=-1)
    
    # Combine Train and Validation for final tuning if needed, but we stick to X_train for simplicity
    grid_search.fit(X_train, y_train) 
    
    best_gbr = grid_search.best_estimator_
    
    # Evaluate the best model on the Validation Set
    mae, rmse, r2 = evaluate_model(best_gbr, X_val, y_val) 
    
    print("\n" + "#"*50)
    print(f"  TUNING COMPLETE: BEST GBR SCORE (R2 on Validation): {r2:.4f}")
    print(f"  Best Parameters Found: {grid_search.best_params_}")
    print("#"*50)
    
    return best_gbr, r2, mae, rmse

def display_feature_importance(model, feature_names):
    """Calculates and prints the feature importance for tree-based models."""
    print("\n" + "*"*50)
    print("         FEATURE IMPORTANCE ANALYSIS (TOP CONTRIBUTORS)")
    print("*"*50)

    # Check if the model supports feature_importances_ (Tree-based models)
    if not hasattr(model, 'feature_importances_'):
        print("Feature importance is only available for Tree-based models (Decision Tree, Gradient Boosting).")
        return

    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df['Importance (%)'] = (feature_importance_df['Importance'] * 100).round(2)
    
    print(feature_importance_df[['Feature', 'Importance (%)']].to_string(index=False))
    print("*"*50)


if __name__ == "__main__":
    
    # Ensure the 'src' directory exists before saving the model
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print("--- STARTING MODEL TRAINING PIPELINE ---")
    
    logistics_df = get_logistics_data()
    
    if logistics_df is not None:
        
        X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_data(logistics_df)
        
        models, results_df = compare_models(X_train, X_val, y_train, y_val)
        
        best_gbr_model, tuned_val_r2, tuned_val_mae, tuned_val_rmse = tune_gradient_boosting(
            X_train, y_train, X_val, y_val
        )
        
        final_best_model = best_gbr_model
        
        # Display feature importance for the tuned GBR model
        display_feature_importance(final_best_model, features) 
        
        # Final evaluation on the unseen Test Set
        final_mae, final_rmse, final_r2 = evaluate_model(final_best_model, X_test, y_test)
        
        print("\n--- FINAL UNBIASED METRICS (ON TEST SET) ---")
        print(f"Model: Gradient Boosting Regressor (Tuned)")
        print(f"MAE: {final_mae:.2f} IDR")
        print(f"RMSE: {final_rmse:.2f} IDR")
        print(f"R2 Score: {final_r2:.4f}")
        
        try:
            joblib.dump(final_best_model, MODEL_FILE)
            print(f"\n[INFO] Best model ({final_best_model.__class__.__name__}) saved to {MODEL_FILE} for API deployment.")
            
            if final_r2 >= 0.85:
                print("\n[SUCCESS] **PHASE I: DATA MODELING GOAL ACHIEVED!** R2 >= 0.85. The new features worked!")
            else:
                print("\n[WARNING] R2 target of 0.85 not yet met. Further feature engineering or data review is needed.")
                
        except Exception as e:
            print(f"[ERROR] Could not save model: {e}")
            
    print("\n--- END OF PIPELINE ---")
