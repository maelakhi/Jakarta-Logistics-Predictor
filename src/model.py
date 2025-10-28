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

# --- Configuration ---
DB_HOST = "localhost"
DB_DATABASE = "jakarta_logistics_db"
DB_USER = "root"
DB_PASSWORD = "200801agis" 

TABLE_NAME = "logistics_transactions"
MODEL_FILE = "src/best_logistics_model.pkl"

CENTER_LAT = -6.2088 
CENTER_LON = 106.8456

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
    """Performs feature engineering and splits data into Train, Validation, and Test sets."""
    
    # 1. Convert date column to datetime object
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    
    # 2. Feature Engineering 
    df['day_of_week'] = df['delivery_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df['distance_from_center'] = np.sqrt(
        (df['pickup_latitude'] - CENTER_LAT)**2 + 
        (df['pickup_longitude'] - CENTER_LON)**2
    )
    
    # --- CRITICAL INTERACTION FEATURES ---
    
    df['demand_weekend_interaction'] = df['demand_volume'] * df['is_weekend']
    df['fuel_distance_cost'] = df['fuel_price_factor'] * df['distance_from_center']
    df['fuel_demand_interaction'] = df['fuel_price_factor'] * df['demand_volume']
    df['fuel_weekend_interaction'] = df['fuel_price_factor'] * df['is_weekend']
    df['fuel_latitude_interaction'] = df['fuel_price_factor'] * df['pickup_latitude']
    # -------------------------------------

    # 3. Define Features (X) and Target (y)
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
    X = df[feature_list]
    y = df['final_price_idr']

    # 4. First split: Separate out the Test Set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=69
    )

    # 5. Second split: Divide the remaining data into Train (60%) and Validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=69 
    )
    
    print(f"[STATUS] Data Split: {len(X_train)} Train, {len(X_val)} Validation, {len(X_test)} Test.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns

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
        
        # Evaluate on the Validation Set
        mae, rmse, r2 = evaluate_model(model, X_val, y_val) 
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score (Val)': r2
        })

    results_df = pd.DataFrame(results).sort_values(by='R2 Score (Val)', ascending=False)
    
    print("\n" + "="*50)
    print("      INITIAL MODEL PERFORMANCE COMPARISON (Validation Set)")
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
    print("        FEATURE IMPORTANCE ANALYSIS (TOP CONTRIBUTORS)")
    print("*"*50)

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
    
    print("--- STARTING MODEL TRAINING PIPELINE ---")
    
    logistics_df = get_logistics_data()
    
    if logistics_df is not None:
        
        # NOTE: X_train, X_val, etc. now have fewer features!
        X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_data(logistics_df)
        
        models, results_df = compare_models(X_train, X_val, y_train, y_val)
        
        best_gbr_model, tuned_val_r2, tuned_val_mae, tuned_val_rmse = tune_gradient_boosting(
            X_train, y_train, X_val, y_val
        )
        
        final_best_model = best_gbr_model
        
        # Display new importance (should be redistributed across the 8 features)
        display_feature_importance(final_best_model, features) 
        
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
                print("\n[SUCCESS] **PHASE I: DATA MODELING GOAL ACHIEVED!** R2 >= 0.85.")
            else:
                print("\n[WARNING] R2 target of 0.85 not yet met. Further feature engineering or data review is needed.")
                
        except Exception as e:
            print(f"[ERROR] Could not save model: {e}")
            
    print("\n--- END OF PIPELINE ---")