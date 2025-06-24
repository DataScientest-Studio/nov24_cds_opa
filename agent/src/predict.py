# src/predict.py

#### Fonction dummy pour les premiers tests #### 
import pandas as pd
import joblib
import os

# Assume your trained model is saved in the project root
MODEL_PATH = 'random_forest_model.joblib' 

def _create_dummy_model():
    """
    Creates and saves a dummy RandomForestClassifier for demonstration.
    YOU WILL REPLACE THIS WITH YOUR REAL TRAINED MODEL.
    """
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    print("Creating a dummy RandomForest model...")
    # Dummy data: 3 features, 2 classes
    X_dummy = np.array([
        [100, 50, 1.5, 0.2, 0.3], [120, 60, 1.8, 0.25, 0.35], [80, 40, 1.2, 0.15, 0.28], # Outperform-like
        [50, 20, 0.8, 0.8, 0.1], [60, 25, 0.9, 0.9, 0.12], [40, 15, 0.7, 0.75, 0.09]   # Underperform-like
    ])
    y_dummy = np.array([0, 0, 0, 1, 1, 1]) # 0 for Outperform, 1 for Underperform

    dummy_model = RandomForestClassifier(random_state=42)
    dummy_model.fit(X_dummy, y_dummy)
    joblib.dump(dummy_model, MODEL_PATH)
    print(f"Dummy model saved to {MODEL_PATH}")

# Ensure a model exists for testing purposes
if not os.path.exists(MODEL_PATH):
    _create_dummy_model()


def predict_outperformance(processed_data: pd.DataFrame) -> str:
    """
    Predicts whether a company will outperform or underperform the market next year.
    Loads a pre-trained RandomForestClassifier model.
    """
    print("Loading prediction model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and save your model first.")
    
    model = joblib.load(MODEL_PATH)
    
    print("Making prediction...")
    # The model expects a 2D array, even for a single sample
    # Ensure columns match what the model was trained on
    # For the dummy model: 'Revenue', 'Net Income', 'EPS', 'Debt_to_Equity', 'ROE'
    expected_cols = ['Revenue', 'Net Income', 'EPS', 'Debt_to_Equity', 'ROE']
    
    # Reindex the processed_data to match model's expected feature order and handle missing
    data_for_prediction = processed_data.reindex(columns=expected_cols, fill_value=0)
    
    if data_for_prediction.empty:
        raise ValueError("Processed data is empty, cannot make prediction.")
    
    prediction_proba = model.predict_proba(data_for_prediction.values) # Probabilities
    prediction_class = model.predict(data_for_prediction.values)[0] # The class (0 or 1)
    
    # Map class labels to meaningful strings
    result = "Outperform" if prediction_class == 0 else "Underperform"
    print(f"Raw prediction class: {prediction_class}, Predicted: {result}, Probabilities: {prediction_proba}")
    return result

if __name__ == '__main__':
    # Example usage for testing
    from src.fetch_data import fetch_fundamental_data
    from src.preprocess import preprocess_financial_data
    
    try:
        aapl_raw = fetch_fundamental_data("AAPL")
        aapl_processed = preprocess_financial_data(aapl_raw)
        aapl_prediction = predict_outperformance(aapl_processed)
        print(f"\nAAPL Prediction: {aapl_prediction}")
    except Exception as e:
        print(f"Error predicting for AAPL: {e}")