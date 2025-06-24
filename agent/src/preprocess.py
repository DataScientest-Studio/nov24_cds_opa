# src/preprocess.py

#### Fonction dummy pour les premiers tests #### 
import pandas as pd

def preprocess_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses raw financial data into features suitable for the prediction model.
    This is a placeholder for your actual data cleaning and feature engineering.
    """
    print("Preprocessing data...")
    # Example: Ensure all expected columns are present, fill NaNs, etc.
    # For this example, we'll just ensure it's numerical and ready
    
    # Simulate selecting/creating features that your RF model expects
    # For example, if your model uses 'Revenue_Growth', 'Net_Income_Margin', 'Current_Ratio'
    # you would calculate them here.
    
    # For simplicity, let's assume the model uses 'Revenue', 'Net Income', 'EPS', 'Debt_to_Equity', 'ROE' directly
    # And we just take the latest year's data as input for prediction
    
    if df.empty:
        raise ValueError("Cannot preprocess empty DataFrame.")

    # Get the latest year's data
    processed_df = df.iloc[-1:].copy()

    # Ensure numerical types
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Handle any NaNs if necessary (e.g., fill with 0 or mean, depending on your model)
    processed_df = processed_df.fillna(0)

    print(f"Preprocessed data:\n{processed_df.head()}")
    return processed_df

if __name__ == '__main__':
    # Example usage for testing
    from src.fetch_data import fetch_fundamental_data
    try:
        aapl_raw = fetch_fundamental_data("AAPL")
        aapl_processed = preprocess_financial_data(aapl_raw)
        print("\nAAPL Data Preprocessed Successfully!")
    except Exception as e:
        print(f"Error preprocessing AAPL data: {e}")