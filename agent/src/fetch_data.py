# src/fetch_data.py 

#### Fonction dummy pour les premiers tests #### 
import pandas as pd
import random

def fetch_fundamental_data(ticker: str) -> pd.DataFrame:
    """
    Fetches fundamental financial data for a given stock ticker from an API.
    
    Use this tool as the very first step when a user asks to 'analyze', 
    'look up', 'get data for', or 'investigate' a specific stock ticker.
    For example, if the user says 'Analyze AAPL', you should call this tool
    with the ticker 'AAPL'.
    """
    print(f"Fetching data for {ticker}...")
    if ticker.upper() == "AAPL":
        # Simulate some data for Apple
        data = {
            'Revenue': [387.5, 394.3, 383.3],
            'Net Income': [94.7, 99.8, 97.0],
            'EPS': [5.85, 6.11, 5.95],
            'Debt_to_Equity': [1.5, 1.6, 1.7],
            'ROE': [1.65, 1.70, 1.68]
        }
        df = pd.DataFrame(data, index=[2021, 2022, 2023])
    elif ticker.upper() == "GOOGL":
        # Simulate some data for Google
        data = {
            'Revenue': [257.6, 282.8, 305.6],
            'Net Income': [76.0, 59.9, 68.7],
            'EPS': [5.61, 4.56, 5.25],
            'Debt_to_Equity': [0.1, 0.1, 0.1],
            'ROE': [0.25, 0.22, 0.23]
        }
        df = pd.DataFrame(data, index=[2021, 2022, 2023])
    else:
        # Simulate a random chance of data existing for other tickers
        if random.random() < 0.7: # 70% chance of data
            data = {
                'Revenue': [random.uniform(50, 500) for _ in range(3)],
                'Net Income': [random.uniform(10, 100) for _ in range(3)],
                'EPS': [random.uniform(1, 10) for _ in range(3)],
                'Debt_to_Equity': [random.uniform(0.1, 2.0) for _ in range(3)],
                'ROE': [random.uniform(0.1, 0.5) for _ in range(3)]
            }
            df = pd.DataFrame(data, index=[2021, 2022, 2023])
        else:
            raise ValueError(f"No fundamental data found for ticker: {ticker.upper()}")
    
    print(f"Fetched data for {ticker}:\n{df.head()}")
    return df

if __name__ == '__main__':
    # Example usage for testing
    try:
        aapl_data = fetch_fundamental_data("AAPL")
        print("\nAAPL Data Fetched Successfully!")
    except ValueError as e:
        print(f"Error fetching AAPL data: {e}")

    try:
        xyz_data = fetch_fundamental_data("XYZ")
        print("\nXYZ Data Fetched Successfully!")
    except ValueError as e:
        print(f"Error fetching XYZ data: {e}")