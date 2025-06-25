# src/preprocess.py

#### Fonction dummy pour les premiers tests #### 
import pandas as pd

def preprocess_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Préprocesse des données financières brutes en features conformes pour le modèle de prédiction.
    """
    df['index'] = df.symbol + '_' + df.calendarYear.astype('string')
    df = df.set_index('index')
    df['marginProfit'] = df['netIncomePerShare'] / df['revenuePerShare']
    df = df.sort_values(by='calendarYear')
    df['revenuePerShare_YoY_Growth'] = ((df['revenuePerShare'] / df['revenuePerShare'].shift(1)) - 1) * 100
    df = df[['marketCap', 'marginProfit', 'roe', 'roic', 'revenuePerShare', 'debtToEquity', 'revenuePerShare_YoY_Growth', 'earningsYield']]
    processed_df = df.dropna()

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