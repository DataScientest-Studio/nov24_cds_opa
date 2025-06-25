# src/fetch_data.py
import requests
import pandas as pd
import os


FMP_API_KEY = os.getenv("FMP_API_KEY")

def fetch_fundamental_data(ticker: str) -> pd.DataFrame:
    """
    Récupère les données fondamentales d'une action à partir de l'API de Financial Modeling Prep selon le ticker
    renseigné par l'utilisateur.
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3/key-metrics/"
    url = f"{BASE_URL}{ticker}?period=annual&apikey={FMP_API_KEY}"

    response = requests.get(url)
    if response.status_code == 200: # Si la requête est OK (succès)
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"Erreur de requête pour {ticker}: {response.status_code}")
        return None

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