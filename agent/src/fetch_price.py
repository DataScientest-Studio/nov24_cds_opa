# agent/src/fetch_price.py

import requests
import pandas as pd
import os
import json
from .fetch_data import APILimitError # On réutilise notre exception

FMP_API_KEY = os.getenv("FMP_API_KEY")

def fetch_price_history(ticker: str, period_days: int = 252) -> pd.DataFrame:
    """
    Récupère l'historique des prix de clôture pour un ticker sur une période donnée.
    
    Args:
        ticker (str): Le ticker de l'action.
        period_days (int): Le nombre de jours dans le passé à récupérer (252 ~ 1 an de bourse).
        
    Returns:
        pd.DataFrame: Un DataFrame avec 'date' en index et 'close' en colonne.
    """
    if not FMP_API_KEY:
        raise ValueError("La clé API FMP_API_KEY n'est pas configurée.")
    
    # L'endpoint "historical-price-full" est plus fiable.
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries={period_days}&apikey={FMP_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Vérifier si la réponse contient les données attendues
        if not data or 'historical' not in data or not data['historical']:
            raise ValueError(f"Aucun historique de prix trouvé pour le ticker '{ticker}'. Il est peut-être invalide.")
            
        df = pd.DataFrame(data['historical'])
        
        # On ne garde que les colonnes utiles et on les renomme pour la clarté
        df = df[['date', 'close']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # On trie par date pour s'assurer que le graphique est correct
        return df.sort_index(ascending=True)

    except requests.exceptions.HTTPError as e:
        raise APILimitError(f"Erreur API en récupérant l'historique des prix pour {ticker}: {e.response.text}")
    except (ValueError, KeyError) as e:
        # Gère les tickers invalides ou les réponses mal formées
        raise ValueError(f"Impossible de traiter les données de prix pour {ticker}: {e}")

if __name__ == '__main__':
    # Test rapide
    try:
        aapl_prices = fetch_price_history("AAPL")
        print("Historique des prix pour AAPL:")
        print(aapl_prices.head())
    except Exception as e:
        print(f"Erreur: {e}")