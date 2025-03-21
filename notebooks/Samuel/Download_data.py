import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import time

# Charger les variables d'environnement (pour la clé API)
load_dotenv()

def get_historical_price(symbol, from_date, to_date, api_key):
    """
    Récupère les données historiques des prix pour un symbole donné.

    Args:
        symbol (str): Le symbole de l'action ou ETF
        from_date (str): Date de début au format YYYY-MM-DD
        to_date (str): Date de fin au format YYYY-MM-DD
        api_key (str): Clé API pour Financial Modeling Prep

    Returns:
        pandas.DataFrame: DataFrame contenant les données historiques
    """
    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full/"
    url = f"{base_url}{symbol}?from={from_date}&to={to_date}&apikey={api_key}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "historical" in data:
            historical_data = data["historical"]
            df = pd.DataFrame(historical_data)
            # Convertir la colonne 'date' en datetime et la définir comme index
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')  # Trier par date croissante
            return df
        else:
            print(f"Pas de données historiques pour {symbol}")
            return None
    else:
        print(f"Erreur de requête pour {symbol}: {response.status_code}")
        return None

# Obtenir la clé API depuis les variables d'environnement
api_key = ""

# Paramètres pour la requête
from_date = "2020-01-01"
to_date = "2024-12-31"

# Liste des tickers à récupérer
tickers = ["AAPL"]

# Créer un dossier pour stocker les données si nécessaire
output_dir = "historical_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Récupérer les données pour chaque ticker
for ticker in tickers:
    print(f"\nRécupération des données pour {ticker}...")

    df = get_historical_price(ticker, from_date, to_date, api_key)

    if df is not None:
        # Sauvegarder les données dans un fichier CSV
        csv_filename = os.path.join(output_dir, f"{ticker}_historical_data_{from_date}_to_{to_date}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"✓ Données sauvegardées dans {csv_filename}")
        print(f"  Nombre de jours: {len(df)}")
        print(f"  Première date: {df['date'].min().strftime('%Y-%m-%d')}")
        print(f"  Dernière date: {df['date'].max().strftime('%Y-%m-%d')}")
    else:
        print(f"✗ Échec de récupération des données pour {ticker}")

    # Ajouter un délai pour éviter de surcharger l'API
    time.sleep(1)

print("\nTraitement terminé pour tous les tickers.")

