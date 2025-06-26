# src/search_ticker.py

import requests
import os
from .fetch_data import APILimitError

FMP_API_KEY = os.getenv("FMP_API_KEY")

def search_ticker(company_name: str) -> str:
    """
    Recherche le ticker le plus pertinent pour un nom d'entreprise donné,
    en priorisant les marchés américains (NYSE, NASDAQ) et la devise USD.
    """
    if not FMP_API_KEY:
        raise ValueError("La clé API FMP_API_KEY n'est pas configurée.")

    BASE_URL = "https://financialmodelingprep.com/api/v3/search"
    # On augmente la limite pour avoir plus de choix
    params = {'query': company_name, 'limit': 10, 'apikey': FMP_API_KEY}

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        results = response.json()
        if not results:
            raise APILimitError(f"Désolé, je n'ai trouvé aucune entreprise correspondant à '{company_name}'.")
        
        # On définit des listes de priorité pour les bourses et les devises
        preferred_exchanges = ["NYSE", "NASDAQ", "PAR"]
        preferred_currency = "USD"
        
        best_ticker = None
        
        # Stratégie 1: On cherche le match parfait (bourse + devise)
        for stock in results:
            if stock.get('exchangeShortName') in preferred_exchanges and stock.get('currency') == preferred_currency:
                best_ticker = stock
                print(f"Match prioritaire trouvé : {best_ticker['symbol']} sur {best_ticker['exchangeShortName']}")
                break # On a trouvé le meilleur, on arrête de chercher

        # Stratégie 2: Si aucun match parfait, on cherche un ticker sur une bourse américaine
        if not best_ticker:
            for stock in results:
                if stock.get('exchangeShortName') in preferred_exchanges:
                    best_ticker = stock
                    print(f"Match de bourse trouvé : {best_ticker['symbol']} sur {best_ticker['exchangeShortName']}")
                    break

        # Stratégie 3: Si toujours rien, on prend le premier résultat comme avant (plan de secours)
        if not best_ticker:
            best_ticker = results[0]
            print(f"Aucun match prioritaire trouvé. Utilisation du premier résultat : {best_ticker['symbol']}")

        final_ticker = best_ticker.get('symbol')
        final_ticker = final_ticker.split('.')[0]  # On retire la partie après le point si elle existe (ex: 'AIR.PA' -> 'AIR')
        found_name = best_ticker.get('name')

        print(f"Ticker sélectionné pour '{company_name}': {final_ticker} ({found_name})")
        return final_ticker

    except requests.exceptions.RequestException as req_err:
        raise APILimitError(f"Impossible de contacter le service de recherche de ticker. Erreur: {req_err}")
    except ValueError as json_err:
        raise APILimitError(f"Réponse invalide reçue du service de recherche. Erreur: {json_err}")