# src/visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

def create_prediction_plot(ticker: str, data: pd.DataFrame, prediction: str) -> str:
    """
    Crée une visualisation des données financières et de la prédiction pour un ticker donné.
    Retourne la chaîne de caractères encodée en base64 de l'image générée.
    """
    print(f"Création de la visualisation pour {ticker} avec le verdict : {prediction}...")

    if data.empty:
        raise ValueError("Aucune donnée à visualiser.")

    # --- MODIFICATION CLÉ ---
    # On choisit des indicateurs pertinents parmi les nouvelles colonnes disponibles.
    # Assurez-vous que ces colonnes existent bien dans vos données finales.
    metrics_to_plot = ['roe', 'debtToEquity', 'earningsYield', 'marginProfit']
    
    # On vérifie que les colonnes existent avant de les utiliser
    available_metrics = [col for col in metrics_to_plot if col in data.columns]
    if not available_metrics:
        raise ValueError(f"Aucune des colonnes de visualisation ({metrics_to_plot}) n'a été trouvée dans les données.")

    plot_data = data[available_metrics].iloc[-1] # On prend la dernière année

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen', 'plum'])
    
    # On ajuste le titre pour refléter la nouvelle logique de prédiction
    if prediction == "Risque Élevé Détecté":
        title_prediction = "Verdict : Risque Élevé Détecté ⚠️"
        fig.patch.set_facecolor('#fff0f0') # Fond légèrement rouge en cas d'alerte
    else:
        title_prediction = "Verdict : Aucun Risque Extrême Détecté"
        fig.patch.set_facecolor('#f0fff0') # Fond légèrement vert sinon

    ax.set_title(f"Indicateurs Clés pour {ticker} (Dernière Année)\n{title_prediction}", fontsize=14)
    ax.set_ylabel('Valeur', fontsize=12)
    ax.set_xlabel('Indicateur', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    print(f"Visualisation créée pour {ticker}.")
    return image_base64

# ... (le reste du fichier, if __name__ == '__main__':, peut rester tel quel pour vos tests)

if __name__ == '__main__':
    # Example usage for testing
    from src.fetch_data import fetch_fundamental_data
    try:
        aapl_raw = fetch_fundamental_data("AAPL")
        # In a real scenario, you'd pass the actual prediction here
        base64_image = create_prediction_plot("AAPL", aapl_raw, "Outperform")
        # To verify, you could decode and save this to a file:
        # with open("test_plot.png", "wb") as f:
        #     f.write(base64.b64decode(base64_image))
        print("\nAAPL Plot created (base64 string returned).")
    except Exception as e:
        print(f"Error creating plot for AAPL: {e}")