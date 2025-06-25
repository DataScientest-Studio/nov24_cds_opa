# tools.py

import pandas as pd
import plotly.express as px
import plotly.io as pio
from langchain_core.tools import tool

# --- Import des logiques de src  ---
from src.fetch_data import fetch_fundamental_data as _fetch_data_logic
from src.preprocess import preprocess_financial_data as _preprocess_data_logic
from src.predict import predict_outperformance as _predict_performance_logic

# --- Définition des outils ---

@tool
def fetch_data(ticker: str) -> str:
    """Récupère les données financières fondamentales pour un ticker boursier donné."""
    return f"[Les données pour {ticker} sont prêtes à être récupérées par le système.]"

@tool
def preprocess_data() -> str:
    """Prépare les données financières récupérées pour la prédiction."""
    return "[L'étape de preprocessing est prête à être exécutée.]"

@tool
def predict_performance() -> str:
    """Prédit la performance d'une action en se basant sur les données prétraitées."""
    return "[L'étape de prédiction est prête à être exécutée.]"

@tool
def display_raw_data() -> str:
    """Affiche le tableau de données financières brutes qui ont été initialement récupérées."""
    return "[Le tableau de données brutes est prêt à être affiché.]"

@tool
def display_processed_data() -> str:
    """Affiche le tableau de données financières traitées et nettoyées, prêtes pour l'analyse."""
    return "[Le tableau de données traitées est prêt à être affiché.]"


@tool
def create_dynamic_chart(
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str = None,
) -> str:
    """
    Crée un graphique dynamique et interactif avec Plotly. Les données sont fournies automatiquement.

    IMPORTANT : Le nom de la colonne que tu fournis pour `y_column` DOIT correspondre EXACTEMENT
    à l'un des noms de la liste de colonnes disponibles qui te sera fournie dans le contexte de la conversation.
    Ne traduis pas et n'invente pas de noms de colonnes.

    Types de graphiques possibles :
    - 'line' pour les données chronologiques (ex: évolution d'une métrique sur plusieurs années).
    - 'bar' pour comparer des catégories ou des valeurs à un instant T.
    
    Args:
        chart_type (str): Le type de graphique à créer.
        x_column (str): Le nom de la colonne pour l'axe des X (généralement 'year' pour les graphiques en ligne).
        y_column (str): Le nom EXACT de la colonne pour l'axe des Y, choisi depuis la liste fournie.
        title (str): Un titre descriptif pour le graphique.
        color_column (str, optional): La colonne pour colorer les éléments du graphique.
    """
    pass


# --- La liste complète des outils disponibles pour l'agent ---
available_tools = [
    fetch_data,
    preprocess_data,
    predict_performance,
    create_dynamic_chart,   # L'unique outil de visualisation
    display_raw_data,
    display_processed_data,
]