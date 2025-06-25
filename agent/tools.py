# tools.py
import pandas as pd
from langchain_core.tools import tool

# --- Import des tools depuis src ---
from src.fetch_data import fetch_fundamental_data as _fetch_data_logic
from src.preprocess import preprocess_financial_data as _preprocess_data_logic
from src.predict import predict_outperformance as _predict_performance_logic
from src.visualize import create_prediction_plot as _visualize_data_logic


# --- LangChain Tools ---
# Le LLM voit les décorateurs. La logique de quel outil doit être appelé
# est gérée par le noeaud `execute_tool` dans `agent.py`.

@tool
def fetch_data(ticker: str) -> str:
    """Récupère les données financières fondamentales pour un ticker boursier donné."""
    return f"[Les données pour {ticker} sont prêtes à être récupérées par le système.]"

@tool
def preprocess_data() -> str:
    """Prépare les données financières récupérées pour la prédiction. Pas d'argument attendu"""
    return "[L'étape de preprocessing est prête à être exécutée.]"

@tool
def predict_performance() -> str:
    """Prédit la performance d'une action en se basant sur les données prétraitées."""
    return "[L'étape de prédiction est prête à être exécutée.]"

@tool
def visualize_data() -> str:
    """Produit une visualisation des données financières et de la prédiction."""
    return "[L'étape de visualisation est prête à être  exécutée.]"

@tool
def display_data() -> str:
    """
    Affiche les données sous forme de tableau.
    Utilise ce tool si l'utilisateur te demande explicitement de 'voir les données', 'montrer un tableau',
    'afficher le DataFrame', ou tout autre requête similaire, APRES avoir récupéré les données, ou les avoir préprocessés.
    """
    return "[Le tableau de données est prêt à être affiché.]"

# C'est cette liste qui est passée à l'agent
available_tools = [
    fetch_data,
    preprocess_data,
    predict_performance,
    visualize_data,
    display_data, 
]