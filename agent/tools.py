# tools.py

import pandas as pd
import plotly.express as px
import plotly.io as pio
from langchain_core.tools import tool
from io import StringIO

# --- Import des logiques de src  ---
from src.search_ticker import search_ticker as _search_ticker_logic
from src.fetch_data import fetch_fundamental_data as _fetch_data_logic
from src.preprocess import preprocess_financial_data as _preprocess_data_logic
from src.predict import predict_outperformance as _predict_performance_logic
from src.fetch_news import fetch_recent_news as _fetch_recent_news_logic
from src.fetch_profile import fetch_company_profile as _fetch_profile_logic

# --- Définition des outils ---
@tool
def search_ticker(company_name: str) -> str:
    """
    Utilise cet outil en PREMIER si l'utilisateur fournit un nom de société (comme 'Apple', 'Microsoft', 'Airbus') 
    au lieu d'un ticker (comme 'AAPL', 'MSFT', 'AIR.PA').
    Cet outil trouve le ticker boursier le plus probable pour un nom d'entreprise.
    
    Args:
        company_name (str): Le nom de l'entreprise à rechercher.
    """
    # La logique réelle est appelée depuis execute_tool_node, ceci est une coquille pour le LLM.
    return "[Le ticker est prêt à être recherché par le système.]"


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


def _create_dynamic_chart_logic(
    data: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str = None
) -> str:
    """Contient la logique de création de graphique, sans être un outil LangChain."""
    try:
        df = data.copy() # On travaille sur une copie
        if 'calendarYear' in df.columns:
            df['calendarYear'] = df['calendarYear'].astype(str)

        if chart_type == 'line':
            fig = px.line(df, x=x_column, y=y_column, title=title, color=color_column, markers=True)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_column, y=y_column, title=title, color=color_column)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column, title=title, color=color_column)
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_column, values=y_column, title=title)
        else:
            return f"Erreur : Le type de graphique '{chart_type}' n'est pas supporté."

        fig.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))
        return pio.to_json(fig)

    except Exception as e:
        # Il est utile de savoir quelle colonne a posé problème
        if isinstance(e, KeyError):
            return f"Erreur: La colonne '{e.args[0]}' est introuvable. Colonnes disponibles: {list(df.columns)}"
        return f"Erreur lors de la création du graphique : {str(e)}"

# L'outil LangChain qui est "vu" par le LLM
@tool
def create_dynamic_chart(
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str = None
) -> str:
    """
    Crée un graphique dynamique et interactif. Les données sont fournies automatiquement.
    Tu DOIS utiliser les noms de colonnes exacts qui te sont fournis dans le contexte actuel.

    Args:
        chart_type (str): Le type de graphique. Supportés : 'line', 'bar', 'scatter', 'pie'.
        x_column (str): Nom exact de la colonne pour l'axe X.
        y_column (str): Nom exact de la colonne pour l'axe Y.
        title (str): Un titre descriptif pour le graphique.
        color_column (str, optional): Nom exact de la colonne pour la couleur.
    """
    return "[L'outil de création de graphique est prêt à être exécuté.]"

@tool
def get_stock_news(ticker: str) -> str:
    """
    Utilise cet outil pour trouver les dernières actualités financières.
    Il fonctionne mieux si on lui fournit à la fois le ticker et le nom de l'entreprise.
    
    Args:
        ticker (str): Le ticker de l'action (ex: 'AAPL').
        company_name (str, optional): Le nom de l'entreprise (ex: 'Apple').
    """
    return "[Les actualités sont prêtes à être récupérées par le système.]"

@tool
def get_company_profile(ticker: str) -> str:
    """
    Utilise cet outil pour obtenir une description générale d'une entreprise.
    Fournit des informations comme le secteur, le CEO, une description de l'activité, le site web et beaucoup d'autres.
    C'est l'outil parfait si l'utilisateur demande "parle-moi de...", "que fait...", ou "qui est..." une entreprise.
    
    Args:
        ticker (str): Le ticker de l'action à rechercher (ex: 'AAPL').
    """
    return "[Le profil de l'entreprise est prêt à être récupéré par le système.]"

# --- La liste complète des outils disponibles pour l'agent ---
available_tools = [
    search_ticker,
    fetch_data,
    get_stock_news,
    get_company_profile,
    preprocess_data,
    predict_performance,
    display_raw_data,
    display_processed_data,
    create_dynamic_chart 
]