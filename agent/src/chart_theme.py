# agent/src/chart_theme.py

# Dictionnaire contenant toutes les préférences graphiques pour les graphiques de Stella.
stella_theme = {

    'colors': [
        '#9467bd',  # Violet Doux
        '#ff7f0e',  # Orange Vif
        '#2ca02c',  # Vert Succès
        '#d62728',  # Rouge Avertissement
        '#1f77b4',  # Bleu Stella (Bleu moyen)
        '#8c564b',  # Marron Terre
        '#e377c2',  # Rose
    ],

    # Des couleurs spécifiques pour certaines métriques clés.
    # Utilisé pour le graphique  de synthèse.
    'metric_colors': {
        'roe': '#2ca02c',               # Le ROE est un signe de rentabilité -> Vert
        'debtToEquity': '#d62728',      # La dette est un risque -> Rouge
        'earningsYield': '#1f77b4',     # Le rendement est une info neutre -> Bleu
        'marginProfit': '#9467bd',      # La marge est une info de performance -> Violet
    },
    
    # Le modèle de base pour les graphiques.
    'template': 'plotly_white',
    
    # La police de caractères pour tous les textes du graphique.
    'font': {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': '#333333' # Une couleur de texte sombre mais pas noire pure
    }
}