import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_correlation_heatmap():
    """
    Charge les données fusionnées et crée une heatmap de corrélation.
    """
    try:
        # Essayer d'abord de charger la matrice de corrélation si elle existe déjà
        try:
            corr_matrix = pd.read_csv("ticker_correlations.csv", index_col=0)
            print("Matrice de corrélation chargée depuis le fichier existant.")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # Si la matrice n'existe pas, charger les données fusionnées et calculer la corrélation
            print("Chargement des données fusionnées...")
            df = pd.read_csv("merged_historical_data.csv")

            # Identifier les colonnes de valeurs (celles qui commencent par 'value_')
            value_cols = [col for col in df.columns if col.startswith('value_')]

            if not value_cols:
                print("Erreur: Aucune colonne commençant par 'value_' trouvée dans les données.")
                return

            # Calculer la matrice de corrélation
            corr_matrix = df[value_cols].corr()
            print("Matrice de corrélation calculée.")

        # Nettoyer les noms des tickers (enlever le préfixe 'value_')
        corr_matrix.index = [idx.replace('value_', '') for idx in corr_matrix.index]
        corr_matrix.columns = [col.replace('value_', '') for col in corr_matrix.columns]

        # Configurer la figure avec une taille appropriée
        plt.figure(figsize=(12, 10))

        # Définir la palette de couleurs - une palette divergente est bonne pour les corrélations
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Créer la heatmap
        sns.heatmap(corr_matrix,
                    annot=False,           # Afficher les valeurs numériques
                    fmt=".2f",            # Format à 2 décimales
                    cmap=cmap,            # Palette de couleurs
                    vmin=-1, vmax=1,      # Échelle de valeurs
                    center=0,             # Centre de la palette de couleurs
                    square=True,          # Cellules carrées
                    linewidths=.5,        # Largeur des lignes entre cellules
                    cbar_kws={"shrink": .8})

        plt.title('Matrice de Corrélation entre les Tickers', fontsize=16, pad=20)
        plt.tight_layout()

        # Sauvegarder la figure
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Heatmap de corrélation sauvegardée dans 'correlation_heatmap.png'")

        # Afficher la figure
        plt.show()

    except Exception as e:
        print(f"Erreur lors de la création de la heatmap: {e}")

if __name__ == "__main__":
    print("Création de la heatmap de corrélation...")
    create_correlation_heatmap()
    print("Terminé.")