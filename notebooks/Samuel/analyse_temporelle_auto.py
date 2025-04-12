import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from matplotlib.patches import Rectangle
import itertools
import os
import io
from PIL import Image
import matplotlib.dates as mdates
import gc

def custom_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def process_pair(df, ticker_1, ticker_2, output_dir='dtw_results'):
    dfclean = df[['date', ticker_1, ticker_2]].dropna()

    # Extraire les séries
    series_1 = np.ravel(dfclean[ticker_1].values).astype(float)
    series_2 = np.ravel(dfclean[ticker_2].values).astype(float)

    series_1_norm = series_1 / np.max(np.abs(series_1))
    series_2_norm = series_2 / np.max(np.abs(series_2))
    d_date = dfclean['date']

    # 1. Calcul DTW
    distance, path = fastdtw(series_1_norm, series_2_norm, dist=custom_euclidean)
    normalization_ratio = distance / (len(series_1) + len(series_2))

    # 2. Création de la figure
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))

    # 3. Plot principal : Séries alignées
    ax1.plot(d_date, series_1_norm, label=ticker_1, color='#1f77b4', linewidth=2)
    ax1.plot(d_date, series_2_norm, label=ticker_2, color='#ff7f0e', linewidth=2)

    # Ajout des connexions DTW (échantillonnées)
    for (i, j) in path[::max(1, len(path)//100)]:
        ax1.plot([d_date[i], d_date[j]], [series_1_norm[i], series_2_norm[j]],
                 color='gray', alpha=0.1, linestyle='-')

    # 4. Matrice de distance
    distance_matrix = np.abs(np.subtract.outer(series_1_norm, series_2_norm))
    im = ax2.imshow(distance_matrix, cmap='YlOrRd', origin='lower')
    plt.colorbar(im, ax=ax2, label='Distance')
    ax2.plot([p[1] for p in path], [p[0] for p in path], color='black', linewidth=0.5)

    # 5. Boîte à métriques
    metrics_text = f"""Distance DTW: {distance:.2f}
Ratio: {normalization_ratio:.4f}
Alignements: {len(path)} points"""
    ax3.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, color='whitesmoke'))
    ax3.text(0.5, 0.7, metrics_text, ha='center', va='center', fontsize=12)
    ax3.axis('off')

    # 6. Paramètres visuels
    ax1.set_title(f"Dynamic Time Warping: {ticker_1} vs {ticker_2}", pad=20)
    ax1.legend(loc='upper right')
    ax2.set_xlabel(ticker_2)
    ax2.set_ylabel(ticker_1)
    ax2.set_title("Matrice de distance")

    # 7. Sauvegarde et fermeture
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{ticker_1}_vs_{ticker_2}_dtw.jpg".replace("value_", "")
    final_path = os.path.join(output_dir, filename)

    # Méthode fiable pour l'enregistrement JPG
    plt.tight_layout()

    # 1. Sauvegarde en PNG dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close("all")

    # 2. Conversion PNG->JPG avec PIL
    buf.seek(0)
    img = Image.open(buf)
    rgb_img = img.convert('RGB')  # Conversion nécessaire pour JPG
    rgb_img.save(final_path, 'JPEG', quality=85, optimize=True)
    buf.close()
    gc.collect()

    return {
        'pair': f"{ticker_1} vs {ticker_2}",
        'distance': distance,
        'normalization_ratio': normalization_ratio,
        'alignment_points': len(path)
    }


# Chargement des données
df = pd.read_csv("merged_historical_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Exclure la première colonne et garder seulement les colonnes 'value_...'
columns_to_compare = [col for col in df.columns if col != df.columns[0] and col.startswith('value_')]

# Créer toutes les combinaisons possibles de paires
pairs = list(itertools.combinations(columns_to_compare, 2))
matplotlib.use('Agg')

# Traiter toutes les paires et stocker les résultats
results = []
for ticker_1, ticker_2 in pairs:
    try:
        result = process_pair(df, ticker_1, ticker_2)
        plt.close("all")
        gc.collect()
        results.append(result)
        print(f"Processed {ticker_1} vs {ticker_2}")
    except Exception as e:
        print(f"Error processing {ticker_1} vs {ticker_2}: {str(e)}")

# Créer un dataframe avec les résultats et le sauvegarder
results_df = pd.DataFrame(results)
results_df.to_csv("dtw_results/summary_results.csv", index=False)

print("Toutes les comparaisons ont été effectuées avec succès!")