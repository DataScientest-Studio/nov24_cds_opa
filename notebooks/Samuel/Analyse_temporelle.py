import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import euclidean  # Import explicite
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

df = pd.read_csv("merged_historical_data.csv")
ticker_1 = "value_TSLA"
ticker_2 = "value_AAPL"
dfclean = df[[ticker_1, ticker_2]].dropna()
dfclean.head()

# Extraire les séries
series_1 = np.ravel(dfclean[ticker_1].values).astype(float)
series_2 = np.ravel(dfclean[ticker_2].values).astype(float)

series_1_norm = series_1 / np.max(np.abs(series_1))
series_2_norm = series_2 / np.max(np.abs(series_2))

###Utilisation de cust_euclidean; je ne sais pas pourquoi, si j'utilise la fonction standard, j'ai un bug
def custom_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 1. Calcul DTW
distance, path = fastdtw(series_1_norm, series_2_norm, dist=custom_euclidean)
normalization_ratio = distance / (len(series_1) + len(series_2))

# 2. Création de la figure
plt.figure(figsize=(14, 10))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((3, 2), (2, 0))
ax3 = plt.subplot2grid((3, 2), (2, 1))

# 3. Plot principal : Séries alignées
ax1.plot(series_1_norm, label=ticker_1, color='#1f77b4', linewidth=2)
ax1.plot(series_2_norm, label=ticker_2, color='#ff7f0e', linewidth=2)

# Ajout des connexions DTW (échantillonnées)
for (i, j) in path[::max(1, len(path)//100)]:
    ax1.plot([i, j], [series_1_norm[i], series_2_norm[j]],
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

# 7. Ajustements finaux
plt.tight_layout()
plt.savefig('dtw_analysis.png', dpi=300)
plt.show()