import numpy as np
from sklearn.cluster import KMeans

def create_cluster(energy_dif_matrix: np.array, k=3):
    print(energy_dif_matrix.shape)
    n_samples, n_frec_div, n_signals = energy_dif_matrix.shape
    
    # Aplanar dimensión z
    flattened_matrix = energy_dif_matrix.reshape(n_samples*n_signals, n_frec_div)
    print(flattened_matrix.shape)
    #kmeans = KMeans(n_clusters=k, random_state=1337, n_init="auto").fit(flattened_matrix)
    #return kmeans