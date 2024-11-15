import numpy as np
from sklearn.cluster import KMeans

def create_cluster(energy_dif_matrix: np.array, k=3):
    kmeans = KMeans(n_clusters=k, random_state=1337, n_init="auto").fit(energy_dif_matrix)
    return kmeans