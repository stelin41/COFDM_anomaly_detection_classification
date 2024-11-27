import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def create_cluster(energy_dif_matrix: np.array, k=5) -> KMeans:
    kmeans = KMeans(n_clusters=k, random_state=1337, n_init="auto").fit(energy_dif_matrix)
    return kmeans

def cluster_mapping(cluster_labels: np.array, sample_labels: np.array, class_mapping: dict) -> dict:
    cluster_map = {}
    for cluster_id in np.unique(cluster_labels):
        class_counts = Counter(sample_labels[np.where(cluster_labels == cluster_id)[0]])
        print(f"\nClases en el cluster {cluster_id}: {class_counts}")
        maj_class = class_counts.most_common(1)[0][0]
        label = next((k for k,v in class_mapping.items() if v == maj_class), "None")
        print(f"Clase mayoritaria: {maj_class}, {label}")
        cluster_map[cluster_id] = maj_class
    return cluster_map