import numpy as np
import random

from tqdm import tqdm
from collections import Counter
from utils_import import load_data
from utils_preprocess import signal_interval, energy_arrays, split_data, compute_energy_matrix_and_labels
from utils_clustering import create_cluster, cluster_mapping
from sklearn.metrics import accuracy_score

random.seed(1337)

# Asumption: all signals consist of 50k samples
n_samples = 50000
interv = 1024 # Hyperparameter 1
array_length = (n_samples // interv) - 1
n_frec_div = 20 # Hyperparameter 2

# Load data
signals_clean = load_data('../dataset/Jamming/Clean', '../dataset/Jamming/metadata.csv')
signals_narrowband = load_data('../dataset/Jamming/Narrowband', '../dataset/Jamming/metadata.csv')
signals_wideband = load_data('../dataset/Jamming/Wideband', '../dataset/Jamming/metadata.csv')

# Partition train=0.8, test=0.2
clean_train, clean_test = split_data(signals_clean, 0.8)
narrowband_train, narrowband_test = split_data(signals_narrowband, 0.8)
wideband_train, wideband_test = split_data(signals_wideband, 0.8)

train = []
train.extend(clean_train)  
train.extend(narrowband_train)  
train.extend(wideband_train) 
test = [] 
test.extend(clean_test)  
test.extend(narrowband_test) 
test.extend(wideband_test) 

N_train = len(train)
N_test = len(test)

print(f"Nº señales entrenamiento: {N_train}")
print(f"Nº señales test: {N_test}")

random.shuffle(train)
random.shuffle(test)

class_mapping = {"Clean": 0, "Narrowband": 1, "Wideband": 2}

# 1) -- Train --

# Building energy arrays for each train signal (x=window samples, y=frecuency divisions z=signal)
train_energy_dif_matrix, sample_labels = compute_energy_matrix_and_labels(train, n_samples, interv, n_frec_div, class_mapping)

# Creating K-Means model based on energy arrays
cluster = create_cluster(train_energy_dif_matrix, k=3)
print(f"\n--- Centros de cluster ---\n{cluster.cluster_centers_}") 

# Mapping cluster to original classes
cluster_map = cluster_mapping(cluster.labels_, sample_labels, class_mapping)
print(f"\nMapping clusters to predominant classes: {cluster_map}")

# 2) -- Test -- ToDo

test_energy_dif_matrix, sample_labels = compute_energy_matrix_and_labels(test, n_samples, interv, n_frec_div, class_mapping)

'''
#y_true = [class_mapping[signal["class"]] for signal in test]
y_pred = cluster.predict(test_energy_dif_matrix)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy of classification: {accuracy}")
'''