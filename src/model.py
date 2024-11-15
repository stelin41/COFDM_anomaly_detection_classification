import numpy as np
import random

from tqdm import tqdm
from utils_import import load_data
from utils_preprocess import signal_interval, energy_arrays, split_data
from utils_clustering import create_cluster
from sklearn.metrics import accuracy_score

# Asumption: all signals consist of 50k samples
n_samples = 50000
interv = 1024
array_length = (n_samples // interv) - 1

# Load data
signals_clean = load_data('../Jamming/metadata.csv', '../Jamming/Clean')
signals_narrowband = load_data('../Jamming/metadata.csv', '../Jamming/Narrowband')
signals_wideband = load_data('../Jamming/metadata.csv', '../Jamming/Wideband')

# Partition train=0.8, test=0.2
clean_train, clean_test = split_data(signals_clean, 0.8)
narrowband_train, narrowband_test = split_data(signals_narrowband, 0.8)
wideband_train, wideband_test = split_data(signals_wideband, 0.8)

train = []
test = []
train.extend(clean_train, narrowband_train, wideband_train)
test.extend(clean_test, narrowband_test, wideband_test)

N_train = sum(len(signals) for signals in train)
N_test = sum(len(signals) for signals in test)

random.shuffle(train)
random.shuffle(test)

# Building energy arrays for each train signal
train_energy_dif_matrix = np.zeros((N_train, array_length), dtype=np.float64)
for i, signal in tqdm(enumerate(train)):
    train_energy_dif_matrix[i] = energy_arrays(signal_interval(signal["data"], n_samples, interv))
    
# Testing K-Means model based on energy arrays
cluster = create_cluster(train_energy_dif_matrix, k=3)

test_energy_dif_matrix = np.zeros((N_test, array_length), dtype=np.float64)
for i, signal in tqdm(enumerate(test)):
    test_energy_dif_matrix[i] = energy_arrays(signal_interval(signal["data"], n_samples, interv))

# Dictionary in order to map class-cluster_val. This is an ASUMPTION -->  Review when cluster results available
class_mapping = {
    "Clean": 0,
    "Narrowband": 1,
    "Wideband": 2
}

y_true = [class_mapping[signal["class"]] for signal in test]
y_pred = cluster.predict(test_energy_dif_matrix)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy of classification: {accuracy}")