import numpy as np
import random

from tqdm import tqdm
from utils_import import load_data
from utils_preprocess import signal_interval, energy_arrays, split_data
from utils_clustering import create_cluster
from sklearn.metrics import accuracy_score

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

# Building energy arrays for each train signal (x=window samples, y=frecuency divisions z=signal)
train_energy_dif_matrix = np.zeros((array_length, n_frec_div, N_train), dtype=np.float64)
for i, signal in tqdm(enumerate(train)):
    train_energy_dif_matrix[:,:,i] = energy_arrays(signal_interval(signal["Data"], n_samples, interv), n_frec_div)
    
# Testing K-Means model based on energy arrays
cluster = create_cluster(train_energy_dif_matrix, k=3)
'''
test_energy_dif_matrix = np.zeros((array_length, n_frec_div, N_test), dtype=np.float64)
for i, signal in tqdm(enumerate(test)):
    test_energy_dif_matrix[:,:,i] = energy_arrays(signal_interval(signal["data"], n_samples, interv), n_frec_div)

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
'''