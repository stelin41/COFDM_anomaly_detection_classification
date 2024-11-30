import numpy as np
import random

from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from .utils_import import load_data
from .utils_preprocess import split_data, compute_energy_matrix_and_labels
from .utils_clustering import create_cluster, cluster_mapping
from .utils_test import predict_labels
from .model_rt import *

if __name__ == '__main__':
    random.seed(1337)

    # Asumption: all signals consist of 50k samples
    n_samples = 50000
    interv = 1024 # Hyperparameter 1
    array_length = (n_samples // interv) - 1
    n_frec_div = 16 # Hyperparameter 2

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

    print(f"Nº señales entrenamiento: {len(train)}")
    print(f"Nº señales test: {len(test)}")

    random.shuffle(train)
    random.shuffle(test)

    class_mapping = {"Clean": 0, "Narrowband Start": 1, "Narrowband Stop": 2, "Wideband Start": 3, "Wideband Stop": 4}

    # 1) -- Train --

    # Building energy arrays for each train signal (x=window samples, y=frecuency divisions z=signal)
    train_energy_dif_matrix, sample_labels = compute_energy_matrix_and_labels(train, n_samples, interv, n_frec_div, class_mapping)

    svc_model = svc(train_energy_dif_matrix, sample_labels)


    # 2) -- Test -- 

    test_energy_dif_matrix, y_true = compute_energy_matrix_and_labels(test, n_samples, interv, n_frec_div, class_mapping)

    y_pred = svc_model.predict(test_energy_dif_matrix)

    print(np.bincount(y_pred))

    print("\n-- SVC --")
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc}")        
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))

    realtime_model = RealtimeModel(svc_model, classes = {"Clean": 0, 
                                                        "Narrowband Start": 1, 
                                                        "Narrowband Stop": 2, 
                                                        "Wideband Start": 3, 
                                                        "Wideband Stop": 4}, 
                                            class_map = {0: "Clean", 
                                                        1: "Narrowband", 
                                                        2: "Clean", 
                                                        3: "Wideband", 
                                                        4: "Clean"}, 
                                            class_type = {0: "Clean", 
                                                        1: "Narrowband", 
                                                        2: "Narrowband", 
                                                        3: "Wideband", 
                                                        4: "Wideband"},
                                            offset=4,
                                            nfft=interv, n_partitions=n_frec_div, verbose=True)

    pred = realtime_model.classificate_recordings(test)

    y_true = [s["Class"] for s in test]
    y_hat = [s["Class"] for s in pred]

    # Accuracy
    acc = accuracy_score(y_true, y_hat)
    print(f"\nAccuracy: {acc}")        

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_hat)
    print(f"\nConfusion Matrix:\n{cm}")

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_hat))