import numpy as np
import random

from tqdm import tqdm

def signal_interval(signal: np.complex64, n_samples=50000, interv=1024) -> np.array:
    '''
    Calcula la matriz donde hay 48 ventanas (filas)
    con 1024 muestras cada una (columnas)
    '''
    n_interv = n_samples // interv
    fft_matrix = np.zeros((n_interv, interv), dtype=np.complex64)
    for i in range(n_interv):
        start_idx = i * interv
        end_idx = (i + 1) * interv
        segment = signal[start_idx:end_idx]
        
        # FFT (frecuency domain transformation)
        fft_matrix[i, :] = np.fft.fft(segment)
        
    return fft_matrix

def energy_arrays(fft_matrix: np.array, n_frec_div:int) -> np.array: 
    '''
    Calcula la matriz de diferencias de energía de un intervalo (fila o ventana)
    con respecto al previo repartido en n_frec_div bloques (columnas) 
    '''
    n_windows = fft_matrix.shape[0]
    length_frec_div = fft_matrix.shape[1] // n_frec_div
    energy_dif = np.zeros((n_windows-1, n_frec_div), dtype=np.float64)
    
    for i in range(n_frec_div):
        energies = np.sum(np.abs(fft_matrix[:,i:i+length_frec_div])**2, axis=1)
        for j in range(1,n_windows):
            energy_dif[j-1,i] = np.log(energies[j]/energies[j-1])
        
    return energy_dif

def split_data(signal_list, train_ratio=0.8):
    '''
    Splits a given dataset in train-test (random)
    '''
    random.shuffle(signal_list)
    
    split_idx = int(len(signal_list) * train_ratio)
    
    train_data = signal_list[:split_idx]
    test_data = signal_list[split_idx:]
    
    return train_data, test_data

def compute_energy_matrix_and_labels(dataset:list, n_samples:int, interv:int, n_frec_div:int, class_mapping:dict):
    '''Builds energy arrays for each train signal 
    (x=window samples, y=frecuency divisions z=signal)'''
    array_length = (n_samples // interv) - 1
    N = len(dataset)
    
    energy_dif_matrix = np.zeros((array_length, n_frec_div, N), dtype=np.float64)
    sample_labels = np.zeros(array_length*N, np.int8)
    for i, signal in tqdm(enumerate(dataset)):
        energy_dif_matrix[:,:,i] = energy_arrays(signal_interval(signal["Data"], n_samples, interv), n_frec_div)
        sample_labels[i*array_length:(i*array_length)+array_length] = class_mapping[signal["Class"]]
        
    return energy_dif_matrix, sample_labels
    
