import numpy as np
import random
from scipy.signal import welch
from scipy.fft import fftshift

from tqdm import tqdm

def signal_interval(signal: np.complex64, n_samples=50000, interv=1024, Fs=12000000) -> np.array:
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

        f, Pxx_spec = welch(segment, Fs, nperseg=interv, return_onesided=False, scaling="density")
        Pxx_spec_dB = 10 * np.log10(Pxx_spec)
        fft_matrix[i, :] = fftshift(Pxx_spec_dB)

        # FFT (frecuency domain transformation)
        #fft_matrix[i, :] = np.fft.fft(segment)
        
    return fft_matrix

def energy_arrays(fft_matrix: np.array, n_frec_div:int, offset=1) -> np.array: 
    '''
    Calcula la matriz de diferencias de energía de un intervalo (fila o ventana)
    con respecto al previo repartido en n_frec_div bloques (columnas) 
    '''
    n_windows = fft_matrix.shape[0]-offset
    length_frec_div = fft_matrix.shape[1] // n_frec_div
    energy_dif = np.zeros((n_windows, n_frec_div), dtype=np.float64)
    
    for i in range(n_frec_div):
        energies = np.sum(np.abs(fft_matrix[:,i*length_frec_div:(i+1)*length_frec_div])**2, axis=1)
        energy_dif[:,i] = np.log(energies[offset:]/energies[:-offset])
        
    return np.abs(energy_dif)

def split_data(signal_list, train_ratio=0.8):
    '''
    Splits a given dataset in train-test (random)
    '''
    random.shuffle(signal_list)
    
    split_idx = int(len(signal_list) * train_ratio)
    
    train_data = signal_list[:split_idx]
    test_data = signal_list[split_idx:]
    
    return train_data, test_data

def compute_energy_matrix_and_labels(dataset:list, n_samples:int, interv:int, n_frec_div:int, class_mapping:dict, anomaly_duration=12500):
    '''Builds energy arrays for each train signal 
    (x=window samples, y=frecuency divisions z=signal)'''
    array_length = (n_samples // interv) - 1
    N = len(dataset)
    
    energy_dif_matrix = np.empty((array_length*N, n_frec_div), dtype=np.float64)
    sample_labels = np.empty(array_length*N, np.int8)
    sample_labels[:] = class_mapping["Clean"]
    for i, signal in tqdm(enumerate(dataset)):
        energy_dif_matrix[array_length*i:array_length*(i+1),:] = energy_arrays(signal_interval(signal["Data"], n_samples, interv), n_frec_div)
        #sample_labels[i*array_length:(i*array_length)+array_length] = class_mapping[signal["Class"]]
        if signal["Class"]!="Clean":
            start = i*array_length + (signal['JammingStartTime']//interv)
            stop = i*array_length + (signal['JammingStartTime']+anomaly_duration)//interv + 1
            sample_labels[start] = class_mapping[signal["Class"]]
            sample_labels[stop] = class_mapping[signal["Class"]]
        
    return energy_dif_matrix, sample_labels
    
