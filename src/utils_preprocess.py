import numpy as np
import random

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

def energy_arrays(fft_matrix: np.array) -> np.array: 
    '''
    Calcula el array de diferencias de energía de un intervalo (ventana)
    con respecto al previo
    '''
    n_windows = fft_matrix.shape[0]
    energy_dif = np.zeros(n_windows-1, dtype=np.float64)

    energies = np.sum(np.abs(fft_matrix)**2, axis=1)
    for i in range(1,n_windows):
        energy_dif[i-1] = np.log(energies[i]/energies[i-1])
        
    return np.array(energy_dif)

def split_data(signal_list, train_ratio=0.8):
    random.shuffle(signal_list)
    
    split_idx = int(len(signal_list) * train_ratio)
    
    train_data = signal_list[:split_idx]
    test_data = signal_list[split_idx:]
    
    return train_data, test_data