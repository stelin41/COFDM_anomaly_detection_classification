import numpy as np
import random
from scipy.signal import welch
from scipy.fft import fftshift
from imblearn.under_sampling import RandomUnderSampler

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

    # the number of samples in each fft interval should be divisible by the number of partitions
    assert (fft_matrix.shape[1] % n_frec_div) == 0 

    n_windows = fft_matrix.shape[0]-offset
    length_frec_div = fft_matrix.shape[1] // n_frec_div
    energy_dif = np.zeros((n_windows, n_frec_div), dtype=np.float64)
    
    for i in range(n_frec_div):
        energies = np.sum(np.abs(fft_matrix[:,i*length_frec_div:(i+1)*length_frec_div])**2, axis=1)
        energy_dif[:,i] = np.log(energies[offset:]/energies[:-offset])
    
    #energy_dif = np.abs(energy_dif)
    #mins = np.min(energy_dif, axis=1)
    #energy_dif = energy_dif-mins[...,None]
    return np.abs(energy_dif)
    #return energy_dif

def split_data(signal_list, train_ratio=0.8):
    '''
    Splits a given dataset in train-test (random)
    '''
    random.shuffle(signal_list)
    
    split_idx = int(len(signal_list) * train_ratio)
    
    train_data = signal_list[:split_idx]
    test_data = signal_list[split_idx:]
    
    return train_data, test_data

def balance(X:np.array, y:np.array, random_state=1337):
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X, y) # returns X, y
    #count = np.bincount(y)
    #size = np.min(count)
    #out = np.empty((size*len(count), X.shape[1]+y.shape[0]), dtype=np.float64)
    #for i, c in enumerate(np.unique(y)):
    #    out[i*size:(i+1)*size, 1:] = np.random.choice(X[y==c, :],size=size, replace=False)
    #    out[i*size:(i+1)*size, 1] = c
    #out = np.random.shuffle(out)
    #return out[:, 1:], out[:, 1] # X, y


def compute_energy_matrix_and_labels(dataset:list, n_samples:int, interv:int, 
                                     n_frec_div:int, class_mapping:dict, 
                                     anomaly_duration = 12500, offset = 2, 
                                     label_offset = 1, remove_middle = True):
    '''
    Builds energy arrays for each train signal 
    (x=window samples, y=frecuency divisions z=signal)
    
    Explanation (example):

            ********ANOMALY*********
      ______|________________X_____|____
       ^--offset--^
            ^--label_offset--^
    -----------------time---------------->

    (X is the interval labeled as not 'Clean')

    label_offset is how many intervals it skips since the start of the anomaly
    offset is how many invervals it skips to compute the energy change metric
    remove_middle indicates if it should remove labels that could be assigned 'Clean' 
                  but could be labeled as not 'Clean'
    
    This is done because the anomaly can start at any point inside an interval,
    and it may not last enougth to be noticeable.
    
    '''
    n_bad_data = offset if remove_middle else 0
    skip = 0 if remove_middle else label_offset
    array_length = (n_samples // interv) - offset - (n_bad_data*2)

    N = len(dataset)
    
    energy_dif_matrix = np.empty((array_length*N, n_frec_div), dtype=np.float64)
    sample_labels = np.empty(array_length*N, np.int8)
    sample_labels[:] = class_mapping["Clean"]
    for i, signal in tqdm(enumerate(dataset)):
        energy_dif = energy_arrays(signal_interval(signal["Data"], n_samples, interv), n_frec_div, offset=offset)
        
        #sample_labels[i*array_length:(i*array_length)+array_length] = class_mapping[signal["Class"]]
        if signal["Class"]!="Clean":
            local_start = signal['JammingStartTime']//interv - offset
            local_stop = (signal['JammingStartTime']+anomaly_duration)//interv - offset
            start = i*array_length + local_start + skip
            stop = i*array_length + local_stop + skip - n_bad_data
            sample_labels[start] = class_mapping[signal["Class"]]
            sample_labels[stop] = class_mapping[signal["Class"]]

            # removes the unused ranges (if remove_middle)
            energy_dif_matrix[array_length*i:array_length*(i+1),:] = np.concatenate(
                        (energy_dif[:local_start,:], energy_dif[(local_start+label_offset):(local_start+label_offset+1),:], 
                        energy_dif[(local_start+offset+1):local_stop,:],
                        energy_dif[(local_stop+label_offset):(local_stop+label_offset+1),:], energy_dif[(local_stop+offset+1):,:]), 
                    axis=0)
        else:
            # The last intervals are cut (if remove_middle) to fit in the energy dif matrix,
            # it is not needed but it simplifies the code
            energy_dif_matrix[array_length*i:array_length*(i+1),:] = energy_dif[:array_length,:]
    energy_dif_matrix, sample_labels = balance(energy_dif_matrix, sample_labels)
    return energy_dif_matrix, sample_labels


""" # backup lol
def compute_energy_matrix_and_labels(dataset:list, n_samples:int, interv:int, 
                                     n_frec_div:int, class_mapping:dict, 
                                     anomaly_duration = 12500, offset = 3):
    '''Builds energy arrays for each train signal 
    (x=window samples, y=frecuency divisions z=signal)'''
    array_length = (n_samples // interv) - offset
    N = len(dataset)
    
    energy_dif_matrix = np.empty((array_length*N, n_frec_div), dtype=np.float64)
    sample_labels = np.empty(array_length*N, np.int8)
    sample_labels[:] = class_mapping["Clean"]
    for i, signal in tqdm(enumerate(dataset)):
        energy_dif_matrix[array_length*i:array_length*(i+1),:] = energy_arrays(signal_interval(signal["Data"], n_samples, interv), n_frec_div, offset=offset)
        #sample_labels[i*array_length:(i*array_length)+array_length] = class_mapping[signal["Class"]]
        if signal["Class"]!="Clean":
            start = i*array_length + (signal['JammingStartTime']//interv) - offset +1
            stop = i*array_length + (signal['JammingStartTime']+anomaly_duration)//interv - offset +1
            sample_labels[start] = class_mapping[signal["Class"]]
            sample_labels[stop] = class_mapping[signal["Class"]]
    energy_dif_matrix, sample_labels = balance(energy_dif_matrix, sample_labels)
    return energy_dif_matrix, sample_labels
"""