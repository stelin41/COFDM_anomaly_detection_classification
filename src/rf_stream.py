from .utils_import import load_data
import numpy as np

class Signal():
    def __init__(self, clean_signal, nfft=1024, smoothing_steps=512):
        
        self.anomaly_enabled = False
        self.anomaly_intensity = 0
        self.anomaly_width = 0
        self.nfft = nfft
        self.index = 0
        z = np.linspace(-10, 10, smoothing_steps) 
        smoothing = 1/(1 + np.exp(-z)) # Sigmoid function
        smoothing_inv = np.flip(smoothing)

        # Smooth transition to make the signal loop look natural
        self.clean_signal = np.concatenate([
                clean_signal[smoothing_steps:-smoothing_steps],
                clean_signal[:smoothing_steps]*smoothing + clean_signal[-smoothing_steps:]*smoothing_inv
            ])
    
    def get_new_samples(self, n):
        assert n<=self.clean_signal.shape[0], "n larger than the signal"
        start = self.index
        self.index = (self.index+n)%self.clean_signal.shape[0]
        if self.index > start:
            output = self.clean_signal[start:self.index]
        else:
            output = np.concatenate([self.clean_signal[start:], self.clean_signal[:self.index]])

        # TODO: add noise

        return output
        
