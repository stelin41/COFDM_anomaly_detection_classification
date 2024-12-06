from .utils_import import load_data
from scipy.signal import welch, firwin, filtfilt
import numpy as np
import random

class Signal():
    def __init__(self, clean_signal, nfft=1024, smoothing_steps=1024, FS = 1, 
                 mode = "Clean", anomaly_intensity=3, anomaly_variability=10):
        """
        FS: sample rate in Hz

        """
        
        self.anomaly_intensity = anomaly_intensity
        self.anomaly_variability = anomaly_variability
        self.mode = mode

        self.nfft = nfft
        self.FS = FS
        self.index = 0
        z = np.linspace(-10, 10, smoothing_steps) 
        smoothing = 1/(1 + np.exp(-z)) # Sigmoid function
        smoothing_inv = np.flip(smoothing)

        # Smooth transition to make the signal loop look natural
        self.clean_signal = np.concatenate([
                clean_signal[smoothing_steps:-smoothing_steps],
                clean_signal[:smoothing_steps]*smoothing + clean_signal[-smoothing_steps:]*smoothing_inv
            ])

        # Calculate PSD and max power
        f, Pxx_spec = welch(
            self.clean_signal, fs=FS, nperseg=nfft, return_onesided=False, scaling="density"
        )
        PSD_signal = 10 * np.log10(Pxx_spec)
        self.max_PSD = np.max(PSD_signal)
    
    def generate_jamming(self, n, mode="Wideband"):
        """
        n: number of samples to generate
        mode: "Narrowband" or "Wideband"

        returns: np.array[np.complex64]
        """
        # Define ISR levels and create jamming signals
        if mode.lower() == "wideband":
            #random.uniform(3, 13)
            ISR_wide = random.uniform(0,self.anomaly_variability) + self.anomaly_intensity
            ISR_to_apply_wide = self.max_PSD + ISR_wide + 10 * np.log10(self.FS)
            gain_factor_wide = np.sqrt(10 ** (ISR_to_apply_wide / 10))
            jamming_signal_wide = gain_factor_wide * self.generate_lpf_noise(
                cutoff=2e6, N=n
            )
            return jamming_signal_wide
        elif mode.lower() == "narrowband":
            #random.uniform(6, 16)
            ISR_narrow = random.uniform(0,self.anomaly_variability) + self.anomaly_intensity 
            ISR_to_apply_narrow = self.max_PSD + ISR_narrow + 10 * np.log10(self.FS)
            gain_factor_narrow = np.sqrt(10 ** (ISR_to_apply_narrow / 10))
            jamming_signal_narrow = gain_factor_narrow * self.generate_lpf_noise(
                cutoff=0.2e6, N=n
            )
            return jamming_signal_narrow
        else:
            raise Exception(f'Mode "{mode}" not recognized.')


    def generate_lpf_noise(self, cutoff: float = 0.25, N: int = 2000):

        # Design filter
        b = firwin(61, cutoff=cutoff, window="hamming", fs=self.FS)

        # Generate noise
        Es = 1
        SNR = 0
        noiseVar = Es / np.power(10, (SNR / 10))
        noise = np.sqrt(noiseVar / 2) * (np.random.randn(N) + 1j * np.random.randn(N))

        output = filtfilt(b, 1, noise)
        return output
    
    def get_new_samples(self, n):
        """
        Generate n new np.array[np.complex64] generated samples, 
        including the anomaly if the current self.mode is not "Clean"
        """
        assert n<=self.clean_signal.shape[0], "n can't be larger than the signal"
        start = self.index
        self.index = (self.index+n)%self.clean_signal.shape[0]
        if self.index > start:
            output = self.clean_signal[start:self.index]
        else:
            output = np.concatenate([self.clean_signal[start:], self.clean_signal[:self.index]])

        if self.mode != "Clean":
            output = output.copy() + self.generate_jamming(n, self.mode)

        return output
        
