import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fftshift
from scipy.signal import welch, spectrogram
        
# Funciones necesarias para mostrar el PSD y Spectrogram, respectivamente
def plot_PSD(signal, fs=1.0, title="", nfft=256, fc=0.0, ax=None):

    f, Pxx_spec = welch(signal, fs, nperseg=nfft, return_onesided=False, scaling="density")
    Pxx_spec_dB = 10 * np.log10(Pxx_spec)
    f = fftshift(f)
    Pxx_spec_dB = fftshift(Pxx_spec_dB)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot((f + fc) / 1e6, Pxx_spec_dB)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("PSD [dB/Hz]")
    ax.set_title(title)
    ax.grid(True)

def plot_spectrogram(
    signal_to_plot,
    fs=1.0,
    nfft=512,
    noverlap=0,
    return_onesided=False,
    title="",
    fc=0.0,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    f, t, Sxx = spectrogram(
        signal_to_plot,
        fs,
        nperseg=nfft,
        noverlap=noverlap,
        return_onesided=return_onesided,
    )
    Sxx_dB = 10 * np.log10(Sxx)

    im = ax.pcolormesh(
        ((fftshift(f) + fc) / (1e6)),
        (t * 1e3),
        fftshift(Sxx_dB.T, axes=1),
        shading="gouraud",
    )
    ax.set(
        xlabel="Frequency [MHz]",
        ylabel="Time [ms]",
        title=title,
        xlim=((fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6),
    )
    ax.grid(True)

    if im.colorbar is None:
        plt.colorbar(im, ax=ax)