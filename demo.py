import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.mlab import window_hanning,specgram,psd
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import random

from scipy.fft import fftshift
from scipy.signal import welch, spectrogram

from src.rf_stream import Signal
from src.utils_import import load_data
from src.utils_preprocess import split_data

from PIL import Image

#from src.utils_exploration import plot_PSD
#from src.rf_stream import 


#SAMPLES_PER_FRAME = 10 
SAMPLES_PER_FRAME = 100 #Number reads concatenated within a single window
nfft = 1024 #NFFT value for spectrogram
overlap = 512 #overlap value for spectrogram
#rate = mic_read.RATE #sampling rate


Fs = 300 # sample rate
Fc = 0 # cutoff frequency
Ts = 1/Fs # sample period
N = 2048 # number of samples to simulate
rate = Fs

freq = 50 # simulates sinusoid at 50 Hz
amp = 1

def get_sample(signal):
    """
    gets the audio data from the microphone
    inputs: audio stream and PyAudio object
    outputs: int16 array
    """

    t = Ts*np.arange(N)
    x = np.exp(1j*2*np.pi*freq*2*t)*amp

    #n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # complex noise with unity power
    #noise_power = 2
    #data = x + n * np.sqrt(noise_power)
    #data = np.random.randint(0, 100, N, dtype=np.int16)

    data = x*0.01 + signal.get_new_samples(N)
    #data = signal.get_new_samples(N)
    return data

def get_specgram(signal,rate):
    """
    takes the FFT to create a spectrogram of the given audio signal
    input: audio signal, sampling rate
    output: 2D Spectrogram Array, Frequency Array, Bin Array
    see matplotlib.mlab.specgram documentation for help
    """
    arr2D,freqs,bins = specgram(10. * np.log10(signal),window=window_hanning,
                                Fs = rate,NFFT=nfft,noverlap=overlap)
    return arr2D,freqs,bins

def calc_psd(signal,rate):
    f, Pxx_spec = welch(signal, rate, nperseg=nfft, return_onesided=False, scaling="density")
    Pxx_spec_dB = 10 * np.log10(Pxx_spec)
    f = fftshift(f)
    Pxx_spec_dB = fftshift(Pxx_spec_dB)
        
    return Pxx_spec_dB, (f + Fc) / 1e6


# I think this is faster but it is not
# in the correct scale
def calc_psd2(signal,rate):
    Pxx,freqs = psd(signal,window=window_hanning,
                                Fs = rate,NFFT=nfft,noverlap=overlap)
    return Pxx,freqs

def get_data_and_model(): # TODO
    # Asumption: all signals consist of 50k samples
    n_samples = 50000
    interv = 1024 # Hyperparameter 1
    array_length = (n_samples // interv) - 1
    n_frec_div = 32 # Hyperparameter 2

    # Load data
    signals_clean = load_data('dataset/Jamming/Clean', 'dataset/Jamming/metadata.csv')
    #signals_narrowband = load_data('dataset/Jamming/Narrowband', 'dataset/Jamming/metadata.csv')
    #signals_wideband = load_data('dataset/Jamming/Wideband', 'dataset/Jamming/metadata.csv')

    # Partition train=0.8, test=0.2
    clean_train, clean_test = split_data(signals_clean, 0.8)
    #narrowband_train, narrowband_test = split_data(signals_narrowband, 0.8)
    #wideband_train, wideband_test = split_data(signals_wideband, 0.8)

    #train = clean_train + narrowband_train + wideband_train
    #test = clean_test + narrowband_test + wideband_test

    #random.shuffle(train)
    #random.shuffle(test)

    #train_energy_dif_matrix, sample_labels = compute_energy_matrix_and_labels(train, n_samples, interv, n_frec_div, class_mapping)

    sample = clean_test[0]["Data"]

    return sample

def main(SEED=1337):
    random.seed(SEED)
    np.random.seed(SEED)

    sample = get_data_and_model()

    #fig = plt.figure()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    delta_f = 5.0
    s = amp * np.sin(2 * np.pi * freq * t)
    l, = ax4.plot(t, s, lw=2)
    ax4.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=freq, valstep=delta_f)
    samp = Slider(axamp, 'Amp', 0.1, 5.0, valinit=amp)


    ###########

    def update(val):
        global freq, amp
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()


    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        sfreq.reset()
        samp.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)
    
    #########################
    
    """
    Launch the stream and the original spectrogram
    """
    signal = Signal(sample, nfft=nfft)
    data = get_sample(signal)
    arr2D,freqs,bins = get_specgram(data,rate)
    """
    Setup the plot paramters
    """
    extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])
    im = ax1.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                    cmap = 'jet',norm = LogNorm(vmin=.01,vmax=1))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Real Time Spectogram')
    plt.gca().invert_yaxis()
    fig.colorbar(im)


    #_, _, psd = ax3.psd(data,window=window_hanning,
    #                            Fs = rate,NFFT=nfft,noverlap=overlap, return_line=True)
    Pxx,freqs = calc_psd(data,rate)
    psd_line, = ax3.plot(freqs,Pxx, '-')
    ax3.set_xlabel("Frequency [MHz]")
    ax3.set_ylabel("PSD [dB/Hz]")
    ax3.set_title("PSD")
    ax3.grid(True)

    ax2.axis('off')

    img = Image.open('mica.jpg')
    rgb = img.convert('RGB')
    img_rgb = np.array(rgb)

    ruido = np.random.normal(0,amp*10,(img_rgb.shape[0],img_rgb.shape[1],3))
    im_gaus = img_rgb + np.array(ruido)
    im_gaus = np.clip(im_gaus, 0, 255).astype(np.uint8)
    im_anom = ax2.imshow(im_gaus, aspect = 'auto')

    def update_fig(n):
        """
        updates the image, just adds on samples at the start until the maximum size is
        reached, at which point it 'scrolls' horizontally by determining how much of the
        data needs to stay, shifting it left, and appending the new data. 
        inputs: iteration number
        outputs: updated image
        """
        data = get_sample(signal)
        
        Pxx,freqs = calc_psd(data,rate)

        psd_line.set_data(freqs,Pxx)

        
        arr2D,freqs,bins = get_specgram(data,rate)
        im_data = im.get_array()
        if n < SAMPLES_PER_FRAME:
            im_data = np.hstack((im_data,arr2D))
            im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
            im_data = np.delete(im_data,np.s_[:-keep_block],1)
            im_data = np.hstack((im_data,arr2D))
            im.set_array(im_data)
            
        ruido = np.random.normal(0,amp*10,(img_rgb.shape[0],img_rgb.shape[1],3))
        im_gaus = img_rgb + np.array(ruido)
        im_gaus = np.clip(im_gaus, 0, 255).astype(np.uint8)
        im_anom.set_array(im_gaus)

        return im

    ############### Animate ###############
    anim = animation.FuncAnimation(fig,update_fig,blit = False,
                                interval=1)
    
    try:
        plt.show()
    except:
        print("Plot Closed")

    ############### Terminate ###############
    #stream.stop_stream()
    #stream.close()
    #pa.terminate()
    print("Program Terminated")

if __name__ == "__main__":
    main()
