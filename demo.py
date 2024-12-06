import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.mlab import window_hanning,specgram,psd
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import random


from scipy.fft import fftshift
from scipy.signal import welch, spectrogram

from src.rf_stream import Signal
from src.utils_import import load_data
from src.utils_preprocess import split_data

from sklearn.decomposition import PCA
from src.utils_preprocess import signal_interval, energy_arrays, compute_energy_matrix_and_labels
from src.model_rt import RealtimeModel, svc


SAMPLES_PER_FRAME = 64 # No. reads concatenated within a single window
nfft = 1024 # NFFT value for spectrogram
overlap = 512 # overlap value for spectrogram

Fs = 12e6 # sample rate
Fc = 0 # cutoff frequency
Ts = 1/Fs # sample period
N = 2048 # number of samples to simulate

var = 0
amp = 10

n_frec_div = 32
n_samples = 50000
num_signal_intervals = n_samples//nfft
n_shifts = 1

class_colors = {
    0: 'blue',    
    1: 'red',   
    2: 'red',     
    3: 'purple',
    4: 'purple'   
}

def get_sample(signal):
    """
    inputs: signal stream
    outputs: np.complex64 array
    """
    return signal.get_new_samples(N)

def get_specgram(signal,rate):
    """
    takes the FFT to create a spectrogram of the given audio signal
    input: audio signal, sampling rate
    output: 2D Spectrogram Array, Frequency Array, Bin Array
    see matplotlib.mlab.specgram documentation for help
    """
    arr2D,freqs,bins = specgram(signal,window=window_hanning,
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

def calc_pca_points(energy_dif,pca,predicted_vals):    
    new_pca_points = pca.transform(energy_dif)
    return pd.DataFrame({"PC1": new_pca_points[:,0], "PC2": new_pca_points[:,1], "class": predicted_vals})

def get_class_color(class_val):
    return class_colors.get(class_val, 'gray')

def get_data_and_model(): # TODO
    # Asumption: all signals consist of 50k samples
    n_samples = 50000
    interv = 1024 # Hyperparameter 1
    array_length = (n_samples // interv) - 1
    n_frec_div = 32 # Hyperparameter 2
    
    class_mapping = {"Clean": 0, "Narrowband Start": 1, "Narrowband Stop": 2, "Wideband Start": 3, "Wideband Stop": 4}

    # Load data
    signals_clean = load_data('dataset/Jamming/Clean', 'dataset/Jamming/metadata.csv')
    signals_narrowband = load_data('dataset/Jamming/Narrowband', 'dataset/Jamming/metadata.csv')
    signals_wideband = load_data('dataset/Jamming/Wideband', 'dataset/Jamming/metadata.csv')

    # Partition train=0.8, test=0.2
    clean_train, clean_test = split_data(signals_clean, 0.8)
    narrowband_train, narrowband_test = split_data(signals_narrowband, 0.8)
    wideband_train, wideband_test = split_data(signals_wideband, 0.8)

    train = clean_train + narrowband_train + wideband_train
    random.shuffle(train)
    train_energy_dif_matrix, sample_labels = compute_energy_matrix_and_labels(train, n_samples, interv, n_frec_div, class_mapping)

    #test = clean_test + narrowband_test + wideband_test
    #random.shuffle(test)
    #test_energy_dif_matrix, sample_labels = compute_energy_matrix_and_labels(test, n_samples, interv, n_frec_div, class_mapping)

    sample = clean_test[0]["Data"]
    
    ###
    # PCA MODEL for visualization with SVM support for model prediction
    pca = PCA(n_components=2)
    pca.fit(train_energy_dif_matrix)
    svm = svc(train_energy_dif_matrix, sample_labels)

    return sample, pca, svm

def main(SEED=1337):
    random.seed(SEED)
    np.random.seed(SEED)

    sample, pca, model = get_data_and_model()
    rt_model = RealtimeModel(model, nfft=nfft, n_partitions=n_frec_div)
    signal = Signal(sample, nfft=nfft, smoothing_steps=len(sample), FS=Fs,
                    anomaly_intensity=amp, anomaly_variability = var) # signal data stream

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
    
    plt.subplots_adjust(left=0.25, bottom=0.25)

    axcolor = 'lightgrey' #lightgoldenrodyellow
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axvar = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    
    samp = Slider(axamp, 'Amplitude', -4.0, 16.0, valinit=amp)
    svar = Slider(axvar, 'Variability', 0, 8.0, valinit=var)

    axpred = plt.axes([0.1, 0.45, 0.15, 0.15], facecolor=axcolor)
    axpred.axis('off')
    tpred = axpred.text(0,0,"Prediction:\nUnknown", horizontalalignment='center',
                        verticalalignment='center', bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 10})

    ###########

    def update(val):
        global amp, var
        amp = samp.val
        var = svar.val
        signal.anomaly_intensity = amp
        signal.anomaly_variability = var

    svar.on_changed(update)
    samp.on_changed(update)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('Clean', 'Wideband', 'Narrowband'),
                        label_props={'color': 'grb', 'fontsize': [12, 12, 12]},
                        radio_props={'s': [64, 64, 64]}, active=0)

    def noise_mode(mode):
        signal.mode = mode
    radio.on_clicked(noise_mode)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        svar.reset()
        samp.reset()
        rt_model.reset()
        radio.clear()
        tpred.set_text("Prediction:\nUnknown")
        tpred.set_backgroundcolor("green")
    button.on_clicked(reset)

    
    #########################
    
    """
    Launch the original spectrogram
    """
    data = get_sample(signal)
    arr2D,freqs,bins = get_specgram(data,Fs)
    """
    Setup the plot paramters
    """
    
    extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])
    im = ax1.imshow(arr2D,aspect='auto',extent = extent,interpolation="bilinear",
                    cmap = 'jet',norm = LogNorm(vmin=10*arr2D.min(),vmax=1e-6))
    #im = ax1.pcolormesh(arr2D, freqs, 10 * np.log10(bins), shading='auto', cmap='inferno')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Real Time Spectogram')
    plt.gca().invert_yaxis()
    fig.colorbar(im)


    Pxx,freqs = calc_psd(data,Fs)
    psd_line, = ax3.plot(freqs,Pxx, '-')
    ax3.set_xlabel("Frequency [MHz]")
    ax3.set_ylabel("PSD [dB/Hz]")
    ax3.set_title("PSD")
    ax3.set_ylim(top=-40)
    ax3.grid(True)
    
    ### 
    # PCA visualization
    global pca_buffer
            
    energy_dif = energy_arrays(signal_interval(data, N, nfft), n_frec_div, offset=1)
    new_pca_points, predicted_vals = pca.transform(energy_dif), model.predict(energy_dif)
    pca_buffer = pd.DataFrame({"PC1": new_pca_points[:, 0], "PC2": new_pca_points[:, 1], "class": predicted_vals})
    pca_buffer['color'] = pca_buffer['class'].apply(get_class_color)
    print(pca_buffer)

    scatter_alpha = np.linspace(0.1,1.0, len(pca_buffer))
    scatter = ax4.scatter(
        pca_buffer["PC1"], pca_buffer["PC2"],
        color=pca_buffer["color"],
        alpha=scatter_alpha
    )
    
    ax4.set_title('PCA')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_xlim(-1.7,1.7)
    ax4.set_ylim(-0.5,0.5)
    ax4.grid(True)
    
    #legend1 = ax4.legend(*scatter.legend_elements(),
    #               loc="upper left", title="")
    #ax4.add_artist(legend1)
    
    fig.subplots_adjust(left=0.25, bottom=0.25, right=0.95)

    ###
    
    ax2.axis('off')
    
    img = Image.open('neko.jpg')
    rgb = img.convert('RGB')
    img_rgb = np.array(rgb)
    im_anom = ax2.imshow(img_rgb, aspect = 'auto')

    def update_fig(n):
        """
        updates the image, just adds on samples at the start until the maximum size is
        reached, at which point it 'scrolls' horizontally by determining how much of the
        data needs to stay, shifting it left, and appending the new data. 
        inputs: iteration number
        outputs: updated image
        """
        global pca_buffer
        
        data = get_sample(signal)

        pred, s_start = rt_model.get_current_prediction(data)
        
        tpred.set_text("Prediction:\n"+pred)
        if signal.mode == pred:
            tpred.set_backgroundcolor("green")
        else:
            tpred.set_backgroundcolor("red")
        
        Pxx,freqs = calc_psd(data,Fs)
        
        psd_line.set_data(freqs,Pxx)
        
        ###
        # PCA update 
        if rt_model.ready:
            buffer = rt_model.fd_buffer[:,:,n_shifts-1]
            predicted_vals = rt_model.predictions[:,n_shifts-1]
            new_pca_points = calc_pca_points(buffer, pca, predicted_vals)
            
            pca_buffer = pd.concat([pca_buffer, new_pca_points], ignore_index=True)
            pca_buffer['color'] = pca_buffer['class'].apply(get_class_color)
            if len(pca_buffer) > num_signal_intervals:
                pca_buffer = pca_buffer.iloc[-num_signal_intervals:]
            scatter.set_offsets(pca_buffer[["PC1", "PC2"]].to_numpy())
            scatter.set_alpha(scatter_alpha[:len(pca_buffer)])
            scatter.set_facecolor(pca_buffer["color"].to_numpy())
        ###
        
        arr2D,freqs,bins = get_specgram(data,Fs)
        im_data = im.get_array()
        if n < SAMPLES_PER_FRAME:
            im_data = np.hstack((im_data,arr2D))
            im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
            im_data = np.delete(im_data,np.s_[:-keep_block],1)
            im_data = np.hstack((im_data,arr2D))
            im.set_array(im_data)

        im_noise = random.uniform(1, 1+var)
        if signal.mode.lower() == "wideband":
            im_noise = (im_noise+amp)*10
        elif signal.mode.lower() == "narrowband":
            im_noise = (im_noise+amp)*2
        else:
            im_noise = 0
            
        ruido = np.random.normal(0,max(im_noise,0),(img_rgb.shape[0],img_rgb.shape[1],3))
        im_gaus = img_rgb + np.array(ruido)
        im_gaus = np.clip(im_gaus, 0, 255).astype(np.uint8)
        im_anom.set_array(im_gaus)

        return im

    ############### Animate ###############
    anim = animation.FuncAnimation(fig,update_fig,blit = False,
                                interval=40)
    
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
