import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.mlab import window_hanning,specgram,psd
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np

from scipy.fft import fftshift
from scipy.signal import welch, spectrogram

#from src.utils_exploration import plot_PSD
#from src.rf_stream import 


#SAMPLES_PER_FRAME = 10 
SAMPLES_PER_FRAME = 100 #Number reads concatenated within a single window
nfft = 1024 #NFFT value for spectrogram
overlap = 1000 #overlap value for spectrogram
#rate = mic_read.RATE #sampling rate


Fs = 300 # sample rate
Fc = 0 # cutoff frequency
Ts = 1/Fs # sample period
N = 2048 # number of samples to simulate
rate = Fs

freq = 50 # simulates sinusoid at 50 Hz

def get_sample():
    """
    gets the audio data from the microphone
    inputs: audio stream and PyAudio object
    outputs: int16 array
    """
    #data = mic_read.get_data(stream,pa)
    t = Ts*np.arange(N)
    x = np.exp(1j*2*np.pi*freq*2*t)
    n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # complex noise with unity power
    noise_power = 2
    data = x + n * np.sqrt(noise_power)
    #data = np.random.randint(0, 100, N, dtype=np.int16)
    return data

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


def main():
    #fig = plt.figure()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = ax4.plot(t, s, lw=2)
    ax4.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=freq, valstep=delta_f)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


    ###########

    def update(val):
        global freq
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
    #dataset = load_dataset(r'dataset/Jamming_test/Clean')

    data = get_sample()
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

    def update_fig(n):
        """
        updates the image, just adds on samples at the start until the maximum size is
        reached, at which point it 'scrolls' horizontally by determining how much of the
        data needs to stay, shifting it left, and appending the new data. 
        inputs: iteration number
        outputs: updated image
        """
        data = get_sample()
        
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
        return im,

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
