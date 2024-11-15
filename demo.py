import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons





from matplotlib.mlab import window_hanning,specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import ipywidgets as widgets
import numpy as np

#from src.rf_stream import 


#SAMPLES_PER_FRAME = 10 #Number of mic reads concatenated within a single window
SAMPLES_PER_FRAME = 100
nfft = 1024#256#1024 #NFFT value for spectrogram
overlap = 1000#512 #overlap value for spectrogram
#rate = mic_read.RATE #sampling rate


Fs = 300 # sample rate
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




def main():
    #fig = plt.figure()

    fig, (ax1, ax2) = plt.subplots(2)
    
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = ax1.plot(t, s, lw=2)
    ax1.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=freq, valstep=delta_f)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

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
    im = ax2.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                    cmap = 'jet',norm = LogNorm(vmin=.01,vmax=1))
    #plt.xlabel('Time (s)')
    #plt.ylabel('Frequency (Hz)')
    #plt.title('Real Time Spectogram')
    plt.gca().invert_yaxis()
    ##plt.colorbar() #enable if you want to display a color bar

    def update_fig(n):
        """
        updates the image, just adds on samples at the start until the maximum size is
        reached, at which point it 'scrolls' horizontally by determining how much of the
        data needs to stay, shifting it left, and appending the new data. 
        inputs: iteration number
        outputs: updated image
        """
        data = get_sample()
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
    pa.terminate()
    print("Program Terminated")

if __name__ == "__main__":
    main()











