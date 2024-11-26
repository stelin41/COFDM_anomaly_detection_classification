

class RealtimeModel():
    def __init__(self, model, nfft=1024, offset=4, n_shifts=0, n_partitions=32):
        """
        Use a energy diff vector classifier to do real time predictions over a stream of data.

        model: Energy vector classifier. Must have a .pretict(X) method.
        nfft: Size of the fft transformation, recommended to be the same as the one used to train the model
        offset: Number of intervals to skip to calculate the energy diff vectors
        n_shifts: Number of overlapping shifted intervals, used to detect 
                  the signal faster in cost of extra computation. It must be 0 or a power of 2.
        n_partitions: Size of the energy vector, must be the same as the one used to train the model.
        """
        self.model = model
        self.nfft = nfft
        self.n_partitions = n_partitions
        self.buffer = []
        self.buffer_classes = []

    def add_new_samples(self, X:np.array):
        pass

    def get_current_prediction(self):
        pass

    def _get_model_output(self):
        pass

    

