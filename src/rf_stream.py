from utils_import import load_data

class Signal():
    def __init__(self, nfft=1024, dataset=None, clean_dataset='../dataset/Jamming/Clean', metadata='../dataset/Jamming/metadata.csv'):
        if dataset != None:
            signals_clean = dataset
        else:
            signals_clean = load_data(clean_dataset, metadata)

        self.anomaly_enabled = False
        self.anomaly_intensity = 0
        self.anomaly_width = 0
        self.nfft = nfft
    
    def get_new_samples(self, n):
        pass
