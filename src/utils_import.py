import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def cache_data(func):
    cache = {}

    def wrapper(*args, **kwargs):
        cache_key = str(args) + str(kwargs)
        
        if cache_key in cache:
            print("Cargando datos desde cache...")
            return cache[cache_key]
        
        result = func(*args, **kwargs)
        cache[cache_key] = result
        
        return result

    return wrapper

@cache_data
def load_data(input_directory:str, metadata:str) -> dict:
    metadata_pd = pd.read_csv(metadata, skip_blank_lines = True, header = 0)
    FileName = list(metadata_pd["FileName"])
    _list = []
    for arch in tqdm(os.scandir(input_directory)): 
        if arch.name in FileName:
            with open(arch.path, "rb") as file:
                signal = np.frombuffer(file.read(), dtype=np.complex64)
                row = metadata_pd.loc[metadata_pd["FileName"] == arch.name].index[0]
                _dict = {x:y for x,y in [("Data",signal),("Class",metadata_pd["SignalType"][row]),("JammingStartTime",metadata_pd["JammingStartTime"][row]),("AveragePower_dB",metadata_pd["AveragePower_dB"][row])]}
                _list.append(_dict)
    return _list