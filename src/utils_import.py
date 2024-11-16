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

def scan_directory(_list:list, input_directory:str, metadata:pd.DataFrame):
    FileName = list(metadata["FileName"])
    for arch in tqdm(os.scandir(input_directory)): 
        if arch.name in FileName:
            with open(arch.path, "rb") as file:
                signal = np.frombuffer(file.read(), dtype=np.complex64)
                row = metadata.loc[metadata["FileName"] == arch.name].index[0]
                _dict = {x:y for x,y in [("Data",signal),("Class",metadata["SignalType"][row]),("JammingStartTime",metadata["JammingStartTime"][row]),("AveragePower_dB",metadata["AveragePower_dB"][row])]}
                _list.append(_dict)
        if os.path.isdir(arch) == True:
            _list = scan_directory(_list,arch.path,metadata)
    return _list

@cache_data
def load_data(input_directory:str, metadata:str) -> list:
    metadata_pd = pd.read_csv(metadata, skip_blank_lines = True, header = 0)
    _list = []
    _list = scan_directory(_list,input_directory,metadata_pd)  
    return _list