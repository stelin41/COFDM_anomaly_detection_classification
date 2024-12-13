
[![running demo](demo.gif)](https://www.youtube.com/watch?v=HhnafOJapxQ)

Installation:
```bash
chmod +x install.sh
./install.sh
```

Running demo:
```bash
source .env/bin/activate
python3 demo.py
```

Assumptions:
- We know how the signal looks like when it's clean (it starts clean)
- The anomaly is additive/subtractive (increases/decreases the energy level)
- The anomaly is at least `(1/n_frec_div)*sample_bandwith` wide


Assumptions that can be compensated or fixed with better control mechanisms:
- The anomaly lasts at least 5`nfft` (it can be reduced to one `nfft` or less)
- The anomaly starts/stops suddently (not gradually)
- There can only be one simultaneous anomaly, and it can't change it's class

As shown in [experiments.ipynb](experiments.ipynb), the model has a high accuracy under these conditions.
```
Accuracy: 0.9997777777777778

Confusion Matrix:
[[1500    0    0]
 [   1 1499    0]
 [   0    0 1500]]
 ```

The dataset is structured like:
- `dataset/Jamming/Clean`
- `dataset/Jamming/Narrowband`
- `dataset/Jamming/Wideband`

Each folder has multiple files, each one is a single IQ recording saved as a np.complex64 numpy buffer.

The metadata file is located at `dataset/Jamming/metadata.csv`, and it should look like this:
```
FileName,SignalType,JammingStartTime,AveragePower_dB
000498000_000012000_000050000_1731003770598630_DVBT.data,Clean,-1,-28.867297172546387
000498000_000012000_000050000_1731084527072217_DVBTWidebandJamming.data,Wideband,18671,-28.867297172546387
000498000_000012000_000050000_1731084527088622_DVBTNarrowbandJamming.data,Narrowband,18671,-28.867297172546387
...
``` 

The config file is located at `dataset/Jamming/config.csv`, and it should look like this (although it is currently being ignored):
```
SampleRate,CenterFrequency,NFFT
12000000.0,498000000.0,1024
```


## Credits

Authors:
- .
- .
- .

Made in collaboration with University of Santiago de Compostela.

Co-tutors:
- Francisco Javier Valera Sánchez 
- Anxo Tato Arias