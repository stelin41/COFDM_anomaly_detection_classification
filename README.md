
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

The config file is located at `dataset/Jamming/config.csv`, and it should look like this:
```
SampleRate,CenterFrequency,NFFT
12000000.0,498000000.0,1024
```