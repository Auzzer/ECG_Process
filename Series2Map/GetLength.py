import pandas as pd
import neurokit2 as nk
import os
from os import path
PATH = os.getcwd()
PATH = path.join(PATH, 'Data')
file = os.listdir(PATH)
Length = []
count = 0
for f in file:
    ecg_signals = pd.read_csv(path.join(PATH, f), header=None).squeeze().dropna()
    _, rpeaks = nk.ecg_peaks(ecg_signals, sampling_rate=300)
    Length.append(len(rpeaks['ECG_R_Peaks']))
    print(count)
    count = count+1

all = len(file)
count = 7984
for count in range(7984, all):
    f = 'A0' + str(count) + '.csv'
    ecg_signals = pd.read_csv(path.join(PATH, f), header=None).squeeze().dropna()
    _, rpeaks = nk.ecg_peaks(ecg_signals, sampling_rate=300)
    Length.append(len(rpeaks['ECG_R_Peaks']))
    print(count)

Length = pd.DataFrame(Length)
Length.to_csv("EachSeriesSplittingNum.csv",index=False,sep=',')