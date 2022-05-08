import pandas as pd
from cydets.algorithm import detect_cycles

import csv
test = pd.read_csv("./Data/A00001.csv", header = None)
# test = pd.Series(test)
test = test.squeeze()
test=test.dropna()
#detect_cycles(test)
#pd.Series.from_csv("./Data/A00001.csv", header = None)


from matrixprofile import *
import numpy as np
a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
matrixProfile.stomp(a,4)






# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images

# Retrieve ECG data from data folder (sampling rate= 1000 Hz)
ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']
ecg_signal = test
# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=300)

# Visualize R-peaks in ECG signal
nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)

# Zooming into the first 5 R-peaks
nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:20000])

# Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=300, method="peak")

# Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'],
                       waves_peak['ECG_P_Peaks'],
                       waves_peak['ECG_Q_Peaks'],
                       waves_peak['ECG_S_Peaks']], ecg_signal)

# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3],
                       waves_peak['ECG_P_Peaks'][:3],
                       waves_peak['ECG_Q_Peaks'][:3],
                       waves_peak['ECG_S_Peaks'][:3]], ecg_signal[:1000])
plt.show()

# visualize T-wave boundaries
signal_dwt_P, waves_dwt_P = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=300, method="dwt", show=True, show_type='bounds_P')
signal_dwt_T, waves_dwt_T = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=300, method="dwt", show=False, show_type='bounds_T')
waves_dwt_P['ECG_T_Onsets'][0]


import os
from os import path
PATH = os.getcwd()
file = os.listdir(path.join(PATH, 'Data_test'))
length = []
count = 0
for f in file:
    temp = pd.read_csv(path.join(PATH, 'Data_test')+'\\'+f, header = None).squeeze().dropna()
    _, rpeaks = nk.ecg_peaks(temp, sampling_rate=300)
    if min(np.diff(rpeaks['ECG_R_Peaks']))<200:
        length.append(count)
    count = count+1






