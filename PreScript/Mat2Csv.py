from scipy.io import loadmat
import os
from os import path
import pandas as pd
target = 'Data/'
PATH = os.getcwd()
file = os.listdir(path.join(PATH, 'training2017'))
"""
f = file[0]
temp = loadmat(path.join( path.join(PATH, 'training2017'),f))
# temp.key()
temp = pd.DataFrame(temp['val'])
output = temp
"""
count=0
for f in file:
    temp = loadmat(path.join(path.join(PATH, 'training2017'), f))
    # temp.key()
    temp = pd.DataFrame(temp['val'])
    # output = output.append(temp)
    print(count)

    temp.to_csv(target + f[:7] + 'csv', index=False, sep=',', header=False)
    count = count + 1
"""
f = file[0]
temp = loadmat(path.join( path.join(PATH, 'training2017'),f))
# temp.key()
temp = pd.DataFrame(temp['val'])
output.append(temp)
output
"""

