import pandas as pd
import os
from os import path


Refer = pd.read_csv("./REFERENCE.csv", header=None)
ReferOrigin = pd.read_csv("./REFERENCE-original.csv", header=None)
FigurePath = './DataFinal/'
count = 0
UltPath = './DataUlta/'
for f in os.listdir(FigurePath):
    item = f[0:6]

    if 'True' in Refer[0] == item:
        row = Refer[0][Refer[0]==item].index[0]
        Name = str(Refer.iloc[row,:][1]) +'.' + str(Refer.iloc[row,:][0]) + '.jpg'
    else:
        row = ReferOrigin[0][ReferOrigin[0] == item].index[0]
        Name = str(ReferOrigin.iloc[row, :][1]) + '.' + str(ReferOrigin.iloc[row, :][0]) + '.jpg'


    os.rename(FigurePath+f, UltPath+Name)
    print(count)
    count+=1