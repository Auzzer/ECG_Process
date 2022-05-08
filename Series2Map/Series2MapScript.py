import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
import neurokit2 as nk
from matplotlib import image

image_size = 16

import os
from PIL import Image
from os import path

def join(img1, img2, flag='horizontal'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    # 统一图片尺寸，可以自定义设置（宽，高）
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        return(joint)
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        return(joint)

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)



def scale(x):
    min = np.min(x)
    max = np.max(x)
    return ((x-max+x-min)/(max-min))

def Series2MapScript(TempPath, OriginPath, FinalPath):
    CountIndicator = 0
    for F in os.listdir(OriginPath):

        ecg_signals = pd.read_csv(OriginPath + F, header=None).squeeze().dropna()
        _, rpeaks = nk.ecg_peaks(ecg_signals, sampling_rate=300)
        Point = rpeaks['ECG_R_Peaks']
        Start = 0
        End = 0
        length = len(Point)

        count = 0
        for item in Point:
            if count < length:
                End = item
                sin_data = scale(np.array(ecg_signals[Start:End]))

                while sin_data.size < image_size:
                    sin_data = np.append(sin_data, sin_data.mean())
                sin_data = sin_data.reshape(1, -1)

                gadf = GramianAngularField(image_size=image_size, method='summation')
                sin_gadf = gadf.fit_transform(sin_data)
                name = TempPath + str(count) + '.png'
                image.imsave(name, sin_gadf[0])
                Start = End + 1
                count = count + 1
            if count == length:
                End = len(ecg_signals)


                if max(np.array(ecg_signals[Start:End])) - min(np.array(ecg_signals[Start:End])) == 0:
                    sin_data = np.repeat(1, image_size)
                    sin_data = sin_data.reshape(1, -1)
                    gadf = GramianAngularField(image_size=image_size, method='summation')
                    sin_gadf = gadf.fit_transform(sin_data)
                    np.place(sin_gadf, sin_gadf != 255, 255)
                else:
                    sin_data = scale(np.array(ecg_signals[Start:End]))
                    while sin_data.size < image_size:
                        sin_data = np.append(sin_data, sin_data.mean())
                    sin_data = sin_data.reshape(1, -1)

                    gadf = GramianAngularField(image_size=image_size, method='summation')
                    sin_gadf = gadf.fit_transform(sin_data)

                name = TempPath + str(count) + '.png'
                image.imsave(name, sin_gadf[0])
        ## surge the figures in Temp
        count = 0
        for f in os.listdir(TempPath):
            img = Image.open(path.join(TempPath, f))  # 打开图片
            # im = np.array(img)  # 转化为ndarray对象
            if count == 0:
                output = img
            else:
                output = join(output, img)  # 横向拼接
            count = count + 1
        # 生成图片
        output.save(FinalPath + str(F[0:6]) + '.jpg')
        del_file(TempPath)
        print(CountIndicator)
        CountIndicator = CountIndicator + 1

    print("Finished!")




"""
TempPath = './temp/'
FinalPath = './DataTransformed/'
OriginPath = './data/'

# Stop point Debug Tool
import pandas as pd
import neurokit2 as nk
import os
from os import path
PATH = os.getcwd()
Count = 7005
F = '/Data/A0'+ str(Count)+ '.csv'


ecg_signals = pd.read_csv(PATH+ F, header=None).squeeze().dropna()
_, rpeaks = nk.ecg_peaks(ecg_signals, sampling_rate=300)
Point = rpeaks['ECG_R_Peaks']
Start = 0
End = 0
length = len(Point)

count = 0
for item in Point:
    if count < length:
        End = item
        sin_data = scale(np.array(ecg_signals[Start:End]))

        while sin_data.size < image_size:
            sin_data = np.append(sin_data, sin_data.mean())
        sin_data = sin_data.reshape(1, -1)

        gadf = GramianAngularField(image_size=image_size, method='summation')
        sin_gadf = gadf.fit_transform(sin_data)
        name = TempPath + str(count) + '.png'
        image.imsave(name, sin_gadf[0])
        Start = End + 1
        count = count + 1
    if count == length:
        End = len(ecg_signals)

        sin_data = scale(np.array(ecg_signals[Start:End]))
        if max(np.array(ecg_signals[Start:End]))- min(np.array(ecg_signals[Start:End])) == 0:
            sin_data = np.repeat(1, image_size)
            sin_data = sin_data.reshape(1, -1)
            gadf = GramianAngularField(image_size=image_size, method='summation')
            sin_gadf = gadf.fit_transform(sin_data)
            np.place(sin_gadf, sin_gadf!=255, 255)
        else:
            while sin_data.size < image_size:
                sin_data = np.append(sin_data, sin_data.mean())
            sin_data = sin_data.reshape(1, -1)

            gadf = GramianAngularField(image_size=image_size, method='summation')
            sin_gadf = gadf.fit_transform(sin_data)



        name = TempPath + str(count) + '.png'
        image.imsave(name, sin_gadf[0])
## surge the figures in Temp
count = 0
for f in os.listdir(TempPath):
    img = Image.open(path.join(TempPath, f))  # 打开图片
    # im = np.array(img)  # 转化为ndarray对象
    if count == 0:
        output = img
    else:
        output = join(output, img)  # 横向拼接
    count = count + 1
# 生成图片
output.save(FinalPath + str(F[0:6]) + '.jpg')


"""

