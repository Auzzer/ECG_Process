from PIL import Image
import os
from os import path

def image_join(img1, img2, flag='vertical'):
    """
    :param png1: path
    :param png2: path
    :param flag: lateral or vertical
    :return:
    """
    size1, size2 = img1.size, img2.size
    if flag == 'lateral': # 横向
        join_image = Image.new('RGB', (size1[0] + size2[0], size1[1])) # 创建一个2原图合并后大小的空白图
        loc1, loc2 = (0, 0), (size1[0], 0)
        join_image.paste(img1, loc1) # 将原图1黏贴到指定位置
        join_image.paste(img2, loc2) # 将原图2黏贴到指定位置

    elif flag == 'vertical': # 纵向
        join_image = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        join_image.paste(img1, loc1)
        join_image.paste(img2, loc2)

    return(join_image)


TargetPath = './DataFinal/'
originPath = './DataTransformed/'

count = 0
for F in os.listdir(originPath):
    Temp = Image.open(originPath+F)
    OutputPath = TargetPath+F
    Temp = Temp.resize((1600, 16))
    size = Temp.size

    up = 16
    below = 0
    for i in range(10):
        if i == 0:
            left = 160 * i
            right = 160 * (i + 1)

            Output = Temp.crop((left, below, right, up))
        else:
            left = 160 * i
            right = 160 * (i + 1)
            Output = image_join(Output, Temp.crop((left, below, right, up)))
    print(Output.size)
    Output.save(OutputPath)
    print(count)
    count = count+1

