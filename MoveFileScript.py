import os
import random
import shutil

FromPath = './train/'
ToPath = './test/'

Fall = os.listdir(FromPath)
length = len(Fall)

list = random.sample(range(length), int(0.2*length))
name = []
for item in list:
    name.append(Fall[item])

for item in name:
    shutil.move(FromPath+item, ToPath+item)

