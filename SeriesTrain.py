from VisionModel.TNT import TNT

import torch

print(f"TorchVersion: {torch.__version__}")

# Hyparametrics Set
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random

batch_size = 128
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 1234


def seed_everything(seed):
    """
    seed:种子数
    对所有随机设置种子数
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

train_dir = './train'
test_dir = './test'

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/ ')[-1].split('.')[0] for path in train_list]
"""
random_idx = np.random.randint(1, len(train_list), size=9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, ax in enumerate(axes.ravel()):
    img = Image.open(train_list[idx])
    ax.set_title(labels[idx])
    ax.imshow(img)
"""
## splite


train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)
print(f"Train Data Num: {len(train_list)}")
print(f" Validation Data Num: {len(valid_list)}")
print(f"Test Data Num: {len(test_list)}")

"""
数据预处理部分
注意：如果是在测试Robust的时候，测试集不要使用增加遮挡和高斯模糊
"""
from torchvision import transforms

train_transforms = transforms.Compose([
    # transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    # transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    # transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 加载数据
from torch.utils.data import DataLoader, Dataset


class CatDog(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transfomed = self.transform(img)

        label = img_path.split("/")[-1].split("/")[0]
        label = 1 if label == "dog" else 0

        return img_transfomed, label


train_data = CatDog(train_list, transform=train_transforms)
valid_data = CatDog(valid_list, transform=val_transforms)
test_data = CatDog(test_list, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(f"check the TrainSet loader: {len(train_data), len(train_loader)}")

print(f"check the ValidSet loader: {len(valid_data), len(valid_loader)}")

###训练前准备
print("开始训练检查")
print("检查GPU")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
print(torch.cuda.get_device_name(0) if (torch.cuda.is_available()) else "No GPU Availale")

################################定义网络################################################
from linformer import Linformer
import time

start_time = time.time()
import torch


model = TNT(
    image_size = 160,       # size of image
    patch_dim = 160,        # dimension of patch token
    pixel_dim = 24,         # dimension of pixel token
    patch_size = 16,        # patch size
    pixel_size = 4,         # pixel size
    depth = 2,              # depth
    num_classes = 3,     # output number of classes
    attn_dropout = 0.1,     # attention dropout
    ff_dropout = 0.1        # feedforward dropout
).to(device)

####################################开始训练##################################

import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

print("网络检查结束")
print("开始检查训练策略")
### Training


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
print("Loss function:", criterion, "\n", "Optimzer:", optimizer, "\n", "Scheduler", scheduler, '\n')

print('开始训练')

Loss_list = []
Acc_list = []
Loss_val_list = []
Acc_val_list = []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    Loss_list.append(float(f"{epoch_loss:.4f} "))
    Acc_list.append(float(f"{epoch_accuracy:.4f} "))
    Loss_val_list.append(float(f"{epoch_val_loss:.4f} "))
    Acc_val_list.append(float(f"{epoch_val_accuracy:.4f} "))
    print(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

end_time = time.time()
time_train = "run time: %d seconds" % (end_time - start_time)
print('终于TM训练完了，统共花了', time_train, 's')
torch.save(model.state_dict(), 'example.pt')

#
# 最简便的方法
## Loss
file = open('Loss2.txt', 'w')
for i in range(len(Loss_list)):
    s = str(Loss_list[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
    file.write(s)
file.close()
print("保存文件成功")

## Acc
file = open('accuracy2.txt', 'w')
for i in range(len(Acc_list)):
    s = str(Acc_list[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
    file.write(s)
file.close()
print("保存文件成功")

## Loss_val
file = open('Loss_val2.txt', 'w')
for i in range(len(Loss_val_list)):
    s = str(Loss_val_list[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
    file.write(s)
file.close()
print("保存文件成功")

## Acc_val
file = open('Acc_val2.txt', 'w')
for i in range(len(Acc_val_list)):
    s = str(Acc_val_list[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
    file.write(s)
file.close()
print("保存文件成功")
