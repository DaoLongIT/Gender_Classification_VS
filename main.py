import glob
import os
import os.path as osp
import random
from typing import Any
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision
from torchvision import models, transforms, datasets
import time
import copy

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self. data_transform[phase](img)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def make_datapath_list(phase="train"):
    rootpath = "./data/gender-classification-dataset/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list

data_dir = './data/gender-classification-dataset/'
image_transforms = ImageTransform(resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), image_transforms.data_transform[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


train_list = make_datapath_list("train")
val_list = make_datapath_list("val")

# MAKE DATASET
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        super().__init__()

    def __len__(self):
        return len(self.file_list)
    
    # tra cho minh buc anh dau ra de lay truyen vao network cua minh
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[45:53]
            if label == "female":
                label = 0
            else:
                label  = img_path[45:52]
                label = 1

        elif self.phase == "val":
            label = img_path[43:49]
            if label == "female":
                label = 0
            else:
                label  = img_path[43:47]
                label = 1

        return img_transformed, label

train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")

# DATALOADER
batch_size = 16

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)

dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# Create network
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=64)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=256)
        self.relu3=nn.ReLU()
        self.fc=nn.Linear(in_features=256*56*56,out_features=2)
    
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)   
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        output=output.view(-1, 256*56*56)  
        output=self.fc(output)

        return output


def train_model(model, criterior, optimizer, scheduler, num_epochs):
    # since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            
            for inputs, labels in dataloader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

model = CNN()
model = model.to(device)
#loss
criterior = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterior, optimizer, exp_lr_scheduler, num_epochs=10)
