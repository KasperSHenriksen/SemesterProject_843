from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import random

#Chooses the available device from your PC. It priotises GPU.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


#Creates dataset of tensors from csv files.
def CreateDataset(pathdir):
    tensors = []
    labels = []

    for idx, folder in enumerate(os.listdir(f'{pathdir}')):
        for filename in tqdm(os.listdir(f'{pathdir}/{folder}')):
            array = np.genfromtxt(f'{pathdir}/{folder}/{filename}',delimiter=',')
            tensor = torch.from_numpy(array).to(device)
            tensors.append(tensor)
            labels.append(np.eye(2)[idx]) #Hot encoded

    temp_list = list(zip(tensors,labels))
    random.shuffle(temp_list)

    tensors, labels = zip(*temp_list)

    return tensors, labels

tensors, labels = CreateDataset('../input/PointCloudv1')
