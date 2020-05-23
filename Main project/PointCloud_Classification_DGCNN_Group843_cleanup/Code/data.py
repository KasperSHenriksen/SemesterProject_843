import os
import h5py
import numpy as np
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self,num_points,partition):
        self.data, self.labels = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self,item):
        data = self.data[item][:self.num_points]
        labels = self.labels[item]
        if self.partition == 'Training':
            np.random.shuffle(data)
        return data, labels

    def __len__(self):
        return self.data.shape[0]

def print_data_info(partition, data, labels):
    print(f'[{partition} Data]')
    print(f'Data Shape: {data.shape} | Type: {data[0].dtype}')
    print(f'Label Shape: {labels.shape} | Type: {labels[0].dtype} \n')

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if(partition == 'Training' or partition == 'Validation'):
        with h5py.File(f'{DATA_DIR}/pointcloud_hdf5.h5','r') as hdf:
            data = hdf[f'{partition}/PointClouds'][:]
            labels = hdf[f'{partition}/Labels'][:]

            print_data_info(partition,data,labels)
            return data, labels

    if(partition == 'Testing'):
        with h5py.File(f'{DATA_DIR}/test_pointcloud_hdf5_v3.h5','r') as hdf: #test_pointcloud_hdf5.h5
            data = hdf[f'{partition}/PointClouds'][:]
            labels = hdf[f'{partition}/Labels'][:]

            print_data_info(partition,data,labels)
            return data, labels
