import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


#if __name__ == '__main__':
#    train = ModelNet40(1024)
#    test = ModelNet40(1024, 'test')
#    for data, label in train:
#        print(data.shape)
#        print(label.shape)


#####################################
#NEW
#####################################
#f = h5py.File(h5_name)
#        data = f['data'][:].astype('float32')
#        label = f['label'][:].astype('int64')

class PointCloudDataset(Dataset):
    def __init__(self,num_points,partition):
        self.data, self.labels = load_data3(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self,item):
        data = self.data[item][:self.num_points]
        labels = self.labels[item]
        if self.partition == 'Training':
            #pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(data)
        return data, labels

    def __len__(self):
        return self.data.shape[0]
#for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):

def load_data2(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')


    f = h5py.File(f'{DATA_DIR}/modelnet40_ply_hdf5_2048.h5')
    print(f.get('Training/PointClouds/pc0'))
    #array = f.get(f['Training/PointClouds/pc0'])
    return
    #for h5_name in glob.glob(f'{DATA_DIR}/modelnet40_ply_hdf5_2048.h5'):
    #    print('This here: ',h5_name)
    #with h5py.File(f'{DATA_DIR}/modelnet40_ply_hdf5_2048.h5','r') as hdf:
        #print(hdf.get(f'{partition}'))
        #print(hdf.get(f'{partition}/Labels'))
        #print(hdf.get(f'{partition}/PointClouds'))
        #data = f['data'][:].astype('float32')
        #label = f['label'][:].astype('int64')

        #data = []
        #for pointcloud in hdf[f'{partition}/PointClouds']:
            #array = hdf.get(f'{partition}/PointClouds/{pointcloud}').astype('float32')
            #array = hdf[{pointcloud}]
            #data.append(array)
        #data = np.array(data)
        #print(data.shape)

        #labels = []
        #for label in hdf[f'{partition}/Labels']:
        #    array = hdf.get(f'{partition}/Labels/{pointcloud}').astype('int64')
        #    labels.append(array)
        #labels = np.array(labels)
        #print(labels.shape)

        #return data, labels

def load_data3(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if(partition == 'Training' or partition == 'Validation'):
        with h5py.File(f'{DATA_DIR}/pointcloud_hdf5.h5','r') as hdf:
            print(hdf.get(f'{partition}'))
            print(hdf.get(f'{partition}/Labels'))
            print(hdf.get(f'{partition}/PointClouds'))

            data = []
            for pointcloud in hdf[f'{partition}/PointClouds']:
                array = np.array(hdf.get(f'{partition}/PointClouds/{pointcloud}'))
                data.append(array)
            data = np.array(data)
            print(data.shape)

            labels = []
            for label in hdf[f'{partition}/Labels']:
                array = np.array(hdf.get(f'{partition}/Labels/{pointcloud}'))
                labels.append(array)
            labels = np.array(labels)

            print(f'Data Shape: {data.shape} | Type: {data[0].dtype}')
            print(f'Label Shape: {labels.shape} | Type: {labels[0].dtype}')

            return data, labels
    
    if(partition == 'Testing'):
        with h5py.File(f'{DATA_DIR}/pointcloud_hdf5_testing.h5','r') as hdf:
            print(hdf.get(f'{partition}'))
            print(hdf.get(f'{partition}/Labels'))
            print(hdf.get(f'{partition}/PointClouds'))

            data = []
            for pointcloud in hdf[f'{partition}/PointClouds']:
                array = np.array(hdf.get(f'{partition}/PointClouds/{pointcloud}'))
                data.append(array)
            data = np.array(data)
            print(data.shape)

            labels = []
            for label in hdf[f'{partition}/Labels']:
                array = np.array(hdf.get(f'{partition}/Labels/{pointcloud}'))
                print(array)
                labels.append(array)
            labels = np.array(labels)

            print(f'Data Shape: {data.shape} | Type: {data[0].dtype}')
            print(f'Label Shape: {labels.shape} | Type: {labels[0].dtype}')

            return data, labels

def read_h5():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    with h5py.File(f'{DATA_DIR}/modelnet40_ply_hdf5_2048.h5','r') as hdf:
        for key in hdf.keys():
            print(key)
            print(list(hdf.get(f'{key}/')))
            print(hdf.get(f'{key}/Labels'))
            print(hdf.get(f'{key}/PointClouds'))
            print('\n')

if __name__ == '__main__':
    #read_h5('Training')
    #load_data2('Training')
    #load_data3('Training')
    load_data3('Testing')
