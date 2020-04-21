import h5py
import os
from tqdm.auto import tqdm
import numpy as np
import random

def CreateDatasetHDF5(pathdir,pointcloud_size):
    with h5py.File(f'{pathdir}/pointcloud_hdf5.h5','w') as hdf:
        defect_group = hdf.create_group('Defect')
        fine_group = hdf.create_group('Fine')

        groups = [defect_group,fine_group]
        folders = os.listdir(f'{pathdir}')

        zipping = zip(groups,folders)

        print(f'Converting cvs to hdf5 as: "pointcloud_hdf5.h5".')
        for group, folder in zipping:
            print(f'Converting {folder} cvs...')
            for idx,pointcloud in enumerate(tqdm(os.listdir(f'{pathdir}/{folder}'))):

                array = np.genfromtxt(f'{pathdir}/{folder}/{pointcloud}',delimiter=',')

                array = np.delete(array,slice(pointcloud_size,len(array)),0)

                group.create_dataset(f'pc{idx}',data=array)

def shuffle(pathdir):
    with h5py.File(f'{pathdir}/pointcloud_hdf5.h5','r') as hdf:
        combined_pointcloud_list = []
        combined_label_list = []

        for index, group in enumerate(list(hdf.keys())):
            pointcloud_list = list(hdf.get(f'{group}'))

            for pointcloud in pointcloud_list:
                pointcloud = hdf.get(f'{group}/{pointcloud}')
                #pointcloud = hdf.get(f'Defect/{pointcloud}')
                pointcloud = np.array(pointcloud)

                combined_pointcloud_list.append(pointcloud)
                combined_label_list.append(np.eye(2)[index])

        temp_zip = list(zip(combined_pointcloud_list,combined_label_list))
        random.shuffle(temp_zip)
        pointclouds, labels = zip(*temp_zip)

        return pointclouds, labels

def combine_and_shuffle(pathdir,validation_ratio = 0.15):
    pointclouds, labels = shuffle(pathdir)
    with h5py.File(f'{pathdir}/pointcloud_hdf5.h5','w') as hdf:
        size = int(len(pointclouds)*validation_ratio)

        #Validation Set
        validation_group = hdf.create_group('Validation')
        pointcloud_group = validation_group.create_group('PointClouds')
        label_group = validation_group.create_group('Labels')
        for idx,(pointcloud,label) in enumerate(zip(pointclouds[-size:],labels[-size:])):
            pointcloud_group.create_dataset(f'pc{idx}',data=pointcloud)
            label_group.create_dataset(f'pc{idx}',data=label)

        #Training Set
        training_group = hdf.create_group('Training')
        pointcloud_group = training_group.create_group('PointClouds')
        label_group = training_group.create_group('Labels')
        for idx,(pointcloud,label) in enumerate(zip(pointclouds[:-size],labels[:-size])):
            pointcloud_group.create_dataset(f'pc{idx}',data=pointcloud)
            label_group.create_dataset(f'pc{idx}',data=label)


path = input("Enter path to dataset: ")
pointcloud_size = int(input("Fixed number of points in point clouds: "))
validation_ratio = float(input("Enter the validation ratio for splitting the dataset into training and validation: "))

CreateDatasetHDF5(path,pointcloud_size)
combine_and_shuffle(path,validation_ratio = validation_ratio)


#Show info about the created dataset
with h5py.File(f'{path}/pointcloud_hdf5.h5','r') as hdf:

    for key in hdf.keys():
        print(key)
        print(list(hdf.get(f'{key}/')))
        print(hdf.get(f'{key}/Labels'))
        print(hdf.get(f'{key}/PointClouds'))
        print('\n')
