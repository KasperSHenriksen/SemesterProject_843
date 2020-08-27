import h5py
import os
from tqdm.auto import tqdm
import numpy as np
import random
import glob


def get_filepaths(pathdir):
    filepaths = glob.glob(f'{pathdir}/**/*.csv',recursive=True)
    filepaths.extend(glob.glob(f'{pathdir}/**/*.CSV',recursive=True))
    return filepaths

def create_dataset_HDF5(pathdir,pointcloud_size):
    with h5py.File(f'{pathdir}/pointcloud_hdf5.h5','w') as hdf:
        defect_group = hdf.create_group('Defect')
        fine_group = hdf.create_group('Fine')

        groups = [defect_group,fine_group]
        class_folders = os.listdir(f'{pathdir}')

        zipping = zip(groups,class_folders)

        print(f'Converting cvs to hdf5 as: "pointcloud_hdf5.h5".')
        for group, class_folders in zipping:
            print(f'Converting {class_folders} cvs...')
            filepaths = get_filepaths(f'{pathdir}/{class_folders}')

            for idx, fp in enumerate(tqdm(filepaths)):
                array = np.genfromtxt(fp, delimiter=',', dtype = np.float32)
                group.create_dataset(f'pc{idx}',data=array)

def shuffle(pathdir):
    with h5py.File(f'{pathdir}/pointcloud_hdf5.h5','r') as hdf:
        combined_pointcloud_list = []
        combined_label_list = []
        for index, group in enumerate(list(hdf.keys())):
            pointcloud_list = list(hdf.get(f'{group}'))
            for pointcloud in pointcloud_list:
                pointcloud = hdf.get(f'{group}/{pointcloud}')

                pointcloud = np.array(pointcloud)

                combined_pointcloud_list.append(pointcloud)
                #combined_label_list.append(np.eye(2,dtype=np.int64)[index]) #Hot encoding
                label = np.array(index,dtype=np.int64)
                combined_label_list.append(np.array(label))

        temp_zip = list(zip(combined_pointcloud_list,combined_label_list))
        random.shuffle(temp_zip)
        pointclouds, labels = zip(*temp_zip)

        return pointclouds, labels

def combine_and_shuffle(pathdir,training,validation_ratio = 0.15):
    pointclouds, labels = shuffle(pathdir)
    with h5py.File(f'{pathdir}/pointcloud_hdf5.h5','w') as hdf:
        if(training == True):
            print(f'point clouds: {len(pointclouds)}. validation ratio: {validation_ratio}')
            size = int(len(pointclouds)*validation_ratio)

            #Validation Set
            validation_group = hdf.create_group('Validation')
            pointcloud_group = validation_group.create_dataset('PointClouds',data=pointclouds[-size:])
            label_group = validation_group.create_dataset('Labels', data=labels[-size:])

            #Training Set
            training_group = hdf.create_group('Training')
            pointcloud_group = training_group.create_dataset('PointClouds', data=pointclouds[:-size])
            label_group = training_group.create_dataset('Labels', data=labels[:-size])

        else:
            size = int(len(pointclouds))

            #Testing Set
            testing_group = hdf.create_group('Testing')
            pointcloud_group = testing_group.create_dataset('PointClouds',data=pointclouds)
            label_group = testing_group.create_dataset('Labels',data=labels)

            print(pointcloud_group[:])



path = input("Enter path to dataset: ")
training_or_testing = input("Training or Testing: ")

if(training_or_testing == 'Training'):
    validation_ratio = float(input("Enter the validation ratio for splitting (0.0-1.0): "))

    create_dataset_HDF5(path,1024)
    combine_and_shuffle(path,training = True,validation_ratio = validation_ratio)
else:
    create_dataset_HDF5(path,1024)
    combine_and_shuffle(path,training = False,validation_ratio = 0)


#Show info about the created dataset
with h5py.File(f'{path}/pointcloud_hdf5.h5','r') as hdf:

    for key in hdf.keys():
        print(key)
        print(list(hdf.get(f'{key}/')))
        print(hdf.get(f'{key}/Labels'))
        print(hdf.get(f'{key}/PointClouds'))
        print('\n')
