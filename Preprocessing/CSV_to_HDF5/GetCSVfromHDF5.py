import h5py
import os
import numpy as np


hdf5_pathname = '/home/kasper/Desktop/test_dataset/NewDataTest/pointcloud_hdf5.h5'

with h5py.File(hdf5_pathname,'r') as hdf:
    for key in hdf.keys():
        print(key)
        print(list(hdf[key]['Labels']))
        pointcloud = list(hdf[key]['PointClouds'])[1]
        array = np.asarray(pointcloud)
        np.savetxt("pointcloud_new.csv",array)
        break
