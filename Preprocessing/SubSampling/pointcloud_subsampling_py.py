import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import time

def get_filepaths(main_folder):
    folders = set(os.listdir(f'{main_folder}'))
    non_subsampled_filepaths =[]
    for folder in folders:
        files = os.listdir(rf'{main_folder}/{folder}')
        for file in files:
            non_subsampled_filepaths.append(rf'{main_folder}/{folder}/{file}')

    return non_subsampled_filepaths

def subsampling_algorithm(main_folder,algorithm,parameter_value):
    non_subsampled_filepaths = get_filepaths(main_folder)


    for filepath in non_subsampled_filepaths:
        os.system(f'CloudCompare -SILENT -O {filepath} -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS {algorithm} {parameter_value}')
        #os.system(f'cloudcompare.CloudCompare -SILENT -O {filepath} -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS {algorithm} {parameter_value}')
        os.remove(filepath)


def remove_label_column(main_folder):
    print('[INFO] Remove label column...')
    filepaths = get_filepaths(main_folder)

    for file in tqdm(filepaths):
        try:
            old_f = pd.read_csv(f'{file}')
            new_f = old_f.drop(old_f.columns[3], axis=1)
            new_f.to_csv(f'{file}',index=False)
        except:
            print(file)

def delete_pointclouds_below_limit(main_folder,limit):
    print('[INFO] Deleting Pointclouds below limit...')
    filepaths = get_filepaths(main_folder)

    for file in tqdm(filepaths):
        old_f = pd.read_csv(f'{file}')
        if len(old_f)+1 < limit:
            os.remove(file)

main_folder = input('Path to data: ')
#main_folder = '/home/kasper/Desktop/Test_data/old_unity2'
cloud_compare_folder = input('Path to CloudCompare: ')
os.chdir(cloud_compare_folder)
#os.chdir('/snap/bin')

remove_label_column(main_folder)

start1 = time.time()
subsampling_algorithm(main_folder,'SPATIAL',0.08)
stop1 = time.time()

start2 = time.time()
subsampling_algorithm(main_folder,'RANDOM',1024)
stop2 = time.time()

print('Space:',stop1-start1)
print('Random:',stop2-start2)
#remove_label_column(main_folder)
delete_pointclouds_below_limit(main_folder,1024)
