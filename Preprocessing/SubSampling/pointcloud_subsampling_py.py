import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm


def get_filepaths(main_folder):
    folders = set(os.listdir(f'{main_folder}'))
    non_subsampled_filepaths =[]
    for folder in folders:
        files = os.listdir(f'{main_folder}\\{folder}')
        for file in files:
            non_subsampled_filepaths.append(f'{main_folder}\{folder}\{file}')

    return non_subsampled_filepaths

def subsampling_algorithm(main_folder,algorithm,parameter_value):
    non_subsampled_filepaths = get_filepaths(main_folder)

    for filepath in non_subsampled_filepaths:
        os.system(f'CloudCompare -SILENT -O {filepath} -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS {algorithm} {parameter_value}')
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
            print('DELET')
            os.remove(file)

os.chdir(r'C:\Program Files\CloudCompare')
main_folder = r'C:\Users\Kasper\Desktop\Compressed\Ham'

subsampling_algorithm(main_folder,'SPATIAL',0.08)
subsampling_algorithm(main_folder,'RANDOM',1024)

remove_label_column(main_folder)
delete_pointclouds_below_limit(main_folder,1024)
