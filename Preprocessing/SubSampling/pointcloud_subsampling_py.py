import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import time
import glob

def get_filepaths(main_folder):
    files = glob.glob(f'{main_folder}/**/*.csv',recursive=True)
    files.extend(glob.glob(f'{main_folder}/**/*.CSV',recursive=True))
    return files

def subsampling_algorithm(main_folder,algorithm,parameter_value):
    os_command = ''
    if os.name == 'nt':
        os_command = 'CloudCompare'
    elif os.name == 'posix':
        os_command = 'cloudcompare.CloudCompare'

    non_subsampled_filepaths = get_filepaths(main_folder)
    for filepath in non_subsampled_filepaths:
        print(filepath)
        os.system(f'{os_command} -SILENT -O {filepath} -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS {algorithm} {parameter_value}')
        os.remove(filepath)

def delete_pointclouds_below_limit(main_folder,limit):
    if (input_function('Delete pointclouds containing less than 1024 points? (y/n): ') == False):
        return

    print('[INFO] Deleting Pointclouds below limit...')
    filepaths = get_filepaths(main_folder)
    for file in tqdm(filepaths):
        old_f = pd.read_csv(f'{file}')
        if len(old_f)+1 < limit:
            os.remove(file)

def delete_defective_pcs_without_defective_points(main_folder):
    if (input_function('Delete pointclouds containing no defected points? (y/n): ') == False):
        return
    
    print('[INFO] Deleting Defective Pointclouds that have lost all their defective points...')
    filepaths = get_filepaths(main_folder+'/Defective')
    for file in tqdm(filepaths):
        csv = pd.read_csv(f'{file}')
        if any(csv.iloc[:, 3] == 1) != True:
            os.remove(file)

def remove_label_column(main_folder):
    if (input_function('Remove label for each point in point clouds? (y/n): ') == False):
        return

    print('[INFO] Remove label column...')
    filepaths = get_filepaths(main_folder)

    for file in tqdm(filepaths):
        try:
            old_f = pd.read_csv(f'{file}')
            new_f = old_f.drop(old_f.columns[3], axis=1)
            new_f.to_csv(f'{file}',index=False)
        except:
            print(file)
    
def input_function(message):
    answer = input(message).lower()
    while(True):
        if answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            answer = input(message).lower()


main_folder = input('Path to data: ')
cloud_compare_folder = input('Path to CloudCompare: ')
os.chdir(cloud_compare_folder)

subsampling_algorithm(main_folder,'SPATIAL',0.03)

subsampling_algorithm(main_folder,'RANDOM',1024)

delete_pointclouds_below_limit(main_folder,1024)

delete_defective_pcs_without_defective_points(main_folder)
remove_label_column(main_folder)