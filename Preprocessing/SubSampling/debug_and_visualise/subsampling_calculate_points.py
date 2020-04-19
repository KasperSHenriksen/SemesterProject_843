import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def subsampling_algorithm(algorithm,non_subsampled_paths,parameter_value):
    for path in non_subsampled_paths:
        joined_path = f'{folder_path}/{path}'
        os.system(f'CloudCompare -SILENT -O {joined_path} -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS {algorithm} {parameter_value}')
    subsampled_paths = set(os.listdir(folder_path))-non_subsampled_paths
    return subsampled_paths

def get_total_points(subsampled_paths):
    defect_n_points = 0
    total_n_points = 0

    for path in subsampled_paths:
        joined_path = f'{folder_path}/{path}'
        file = open(joined_path)

        for line in file.readlines():
            label = int(line.split(',')[3][0])
            total_n_points += 1
            if label == 1:
                defect_n_points += 1

        file.close()
        os.remove(joined_path)

    total_n_points = total_n_points/len(subsampled_paths)
    defect_n_points = defect_n_points/len(subsampled_paths)
    return total_n_points, defect_n_points

def subsample(algorithm,non_subsampled_paths,min,max,step):
    total_points_list = []
    defect_points_list = []

    for parameter_value in np.arange(min,max,step):
        subsampled_paths = subsampling_algorithm(algorithm,non_subsampled_paths,parameter_value)
        total_n_points, defect_n_points = get_total_points(subsampled_paths)
        total_points_list.append([parameter_value,total_n_points])
        defect_points_list.append([parameter_value,defect_n_points])

    df_total = pd.DataFrame(np.array(total_points_list),columns=['Parameter Value','Points'])
    df_defect = pd.DataFrame(np.array(defect_points_list),columns=['Parameter Value','Points'])
    df_concat = pd.concat([df_total.assign(dataset='Total'),df_defect.assign(dataset='Defect')])
    return df_concat


os.chdir(input('Enter CloudCompare Path: '))
folder_path = input('Enter Dataset Path: ')

#folder_path = r'C:\Users\Kasper\Desktop\Compressed\Test\data'
non_subsampled_filepaths = set(os.listdir(folder_path))

octree_df = subsample('OCTREE',non_subsampled_filepaths,1,21,1)
octree_df = octree_df.rename(columns={'Parameter Value':'Octree Level'})
o_fig = sns.lineplot(x="Octree Level", y="Points",hue='dataset', ci=None, data=octree_df)
o_fig.set_yscale('log')

spatial_df = subsample('SPATIAL',non_subsampled_filepaths,.01,.15,.01)
spatial_df = spatial_df.rename(columns={'Parameter Value':'Minimum Distance'})
s_fig = sns.lineplot(x="Minimum Distance", y="Points",hue='dataset', ci=None, data=spatial_df)
s_fig.set_yscale('log')

random_df = subsample('RANDOM',non_subsampled_filepaths,38000,1,-1000)
random_df = random_df.rename(columns={'Parameter Value':'Remaining Points'})
r_fig = sns.lineplot(x="Remaining Points", y="Points",hue='dataset', ci=None, data=random_df)
r_fig.set_yscale('log')

plt.show()
