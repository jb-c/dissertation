import pandas as pd
# The path to the MIMIC 3 Data folders, as formatted by https://github.com/YerevaNN/mimic3-benchmarks
# The root dir for all of the FORMATTED MIMIC3-data
root = '../../mimic3-benchmarks/data/'
root = '/mnt/sdd/MSc_projects/jbarker/code/mimic3-benchmarks/data/'

# A list of sub folders, one for each of the tasks commonly performed on the MIMIC3 dataset
task_names = ['in-hospital-mortality','length-of-stay','phenotyping']
selected_task_index = 0

# Task specific paths
task_root_path = f'{root}/{task_names[selected_task_index]}'
train_path = f'{task_root_path}/train'
test_path = f'{task_root_path}/test'

# Lists of individual episode file paths
# Ensures train/test_episode_paths are in same order as listfile
train_listfile_path = f'{train_path}/listfile.csv'
test_listfile_path  = f'{test_path}/listfile.csv'

train_listfile = pd.read_csv(train_listfile_path)
test_listfile  = pd.read_csv(test_listfile_path)

train_episode_paths = [f'{train_path}/{f}' for f in train_listfile.stay.values]
test_episode_paths  = [f'{test_path}/{f}' for f in test_listfile.stay.values]