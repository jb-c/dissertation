import sys
sys.path.append('../')
import data.mimic3_path as path
import pandas as pd


def load():
    '''
    Loads the mimic3 data for the selevted task (defined in path.py file)
    '''
    train_list_file_df = pd.read_csv(path.train_listfile_path)
    train_dfs = [pd.read_csv(f) for f in path.train_episode_paths]

    test_list_file_df = pd.read_csv(path.train_listfile_path)
    test_dfs = [pd.read_csv(f) for f in path.train_episode_paths]

    y_train = train_list_file_df.y_true.values
    y_test  = test_list_file_df.y_true.values



    return train_dfs,test_dfs,y_train,y_test