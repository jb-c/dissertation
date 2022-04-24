import sys
sys.path.append('../')
import data.mimic3_path as path
import pandas as pd
import numpy as np


def load(fill_na = True):
    '''
    Loads the mimic3 data for the selevted task (defined in path.py file)
    '''
    train_list_file_df = pd.read_csv(path.train_listfile_path)
    test_list_file_df = pd.read_csv(path.test_listfile_path)
    
    # Cut some data
    ########################################
    frac = 0.05; N_train = int(frac*len(train_list_file_df)); N_test  = int(frac*len(test_list_file_df)) # frac = 0.25
    #frac = 0.05; N_train = int(10); N_test  = int(10) # frac = 0.25


    np.random.seed(1)

    I_train = np.random.choice(np.arange(len(train_list_file_df)), size=N_train).astype(int)
    I_test  = np.random.choice(np.arange(len(test_list_file_df)), size=N_test).astype(int)
    
    train_list_file_df = train_list_file_df.iloc[I_train]
    test_list_file_df = test_list_file_df.iloc[I_test]

    train_episode_paths = [f'{path.train_path}/{f}' for f in train_list_file_df.stay.values]
    test_episode_paths = [f'{path.test_path}/{f}' for f in test_list_file_df.stay.values]

    ########################################
    print(f'Loading MIMIC Data ({N_train}, {N_test})')
    
    train_dfs = [pd.read_csv(f) for f in train_episode_paths]
    test_dfs = [pd.read_csv(f) for f in test_episode_paths]

    y_train = train_list_file_df.y_true.values
    y_test  = test_list_file_df.y_true.values

    print("MIMIC Data Loaded")

    ## Drop categorical cols
    cols_to_drop = ['Glascow coma scale motor response', 'Glascow coma scale total','Glascow coma scale verbal response','Glascow coma scale eye opening','Capillary refill rate']
    test_dfs  = [test_df.select_dtypes(['number']).drop(cols_to_drop,axis=1,errors='ignore') for test_df in test_dfs]
    train_dfs = [train_df.select_dtypes(['number']).drop(cols_to_drop,axis=1,errors='ignore') for train_df in train_dfs]

    ## Rename col
    test_dfs  = [test_df.rename(columns={'Hours':'time'}) for test_df in test_dfs]
    train_dfs = [train_df.rename(columns={'Hours':'time'}) for train_df in train_dfs]

    ## Filter
    if False:
        test_dfs =  [df for df in test_dfs if keep_mimic_df(df)]
        train_dfs = [df for df in train_dfs if keep_mimic_df(df)]

    print("\t Cols Sorted")

    if fill_na:
        ## Fill first and last row with col mean
        col_means = np.nanmean(np.vstack(train_dfs), axis=0)
        nan_dict = {train_dfs[0].columns[i]: mu for i, mu in enumerate(col_means)}

        # Iterate over so don't get set value on slice error
        for i in range(len(train_dfs)):
            train_dfs[i].iloc[[0, -1]] = train_dfs[i].iloc[[0, -1]].fillna(nan_dict, axis=0)
        for i in range(len(test_dfs)):
            test_dfs[i].iloc[[0, -1]] = test_dfs[i].iloc[[0, -1]].fillna(nan_dict, axis=0)


        ## Then interpolate
        if fill_na in ['linear','cubic']:
            print("Interplating NANS in MIMIC3 Loader File")
        # Iterate over so don't get set value on slice error
            for i in range(len(train_dfs)):
                train_dfs[i] = train_dfs[i].interpolate(method=fill_na)
            for i in range(len(test_dfs)):
                test_dfs[i]  = test_dfs[i].interpolate(method=fill_na)

    return train_dfs,test_dfs,np.array(y_train),np.array(y_test)


















def keep_mimic_df(df, number_of_vars_with_20_or_more_observations=5):
    '''
    Returns a bool - indicating if the time series is dodgy or not
    '''
    length_of_stay = df['time'].iloc[-1]

    # Have valid duration
    have_valid_duration = (4 <= length_of_stay) and (length_of_stay <= 72)

    # Count number of observations of each variable
    counts = (~df[[c for c in df.columns if c != 'time']].isna()).sum()
    temp = np.partition(-counts, number_of_vars_with_20_or_more_observations)
    highest_counts = -temp[:number_of_vars_with_20_or_more_observations]
    have_enough_data = highest_counts.min() > 20

    # Quick sanity checks
    not_null = len(df) > 10

    return have_valid_duration and have_enough_data and not_null