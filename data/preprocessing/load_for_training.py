import sys,torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.mimic3_loader import load as load_mimc_data
from data.truecolours_parser import load_for_NANCY_UCEIS_regression as load_tc_data
from data.truecolours_parser import load_for_NANCY_UCEIS_series_to_series as load_tc_data_multilabel
from data.truecolours_parser import load_for_forecasting as load_tc_data_series
from sklearn.preprocessing import StandardScaler


def load_data(data_source='tc', scale_data=True, stack_data = True, return_tensors = False ,**kwargs):
    '''
    Loads data for the task we choose
    '''
    X_train, X_test, y_train, y_test = load_train_test_split(data_source,**kwargs)

    #----------------------------------   Scale Data      ---------------------------------------------------
    # We scale X by a standard scaler, and then y either not at all or by 4/8 for the Nancy/UCEIS.
    #--------------------------------------------------------------------------------------------------------
    if scale_data:
        if 'local_or_global_time_scaling' not in kwargs:
            kwargs['local_or_global_time_scaling'] = 'local'

        X_train_scaled, X_test_scaled = scale_data_helper_func(X_train,X_test,scale_time_col_separately = True, local_or_global_time_scaling = kwargs['local_or_global_time_scaling'])
        y_train_scaled, y_test_scaled = np.array([yi.copy() for yi in y_train]), np.array([yi.copy() for yi in y_test])

        if data_source == 'tc_multi_label':
            # Scale the time col on y as well, so it matches up with X_scaled
            # Important in load_for_NANCY_UCEIS_series_to_series we don't discard nans so series have same len
            for yi,zi in zip([*y_train_scaled,*y_test_scaled],[*X_train_scaled,*X_test_scaled]):
                yi['time'] = zi['time'].values # .values to ignore index
                yi[['UCEIS (0 to 8)','Nancy Index']] /= np.array([8,4])
        elif data_source == 'tc':
            y_train_scaled /= np.array([8,4])
            y_test_scaled /= np.array([8,4])
    else:
        X_train_scaled, X_test_scaled = [xi.copy() for xi in X_train], [xi.copy() for xi in X_test]
        y_train_scaled, y_test_scaled = np.array([yi.copy() for yi in y_train]), np.array([yi.copy() for yi in y_test])

    #----------------------------------   Stack Data      ---------------------------------------------------
    # Stack X up
    #--------------------------------------------------------------------------------------------------------

    if stack_data:
        max_length = np.max([len(xi) for xi in [*X_train, *X_test]])

        # ffill and stack the train and test dfs
        for i in range(len(X_train)):
            ffill_df = X_train_scaled[i].iloc[
                np.repeat(-1, max_length - len(X_train_scaled[i]))]  # The last row of xi repeated correct # times
            X_train_scaled[i] = pd.concat([X_train_scaled[i], ffill_df], ignore_index=True)
        for j in range(len(X_test)):
            ffill_df = X_test_scaled[j].iloc[
                np.repeat(-1, max_length - len(X_test_scaled[j]))]  # The last row of xi repeated correct # times
            X_test_scaled[j] = pd.concat([X_test_scaled[j], ffill_df], ignore_index=True)

        X_train_scaled = np.stack(X_train_scaled)
        X_test_scaled = np.stack(X_test_scaled)

    if return_tensors:
        return to_tensor(X_train_scaled), to_tensor(X_test_scaled),to_tensor(y_train_scaled),to_tensor(y_test_scaled)
    else:
        return X_train_scaled, X_test_scaled, y_train_scaled,y_test_scaled


def load_train_test_split(data_source = 'tc',**kwargs):
    '''
    Loads data into X_train,X_test,y_train,y_test
    '''
    if data_source.startswith('tc'):
        #################################################################################################
        #-------------------------------------- True Colours Data  --------------------------------------
        #################################################################################################

        #### Kwargs unpacking/sorting out ####
        if 'lookback_window' not in kwargs:
            kwargs['lookback_window'] = 60
        if 'test_frac' not in kwargs:
            kwargs['test_frac'] = 0.15
        if 'pickle_path' not in kwargs:
            kwargs['pickle_path'] = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/data/UC_and_CD_df.p'
        if 'update_pickle' not in kwargs:
            kwargs['update_pickle'] = False
        if 'drop_summary_col' not in kwargs:
            kwargs['drop_summary_col'] = True

        #### Load tc data for specific task ####
        if data_source == 'tc':
            X_dfs,y = load_tc_data(path_to_pickle=kwargs['pickle_path'],update_pickle=kwargs['update_pickle'],lookback_window=kwargs['lookback_window'])
        elif data_source == 'tc_multi_label':
            X_dfs,y, _, _ = load_tc_data_multilabel(path_to_pickle=kwargs['pickle_path'],update_pickle=kwargs['update_pickle'])
        elif data_source == 'tc_timeseries':
            X_dfs,X_dfs_CD = load_tc_data_series(path_to_pickle=kwargs['pickle_path'],update_pickle=kwargs['update_pickle'])
            y = np.ones(len(X_dfs)) # dummy y so the rest of the code in this file doesn't break

        #### Drop cols if needed ####
        if kwargs['drop_summary_col']:
            X_dfs = [xi.drop(labels=['summary'], axis=1) for xi in X_dfs]
        X_train, X_test, y_train, y_test = train_test_split(X_dfs, y, test_size=kwargs['test_frac'],random_state=101)

    elif data_source == 'mimic':
        #################################################################################################
        # --------------------------------------   Mimic Data      --------------------------------------
        #################################################################################################
        if 'fill_na' not in kwargs:
            kwargs['fill_na'] = True
        X_train, X_test, y_train, y_test = load_mimc_data(fill_na=kwargs['fill_na'])

    return  X_train, X_test, y_train, y_test



















def to_tensor(a):
    return torch.from_numpy(a).float()

def scale_data_helper_func(X_train,X_test,scale_time_col_separately = True, local_or_global_time_scaling = 'local'):
    '''
    Using a standard scaler, returns scaled data.
    We fit the scaler only on the train data
    '''
    scaler = StandardScaler().fit(np.vstack(X_train).reshape(-1, np.vstack(X_train).shape[-1]))

    X_train_scaled = [pd.DataFrame(scaler.transform(xi.values),columns=X_train[0].columns) for xi in X_train]
    X_test_scaled = [pd.DataFrame(scaler.transform(xi.values),columns=X_test[0].columns) for xi in X_test]

    if scale_time_col_separately and (local_or_global_time_scaling == 'global'):
        time_global_min = np.max([xi.loc[:,'time'].min() for xi in X_train])
        time_global_max = np.max([xi.loc[:,'time'].max() for xi in X_train])

        X_train_scaled = [local_time_col_scale_func(X_train_scaled[i],X_train[i],time_global_min,time_global_max) for i in range(len(X_train_scaled))]
        X_test_scaled = [local_time_col_scale_func(X_test_scaled[i],X_test[i],time_global_min,time_global_max) for i in range(len(X_test_scaled))]

    elif scale_time_col_separately and (local_or_global_time_scaling == 'local'):
        X_train_scaled = [local_time_col_scale_func(X_train_scaled[i], X_train[i]) for i in range(len(X_train_scaled))]
        X_test_scaled = [local_time_col_scale_func(X_test_scaled[i], X_test[i]) for i in range(len(X_test_scaled))]

    return X_train_scaled, X_test_scaled



def local_time_col_scale_func(x,x_reference = None,min_value = None, max_value = None):
    '''
    Lil helper function to scale the time column in days
    '''
    if min_value is None:
        min_value = x_reference.iloc[:,0].min()
    if max_value is None:
        max_value = x_reference.iloc[:,0].max()
    if x_reference is None:
        x_reference = x

    x.iloc[:,0] = (x_reference.iloc[:,0].values - min_value) / (max_value - min_value)
    return x

