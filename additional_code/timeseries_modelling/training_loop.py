import os,sys,torch, torchcde, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

sys.path.append('../../')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
device = torch.device("cuda")

from data.preprocessing.load_for_training import load_data, to_tensor

#-----------------------------------------------------------------------------------------------------------------------
#
#                                NEURAL ODE for Time Series Forecasting
#
#-----------------------------------------------------------------------------------------------------------------------

time_now = datetime.today().strftime("%Y_%m_%d_%H:%M")
save_dir    = '/experiments/UCEIS_and_NANCY_regression/saved_models'
save_folder = f'{save_dir}/{time_now}'

def train(num_epochs=5,batch_size=20,save_every_k_epochs=1,**kwargs):
    '''
    Main training loop for the UCEIS and NANCY regression task
    '''
    # -----------------------------------------------------------------------------------
    #                                     KWARGS
    # -----------------------------------------------------------------------------------
    if 'lookback_window' not in kwargs:
        kwargs['lookback_window'] = 10
        L = kwargs['lookback_window']
    if 'horizon_length' not in kwargs:
        kwargs['horizon_length'] = 20
        H = kwargs['horizon_length']
    if 'num_hidden_channels' not in kwargs:
        kwargs['num_hidden_channels'] = 32
    if 'dropout_p' not in kwargs:
        kwargs['dropout_p'] = 0.5
    if 'sde' not in kwargs:
        kwargs['sde'] = False
    if 'local_or_global_time_scaling' not in kwargs:
        kwargs['local_or_global_time_scaling'] = 'local'

    # -----------------------------------------------------------------------------------
    #                                    Prep Data
    # -----------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = load_data(data_source='tc_timeseries',return_tensors=False,scale_data=True,stack_data=False,local_or_global_time_scaling=kwargs['local_or_global_time_scaling'])

    # Filter to those with enough data to perform context/forecast split
    X_train_filtered,X_test_filtered = [], []
    for X_arr,T_arr in zip([X_train,X_test],[X_train_filtered,X_test_filtered]):
        for xi in X_arr:
            ni = int(np.ceil(len(xi) / (L + H)))
            for i in range(1,ni):
                T_arr.append(xi.iloc[(i-1)*(L+H):i*(L+H)])
    # Split up time series and then stack the data
    X_train_filtered, X_test_filtered = np.stack(X_train_filtered), np.stack(X_test_filtered)
    X_train,y_train = X_train_filtered[:,:L,:], X_train_filtered[:,L:,:]
    X_test, y_test  = X_test_filtered[:,:L,:], X_test_filtered[:,L:,:]

    # Move tensors to device
    X_train=to_tensor(X_train);X_test=to_tensor(X_test);y_train=to_tensor(y_train);y_test=to_tensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # -----------------------------------------------------------------------------------
    #                                   Load Model
    # -----------------------------------------------------------------------------------


    return X_train,y_train, X_test,y_test