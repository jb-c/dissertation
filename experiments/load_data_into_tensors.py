from data.truecolours_parser import load_for_NANCY_UCEIS_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch


def return_X_y_train_and_test(lookback_window_in_days = 365,
                              test_frac=0.3,
                              random_state=42,
                              pickle_path = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/data/UC_and_CD_df.p'):
    '''
    A function to load and scale the data for UCEIS & NANCY regression
    Returns X_train, X_test, y_train, y_test
    '''

    # 1. Load data from pkl files
    X, y = load_for_NANCY_UCEIS_regression(path_to_pickle=pickle_path, lookback_window=lookback_window_in_days)

    # 2. Need to make all timeseries the same length (for batching) we ffill as dX_t = 0 and so has no impact on model
    max_length = np.max([len(xi) for xi in X])
    for i in range(len(X)):
        X[i] = X[i].drop(labels=['summary'], axis=1)
        ffill_df = X[i].iloc[np.repeat(-1, max_length - len(X[i]))]  # The last row of xi repeated correct # times
        X[i] = pd.concat([X[i], ffill_df], ignore_index=True)

    # 3. Stack into a big matrix
    X = np.stack(X)

    # 4. Scale/Standardise data
    X_scaler = StandardScaler() # SKlearn standard scaler
    X_scaled = X_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_scaled[:,:,0] = X[:,:,0]/X[:,:,0].max() # Rescale time coord to be in [0,1]
    y_scaled = y / np.array([8,4]) # Divide by max of NANCY / UCEIS clinical index

    # 5. Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,test_size = test_frac,random_state = random_state  )

    # 6. Convert to torch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test  = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test  = torch.from_numpy(y_test).float()

    return X_train, X_test, y_train, y_test
