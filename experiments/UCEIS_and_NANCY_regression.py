import os
import sys
sys.path.append('../')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting.metrics import MAPE
from datetime import datetime

import torch
import torchcde

device = torch.device("cuda:0")

from tqdm import tqdm

from neural_cdes.regression_cde import NeuralCDEModel
from data.truecolours_parser import load_for_forecasting,load_for_NANCY_UCEIS_regression

#-----------------------------------------------------------------------------------------------------------------------
#
#                                NEURAL CDE FOR REGRESSING NANCY & UCEIS SCORES
#
#-----------------------------------------------------------------------------------------------------------------------

pickle_path = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/data/UC_and_CD_df.p'
save_dir    = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/experiments/saved_models'
lookback_window_in_days = 365

num_hidden_channels = 32
interpolation_method = 'cubic'


def main(num_epochs = 5, train_frac = 0.8):
    # -----------------------------------------------------------------------------------
    #                               Loading & Parsing Data
    # -----------------------------------------------------------------------------------
    X,y =  load_for_NANCY_UCEIS_regression(path_to_pickle=pickle_path, lookback_window=lookback_window_in_days)


    # Need to make all timeseries the same length (for batching) we ffill as dX_t = 0 and so has no impact on model
    max_length = np.max([len(xi) for xi in X])
    for i in range(len(X)):
        ffill_df = X[i].iloc[np.repeat(-1,max_length - len(X[i]))] # The last row of xi repeated correct # times
        X[i] = pd.concat([X[i],ffill_df],ignore_index=True)
    X = np.stack(X)


    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    X_tensor = torch.nn.functional.normalize(X_tensor)
    y_tensor = torch.nn.functional.normalize(y_tensor)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    print(f'X.shape = {X.shape}\ny.shape = {y.shape}')

    # -----------------------------------------------------------------------------------
    #                              Load Model & Prep Data
    # -----------------------------------------------------------------------------------
    model = NeuralCDEModel(input_channels = X_tensor.size(-1), hidden_channels = num_hidden_channels,
                           output_channels = y_tensor.size(-1), interpolation_method=interpolation_method)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Prep data by turning it into a continuous path
    if interpolation_method == 'cubic':
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_tensor)

    dataset = torch.utils.data.TensorDataset(coeffs, y_tensor)

    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50)

    # -----------------------------------------------------------------------------------
    #                                  Training
    # -----------------------------------------------------------------------------------
    import  time
    start_time = time.time()


    print(f'Starting Training on {len(train_dataset)} samples and testing on {len(test_dataset)} samples')
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred_y, batch_y)
            print(f'\t Perfoming backprop for batch {i+1}/{len(dataloader)}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            elapsed_time = time.time() - start_time
            print(f'Epoch: {epoch+1}/{num_epochs}  Batch: {i+1}/{len(dataloader)} Training loss: {loss.item()} Elapsed Time : {np.round(elapsed_time / 60,2)} mins')

            # Make test predictions
            test_coeffs, test_y = test_dataset[:]
            pred_y_test = model(test_coeffs).squeeze(-1)
            test_loss_MAPE = MAPE()(pred_y_test, test_y)
            test_loss_MSE  = torch.nn.functional.mse_loss(pred_y_test, test_y)
            print(f'\t Test loss MAPE : {test_loss_MAPE} and Test loss MSE : {test_loss_MSE}')


    return model, coeffs, X_tensor, y_tensor

if __name__ == '__main__':
    model, coeffs, X_tensor, y_tensor = main(10,0.9)
    time_now = datetime.today().strftime("%Y_%m_%d_%H:%M")
    torch.save(model.state_dict(), f'{save_dir}/model_{time_now}.p')
    torch.save({'input_channels': model.func.input_channels, 'hidden_channels': model.func.hidden_channels,
                'output_channels': model.output_channels, 'interpolation_method': model.interpolation_method},
               f'{save_dir}/args_{time_now}.p')
