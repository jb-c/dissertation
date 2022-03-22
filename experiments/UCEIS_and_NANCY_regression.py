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

import torch, torchcde, time, os

device = torch.device("cuda")

from tqdm import tqdm

from neural_cdes.regression_cde import CDEModel as NeuralCDEModel
from data.preprocessing.load_for_training import load_data

#-----------------------------------------------------------------------------------------------------------------------
#
#                                NEURAL CDE FOR REGRESSING NANCY & UCEIS SCORES
#
#-----------------------------------------------------------------------------------------------------------------------

time_now = datetime.today().strftime("%Y_%m_%d_%H:%M")
save_dir    = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/experiments/saved_models'
save_folder = f'{save_dir}/{time_now}'

if (not os.path.exists(save_folder)) and (__name__ == '__main__'):
    os.mkdir(save_folder)

save_every_k_epochs = 1
num_hidden_channels = 32
interpolation_method = 'cubic'


def main(num_epochs = 5, batch_size = 50, sde = False, dropout_p = 0.5):
    '''
    Main training method for a neural CDE
    '''

    # -----------------------------------------------------------------------------------
    #                              Prep Data
    # -----------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = load_data(data_source='tc',return_tensors=True,lookback_window_in_days=500,update_pickle=True)

    # Move tensors to GPU
    X_train=X_train.to(device);X_test=X_test.to(device)
    y_train=y_train.to(device);y_test=y_test.to(device)

    # Prep data by turning it into a continuous path
    if interpolation_method == 'cubic':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
        test_coeffs  = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, y_train)
    test_dataset = torch.utils.data.TensorDataset(test_coeffs, y_test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # -----------------------------------------------------------------------------------
    #                                   Load Model
    # -----------------------------------------------------------------------------------
    if sde:
        model = NeuralCDEModel(input_channels = X_train.size(-1), hidden_channels = num_hidden_channels,
                               output_channels = y_train.size(-1), interpolation_method=interpolation_method,
                               sde=True,adjoint=True,method='reversible_heun',dropout_p = dropout_p,dt=0.1)
    else:
        model = NeuralCDEModel(input_channels = X_train.size(-1), hidden_channels = num_hidden_channels,
                               output_channels = y_train.size(-1), interpolation_method=interpolation_method)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # -----------------------------------------------------------------------------------
    #                                  Training
    # -----------------------------------------------------------------------------------
    start_time = time.time()
    print(f'Starting Training on {len(train_dataset)} samples and testing on {len(test_dataset)} samples')

    history = pd.DataFrame(columns=['epoch','train_loss','test_loss'])

    for epoch in range(num_epochs):
        for i, batch in tqdm(enumerate(dataloader),desc='\tBatch : '):
            #############################################################
            optimizer.zero_grad()
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #############################################################

            elapsed_time = time.time() - start_time

        # Make test predictions
        pred_y_train = model(train_coeffs).squeeze(-1)
        pred_y_test = model(test_coeffs).squeeze(-1)
        train_loss = torch.nn.functional.mse_loss(pred_y_train, y_train)
        test_loss = torch.nn.functional.mse_loss(pred_y_test, y_test)
        history.loc[len(history)] = [i+1,loss.item(), test_loss.item()]

        print(f'Epoch: {epoch+1}/{num_epochs}  Batch: {i+1}/{len(dataloader)} Training loss: {train_loss.item()} Test Loss: {test_loss.item()} Elapsed Time : {np.round(elapsed_time / 60,2)} mins')


        if ((epoch+1)%save_every_k_epochs) == 0:
            save_model(model, save_folder, f'_at_epoch_{epoch+1}')
            print('\t\t Saved a Checkpoint')

    return model, history



def save_model(model, save_folder, name):
    '''
    Quick Wrapper to Save a Model To a dir
    '''
    torch.save(model.state_dict(), f'{save_folder}/model{name}.p')
    torch.save({'input_channels': model.func.input_channels, 'hidden_channels': model.func.hidden_channels,'output_channels': model.output_channels,
                'interpolation_method': model.interpolation_method,'sde':model.is_sde,'dropout_p':model.dropout_p}, f'{save_folder}/args{name}.p')


if __name__ == '__main__':
    model, history = main(num_epochs = 100, batch_size = 50, sde = True, dropout_p = 0.5)
    save_model(model,save_folder,'')
    history.to_csv(f'{save_folder}/history.csv',index=False)
    print('Finished Training')
    torch.cuda.empty_cache()
