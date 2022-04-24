import os,sys,torch, torchcde, time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.append('../../')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda:1")


from neural_cdes.regression_cde import CDEModel
from data.preprocessing.load_for_training import load_data

#-----------------------------------------------------------------------------------------------------------------------
#
#                                NEURAL CDE FOR MIMIC3 Mortaility Prediction
#
#-----------------------------------------------------------------------------------------------------------------------

time_now = datetime.today().strftime("%Y_%m_%d_%H:%M")
save_dir    = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/experiments/mimic/saved_models'
save_folder = f'{save_dir}/{time_now}'

def train(num_epochs=5,batch_size=20,save_every_k_epochs=1,**kwargs):
    '''
    Main training loop for the MIMIC3 Classification Task
    '''
    # -----------------------------------------------------------------------------------
    #                                     KWARGS
    # -----------------------------------------------------------------------------------
    if 'interpolation_method' not in kwargs:
        kwargs['interpolation_method'] = 'cubic'
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
    X_train, X_test, y_train, y_test = load_data(data_source='mimic',return_tensors=True,local_or_global_time_scaling=kwargs['local_or_global_time_scaling'])

    # Add extra dim corresponding to channels = 1 dimension
    y_train = torch.unsqueeze(y_train, -1) ; y_test = torch.unsqueeze(y_test, -1)

    # Move tensors to GPU
    X_train=X_train.to(device);X_test=X_test.to(device)
    y_train=y_train.to(device);y_test=y_test.to(device)

    # Prep data by turning it into a continuous path
    if kwargs['interpolation_method'] == 'cubic':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
        test_coeffs  = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test)
    elif kwargs['interpolation_method'] == 'linear':
        train_coeffs = torchcde.linear_interpolation_coeffs(X_train)
        test_coeffs  = torchcde.linear_interpolation_coeffs(X_test)
    else:
        raise ValueError('Interpolation method not supported yet')

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, y_train)
    test_dataset = torch.utils.data.TensorDataset(test_coeffs, y_test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # -----------------------------------------------------------------------------------
    #                                   Load Model
    # -----------------------------------------------------------------------------------
    if kwargs['sde']:
        model = CDEModel(input_channels=X_train.size(-1),hidden_channels=kwargs['num_hidden_channels'],
                         output_channels=y_train.size(-1),interpolation_method=kwargs['interpolation_method'],
                         sde=True,dropout_p=kwargs['dropout_p'],adjoint=True,method='reversible_heun',dt=1)
    else:
        model = CDEModel(input_channels=X_train.size(-1),hidden_channels=kwargs['num_hidden_channels'],
                         output_channels=y_train.size(-1),interpolation_method=kwargs['interpolation_method'],
                         sde=False,adjoint=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # -----------------------------------------------------------------------------------
    #                                  Training
    # -----------------------------------------------------------------------------------
    start_time = time.time()
    print(f'Starting Training on {len(train_dataset)} samples and testing on {len(test_dataset)} samples')
    history = pd.DataFrame(columns=['epoch','train_loss','test_loss'])

    for epoch in range(num_epochs):
        for i, batch in tqdm(enumerate(dataloader), desc=f'\tBatch'):
            #############################################################
            optimizer.zero_grad()
            batch_coeffs, batch_y = batch
            batch_y = batch_y

            # Has size (batch, num_out_channels)
            pred_y = model(batch_coeffs)
            loss = torch.nn.functional.binary_cross_entropy(pred_y,batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #############################################################

            elapsed_time = time.time() - start_time

        # Make test predictions
        pred_y_train = model(train_coeffs)
        pred_y_test = model(test_coeffs)


        train_loss = torch.nn.functional.binary_cross_entropy(pred_y_train, y_train)
        test_loss = torch.nn.functional.binary_cross_entropy(pred_y_test, y_test)
        history.loc[len(history)] = [epoch + 1, train_loss.item(), test_loss.item()]

        print(f'Epoch: {epoch + 1}/{num_epochs}  Batch: {i + 1}/{len(dataloader)} Training loss: {train_loss.item()} Test Loss: {test_loss.item()} Elapsed Time : {np.round(elapsed_time / 60, 2)} mins')
        if ((epoch+1)%save_every_k_epochs) == 0:
            save_model(model,kwargs,save_folder, f'_at_epoch_{epoch+1}')
            print('\t\t Saved a Checkpoint')

    return model,history,kwargs






def save_model(model,kwargs,save_folder,name):
    '''
    Quick Wrapper to Save a Model To a dir
    '''
    if (not os.path.exists(save_folder)):
        os.mkdir(save_folder)

    torch.save(model.state_dict(), f'{save_folder}/model{name}.p')
    torch.save(kwargs,f'{save_folder}/kwargs{name}.p')



if __name__ == '__main__':
    model, history, kwargs = train(num_epochs = 100, batch_size = 500, sde = True, dropout_p = 0.25, t_pinball = 0.2)
    save_model(model,kwargs,save_folder,'')
    history.to_csv(f'{save_folder}/history.csv',index=False)
    print('Finished Training')
    torch.cuda.empty_cache()