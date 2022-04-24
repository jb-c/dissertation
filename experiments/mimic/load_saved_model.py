import sys,torch, torchcde,os
import pandas as pd
import numpy as np
sys.path.append('../../')
from neural_cdes.regression_cde import CDEModel
from data.preprocessing.load_for_training import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda:1")


def main(model_name,return_numpy = True,at_epoch = None,root_dir = '/mnt/sdd/MSc_projects/jbarker/code/dissertation/experiments/mimic/saved_models', **kwargs_into_func_call):
    '''
    Helpful function to load a model and return it along with predictions - mostly copying the training loop code
    '''
    at_epoch_str = '' if (at_epoch is None) else f'_at_epoch_{at_epoch}'
    kwargs = torch.load(f'{root_dir}/{model_name}/kwargs{at_epoch_str}.p')

    try:
        history = history = pd.read_csv(f'{root_dir}/{model_name}/history.csv')
    except:
        history = pd.DataFrame()

    if 'sde' in kwargs_into_func_call:
        kwargs['sde'] = kwargs_into_func_call['sde']
    if 'return_predictions' not in kwargs_into_func_call:
        kwargs_into_func_call['return_predictions'] = True
    if 'num_stochastic_samples' not in kwargs_into_func_call:
        kwargs_into_func_call['num_stochastic_samples'] = 10


    # -----------------------------------------------------------------------------------
    #                                    Prep Data
    # -----------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = load_data(data_source='mimic')
    #X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_data(data_source=data_source,stack_data=False,scale_data=False,lookback_window=kwargs['lookback_window'],local_or_global_time_scaling=kwargs['local_or_global_time_scaling'])

    # Add extra dim corresponding to channels = 1 dimension
    y_train = torch.unsqueeze(y_train, -1) ; y_test = torch.unsqueeze(y_test, -1)

    # Move tensors to GPU
    X_train=X_train.to(device);X_test=X_test.to(device)
    y_train=y_train.to(device);y_test=y_test.to(device)

    # -----------------------------------------------------------------------------------
    #                                   Load Model
    # -----------------------------------------------------------------------------------
    if kwargs['sde']:
        model = CDEModel(input_channels=X_train.size(-1),hidden_channels=kwargs['num_hidden_channels'],
                         output_channels=y_train.size(-1),interpolation_method=kwargs['interpolation_method'],
                         sde=True,dropout_p=kwargs['dropout_p'],adjoint=True,method='reversible_heun',dt=0.1)
    else:
        model = CDEModel(input_channels=X_train.size(-1),hidden_channels=kwargs['num_hidden_channels'],
                         output_channels=y_train.size(-1),interpolation_method=kwargs['interpolation_method'],
                         sde=False,adjoint=True)

    model.load_state_dict(torch.load(f'{root_dir}/{model_name}/model{at_epoch_str}.p', map_location=device))
    model.eval()
    print("Model Loaded")
    # -----------------------------------------------------------------------------------
    #                                   Make Predictions
    # -----------------------------------------------------------------------------------
    # Prep data by turning it into a continuous path
    if kwargs['interpolation_method'] == 'cubic':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
        test_coeffs  = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test)
    elif kwargs['interpolation_method'] == 'linear':
        train_coeffs = torchcde.linear_interpolation_coeffs(X_train)
        test_coeffs  = torchcde.linear_interpolation_coeffs(X_test)
    else:
        raise ValueError('Interpolation method not supported yet')


    if kwargs_into_func_call['return_predictions']:
        print("Computing predictions")
        if kwargs['sde']:
            y_hat_train = model(train_coeffs, return_path=True, num_stochastic_samples=kwargs_into_func_call['num_stochastic_samples'])
            y_hat_test =  model(test_coeffs, return_path=True, num_stochastic_samples=kwargs_into_func_call['num_stochastic_samples'])
        else:
            y_hat_train = model(train_coeffs, return_path=True)
            y_hat_test =  model(test_coeffs, return_path=True)


        # Aggregate samples
        if kwargs['sde']:
            mu_train = y_hat_train.mean(axis=1)
            sigma_train = y_hat_train.std(axis=1)

            y_hat_train = torch.stack([mu_train-sigma_train,mu_train,mu_train+sigma_train],axis=-1)


            mu_test = y_hat_test.mean(axis=1)
            sigma_test = y_hat_test.std(axis=1)

            y_hat_test = torch.stack([mu_test-sigma_test,mu_test,mu_test+sigma_test],axis=-1)




    if kwargs_into_func_call['return_predictions']:
        if return_numpy:
            return [X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy(), y_hat_train.detach().numpy(), y_hat_test.detach().numpy()] ,[model,history,kwargs]
        return [X_train, X_test, y_train, y_test, y_hat_train, y_hat_test], [model, history,kwargs]

    else:
        if return_numpy:
            return [X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy()],[model,history,kwargs]
        return [X_train, X_test, y_train, y_test], [model, history, kwargs]



if __name__ == '__main__':
    main('a',return_predictions=False)