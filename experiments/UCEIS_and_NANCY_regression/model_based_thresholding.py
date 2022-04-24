import numpy as np
from load_saved_model_and_data import main as load_model_and_predictions


def main(model_name,at_epoch=None,**kwargs):
    ## Load model and compute predictions
    output = load_model_and_predictions(model_name,at_epoch=at_epoch,sde=False,**kwargs)
    [X_train, X_test, y_train, y_test, y_hat_train, y_hat_test], [X_train_raw, X_test_raw, y_train_raw, y_test_raw] ,[model,history,kwargs] = output

    # If we have trained with t-pinball loss, disregard the upper and lower bounds
    if ('t_pinball' in kwargs) and (kwargs['t_pinball'] is not None):
        threshold_y_hat_test = np.zeros(y_hat_test.shape[:-2]  + y_hat_test.shape[-1:])
        threshold_y_hat_train = np.zeros(y_hat_train.shape[:-2] + y_hat_train.shape[-1:])
    else:
        threshold_y_hat_test = np.zeros(y_hat_test.shape[:-1])
        threshold_y_hat_train = np.zeros(y_hat_train.shape[:-1])
    ##
    ## Predicted Thresholds Based On Predicted NANCY & UCEIS
    ## y = [batch,time, [UCEIS,NANCY]]


    for threshold_var,y_hat in zip([threshold_y_hat_train,threshold_y_hat_test],[y_hat_train,y_hat_test]):

        if ('t_pinball' in kwargs) and (kwargs['t_pinball'] is not None):
            a  = (np.tile(np.array([8, 4]),reps=3).reshape(2,3))
            t1 = (np.tile(np.array([1, 1]),reps=3).reshape(2,3))
            t2 = (np.tile(np.array([4, 2]), reps=3).reshape(2,3))
        else:
            a  = np.array([8, 4])
            t1 = np.array([1, 1])
            t2 = np.array([4, 2])

        in_remission   = np.all((y_hat * a) <= t1, axis=2)
        disease_active = np.all((y_hat * a) >= t2, axis=2)

        # Label = 0 (remission), 2 (active) or 1 (neither)
        label = np.where(in_remission,0,1)
        threshold_var[::] = np.where(disease_active,2,label)


    ##
    ## Add predicted thresholds to the dataframes as an extra col
    ##

    for X_raw,threshold_var in zip([X_train_raw,X_test_raw],[threshold_y_hat_train,threshold_y_hat_test]):
        labels = ['_low','','_high'] if ('t_pinball' in kwargs) and (kwargs['t_pinball'] is not None) else ['']


        for i, Xi in enumerate(X_raw):
            for j,l in enumerate(labels):
                if len(Xi) <= threshold_var.shape[-2]:
                    Xi[f'predicted_band{l}'] = threshold_var[i, :len(Xi)][:,j]
                else:

                    Xi[f'predicted_band{l}'] = np.insert(threshold_var[i], 0,np.nan,axis=0)[:,j]



    ##
    ## Add thresholds based on summary
    ##
    for X_raw in [X_train_raw,X_test_raw]:
        for i, Xi in enumerate(X_raw):
            if 'summary' not in Xi.columns:
                Xi['summary'] = Xi[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].sum(axis=1).values

            Xi['band'] = np.where(Xi['summary'] >= 6,2,1)
            Xi['band'] = np.where(Xi['summary'] <= 2, 0, Xi['band'])

    return [X_train_raw, X_test_raw], kwargs, [y_train, y_test]


if __name__ == '__main__':
    main('2022_04_01_21:21')