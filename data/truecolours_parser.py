import sys
sys.path.append('../')
sys.path.append('../data')
from truecolours_loader import load_and_parse_by_disease_from_pickle
import numpy as np
import pandas as pd






def load_for_NANCY_UCEIS_regression(path_to_pickle='UC_and_CD_df.p',update_pickle=False, lookback_window = 60, min_allowable_samples = 10):
    '''
    Loads and parses the truecolours dataset in X,y format

    :param root_path: The file path to the source folder of the true colours dataset
    :param lookback_window: The number of days to allow in a lookback window
    :param imputation_method: The method used to impute missing data
    :return: X, y for predition of the UCIS & NANCY labels
    '''

    # Load and perform basic cleaning on data
    UC_df, CD_df = load_and_parse_by_disease_from_pickle(path_to_pickle, update_pickle)
    X_CD, y_CD, X_UC, y_UC = [],[],[],[]
    for df,label in zip([UC_df, CD_df],['UC','CD']):

        # -----------------------------------------------------------------------------------
        #                        Getting Days with NANCY & UCEIS
        # -----------------------------------------------------------------------------------
        have_NANCY_and_UCEIS = (~df['UCEIS (0 to 8)'].isna()) & (~df['Nancy Index'].isna())
        truth_events = df.loc[have_NANCY_and_UCEIS, ['participant_id', 'response_date','UCEIS (0 to 8)','Nancy Index']].drop_duplicates().reset_index(drop=True)
        qs = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'] if label == 'UC' else ['q1', 'q2', 'q3']
        # -----------------------------------------------------------------------------------
        #                                 Formatting as X,y
        # -----------------------------------------------------------------------------------
        for i, event in truth_events.iterrows():
            p_id, date, UCEIS_val, NANCY_val = event

            # Get data for this patient for UC and CD events resp
            x = df.loc[(df['participant_id'] == p_id) &
                       (df['response_date'] < date) &
                       (df['response_date'] >= date - pd.Timedelta(lookback_window, unit='days'))
                       ,['time', 'days_since_last_response', 'summary', *qs]]

            # Drop all nan response cols
            x = x[~x[qs].isna().all(axis=1)]
            # The y label we want to predict
            y = np.array([UCEIS_val, NANCY_val])

            if len(x) > min_allowable_samples:
                # Reset time col
                x['time'] = x['time'] - x['time'].iloc[0]
                if label == 'UC':
                    X_UC.append(x)
                    y_UC.append(y)
                elif label == 'CD':
                    X_CD.append(x)
                    y_CD.append(y)

        # -----------------------------------------------------------------------------------
        #                                 IMPORTANT NOTE
        # -----------------------------------------------------------------------------------
        # We have no data for Chrons disease patients, they are not included in this task

    return X_UC,np.stack(y_UC)















def load_for_forecasting(path_to_pickle='UC_and_CD_df.p',update_pickle=False , N = 100, min_allowable_samples = 10):
    '''
    :param root_path: The file path to the source folder of the true colours dataset
    :param N: The number of days to allow in a lookback window, if N = 0 then returns all data for a patient
    :param min_allowable_samples: The min number of data points we need to consider a patient
    :return:
    '''

    # Load and perform basic cleaning on data
    UC_df, CD_df = load_and_parse_by_disease_from_pickle(path_to_pickle,update_pickle)

    # Drop Nancy, UCEIS and Timestamp columns
    UC_df = UC_df.drop(axis=1,labels=['UCEIS (0 to 8)','Nancy Index','response_date'])
    CD_df = CD_df.drop(axis=1, labels=['UCEIS (0 to 8)', 'Nancy Index','response_date'])

    # Filter to only patients with enough samples
    patient_has_enough_samples_CD = (CD_df.groupby('participant_id').size() > max(N,min_allowable_samples)).reset_index()
    patient_has_enough_samples_UC = (UC_df.groupby('participant_id').size() > max(N,min_allowable_samples)).reset_index()
    patients_with_enough_samples_CD = patient_has_enough_samples_CD.loc[patient_has_enough_samples_CD[0], 'participant_id'].values
    patients_with_enough_samples_UC = patient_has_enough_samples_UC.loc[patient_has_enough_samples_UC[0], 'participant_id'].values
    UC_df = UC_df[UC_df['participant_id'].isin(patients_with_enough_samples_UC)]
    CD_df = CD_df[CD_df['participant_id'].isin(patients_with_enough_samples_CD)]

    # Parse into desired format
    X_UC, X_CD = [], []

    for df, label in zip([UC_df, CD_df], ['UC', 'CD']):
        for pid, p_df in df.groupby('participant_id'):
            p_df = p_df.drop(axis=1,labels = ['participant_id']) # Need to do after group by

            if N > 0:
                m = int(np.floor(len(p_df) / N))
                p_df_split = np.array_split(p_df.iloc[:N * m], m)
            else:
                p_df_split = [p_df]

            for pp_df in p_df_split:
                pp_df['time'] = pp_df['time'] - pp_df['time'].iloc[0] # Reset time variable

            if label == 'UC':
                X_UC = X_UC + p_df_split
            elif label == 'CD':
                X_CD = X_CD + p_df_split

    if N > 0:
        return np.stack(X_UC), np.stack(X_CD)
    else:
        return X_UC, X_CD