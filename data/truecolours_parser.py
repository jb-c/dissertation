import sys
sys.path.append('../')
sys.path.append('../data')
from truecolours_loader import load_and_parse_by_disease_from_pickle
import numpy as np
import pandas as pd


def load_for_NANCY_UCEIS_series_to_series(path_to_pickle='UC_and_CD_df.p', update_pickle=False,min_allowable_samples=10):
    '''
    Loads and parses the truecolours dataset in X,y format

    :param root_path: The file path to the source folder of the true colours dataset
    :param lookback_window: The number of days to allow in a lookback window
    :param imputation_method: The method used to impute missing data
    :return: X, y for predition of the UCIS & NANCY labels
    '''

    # Load and perform basic cleaning on data
    UC_df, CD_df = load_and_parse_by_disease_from_pickle(path_to_pickle, update_pickle)
    X_CD, y_CD, X_UC, y_UC = [], [], [], []
    for df, label in zip([UC_df, CD_df], ['UC', 'CD']):

        # -----------------------------------------------------------------------------------
        #                        Getting Days with NANCY & UCEIS
        # -----------------------------------------------------------------------------------
        have_NANCY_and_UCEIS = (~df['UCEIS (0 to 8)'].isna()) & (~df['Nancy Index'].isna())
        patients_with_NANCY_and_UCEIS = df.loc[have_NANCY_and_UCEIS, 'participant_id'].drop_duplicates().reset_index(
            drop=True)
        qs = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'] if label == 'UC' else ['q1', 'q2', 'q3']
        # -----------------------------------------------------------------------------------
        #                                 Formatting as X,y
        # -----------------------------------------------------------------------------------
        for p_id in patients_with_NANCY_and_UCEIS:
            # Get data for this patient for UC and CD events resp
            x = df.loc[(df['participant_id'] == p_id)
            , ['time', 'days_since_last_response', 'summary', *qs]]

            # The y label we want to predict
            y = df.loc[(df['participant_id'] == p_id),
                       ['time', 'UCEIS (0 to 8)', 'Nancy Index']]

            idx_not_nans = ~x[qs].isna().all(axis=1)
            idy_not_nans = ~y[['UCEIS (0 to 8)', 'Nancy Index']].isna().all(axis=1)
            if (len(idx_not_nans) > min_allowable_samples) and (len(idy_not_nans) > 0):
                if label == 'UC':
                    X_UC.append(x)
                    y_UC.append(y)
                elif label == 'CD':
                    X_CD.append(x)
                    y_CD.append(y)

    return X_UC, y_UC, X_CD, y_CD




def load_for_NANCY_UCEIS_regression(path_to_pickle='UC_and_CD_df.p',update_pickle=False, lookback_window = 60, min_allowable_samples = 10, max_allowed_gap_to_y_from_last_x = 15):
    '''
    Loads and parses the truecolours dataset in X,y format

    :param root_path: The file path to the source folder of the true colours dataset
    :param lookback_window: The number of days to allow in a lookback window
    :return: X, y for predition of the UCIS & NANCY labels
    '''
    X_UC, y_UC, X_CD, y_CD = load_for_NANCY_UCEIS_series_to_series(path_to_pickle, update_pickle,min_allowable_samples)
    X_out,y_out = [],[]


    # -----------------------------------------------------------------------------------
    #                                 IMPORTANT NOTE
    # -----------------------------------------------------------------------------------
    # We have no data for Chrons disease patients, they are not included in this task

    for Xi,yi in zip(X_UC,y_UC):

        # -----------------------------------------------------------------------------------
        #                                 Formatting as X,y
        # -----------------------------------------------------------------------------------
        for i, (t,UCEIS_val,NANCY_val) in yi[(~yi.isna()).all(axis=1)].iterrows():

            # Get data for this patient in the lookback window
            x = Xi.loc[(Xi['time'] <= t) & (Xi['time'] >= t - lookback_window)]
            # Drop all nan response cols
            x = x[~x[['q1','q2','q3','q4','q5','q6']].isna().all(axis=1)]
            # The y label we want to predict
            y = np.array([UCEIS_val, NANCY_val])



            if len(x) > min_allowable_samples:
                time_of_last_response = x.iloc[-1]['time']
                time_delta_to_prediction = t - time_of_last_response

                if time_delta_to_prediction < max_allowed_gap_to_y_from_last_x:
                    # Reset time col
                    x['time'] = x['time'] - x['time'].iloc[0]

                    # Add to arrays
                    X_out.append(x)
                    y_out.append(y)

    return X_out,np.stack(y_out)



def load_for_forecasting(path_to_pickle='UC_and_CD_df.p',update_pickle=False , min_allowable_samples = 10):
    '''
    :param root_path: The file path to the source folder of the true colours dataset
    :param min_allowable_samples: The min number of data points we need to consider a patient
    :return:
    '''

    # Load and perform basic cleaning on data
    UC_df, CD_df = load_and_parse_by_disease_from_pickle(path_to_pickle,update_pickle)

    # Drop Nancy, UCEIS and Timestamp columns
    UC_df = UC_df.drop(axis=1,labels=['UCEIS (0 to 8)','Nancy Index','response_date'])
    CD_df = CD_df.drop(axis=1,labels=['UCEIS (0 to 8)','Nancy Index','response_date'])

    # Parse into desired format
    X_UC, X_CD = [], []

    # -----------------------------------------------------------------------------------
    #                                 Formatting as X
    # -----------------------------------------------------------------------------------
    for df, label in zip([UC_df, CD_df], ['UC', 'CD']):
        qs = ['q1','q2','q3','q4','q5','q6'] if label == 'UC' else ['q1','q2','q3']

        for pid, p_df in df.groupby('participant_id'):
            p_df = p_df.drop(axis=1,labels = ['participant_id']) # Need to do after group by
            p_df = p_df[(~p_df[qs].isna()).all(axis=1)] # Filter to rows with at least one question response
            p_df = p_df[['time','days_since_last_response','summary',*qs]] # Reorder cols

            if len(p_df) > min_allowable_samples:
                p_df['time'] = p_df['time'] - p_df['time'].iloc[0] # Reset time variable

                if label == 'UC':
                    X_UC.append(p_df)
                elif label == 'CD':
                    X_CD.append(p_df)

    return X_UC, X_CD