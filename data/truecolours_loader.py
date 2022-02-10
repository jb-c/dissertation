import json
#from data.truecolours_path import root, task
from truecolours_path import root, task
from tqdm import tqdm
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
#
#
# Each patient record can have cols in {'data', 'participant_id', 'modelName', 'isHidden', 'scheduleOpenedAt',
# 'response_date','scheduleId', 'scheduleClosedAt', 'scheduleEventIndex', 'isNoResponse', 'revision', '_id',
# 'messageIds', 'replaces', 'externalRef', 'isAdhoc'}
#-----------------------------------------------------------------------------------------------------------------------


def load(root_path = root):
    '''
    Loads the true_colours data from the root directory and parses into a single dataset
    :param root_path: The file path to the source folder of the true colours dataset
    :return: a dataframe contaning parsed data
    '''
    # --------- Load data from json ---------
    with open(root_path) as f:
        data = json.load(f)
    IDS = list(data.keys())

    # Most of the data is not that useful to us
    useful_columns = ['data', 'participant_id', 'modelName','response_date','scheduleOpenedAt','scheduleClosedAt']
    df = pd.DataFrame({})

    for ID in tqdm(IDS,desc='Processing Patient : '):
        p_df = pd.DataFrame(data[ID])
        p_df = p_df[[c for c in p_df.columns if c in useful_columns]] # Only keep useful cols

        # Floor data_date to the day and get rid of missing questionnaire data records
        if 'response_date' in p_df.columns:
            p_df['response_date'] = pd.to_datetime(p_df['response_date'],format="%Y-%m-%dT%H:%M:%S.%fZ")
            p_df.response_date = p_df.response_date.dt.floor('d')
        if 'isNoResponse' in p_df.columns:
            p_df = p_df[p_df.isNoResponse != True] # Ignore periods where there was no data

        df = pd.concat([df, p_df])

    print('Formatting the loaded dataframe')

    # Explode the column of dictionaries into many columns
    df.data.fillna({},inplace=True) # Needed so next line doesn't error
    df = df.join(pd.json_normalize(df.data))
    df.drop(['data'],inplace=True,axis=1)

    df.sort_values(by=['participant_id','response_date'],inplace=True)
    df.reset_index(inplace=True,drop=True)

    return df

def load_for_binary_classification(root_path = root, lookback_window = 60):
    '''
    Loads and parses the truecolours dataset for binary classification of disease activity

    :param root_path: The file path to the source folder of the true colours dataset
    :param lookback_window: The number of days to allow in a lookback window
    :return: X, y for binary classification of if disease is active or not
    '''
    df = load(root_path)

    print('Formatting for binary classification task')

    # Filter for only daily sccai results - for the moment
    df = df[df.modelName == 'gut_dsccai'].reset_index(drop=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Filter for only patients with UCEIS and Nancy results
    print('\t Filter to patients with UCEIS and NANCY test results')
    has_UCEIS_idx = np.argwhere((~df['UCEIS (0 to 8)'].isna()).values).flatten()
    has_Nancy_idx = np.argwhere((~df['Nancy Index'].isna()).values).flatten()
    patints_with_UCEIS_and_Nancy = np.intersect1d(df.iloc[has_UCEIS_idx]['participant_id'].values,df.iloc[has_Nancy_idx]['participant_id'].values)
    df = df[df['participant_id'].isin(patints_with_UCEIS_and_Nancy)]

    # Match Up the Dates on UCEIS and NANCY test results
    print('\t Aligning dates on UCEIS and NANCY test results')
    df = df.groupby('participant_id').apply(align_nancy_and_uceis_test_dates)

    print('Formatting into X,y format')

    # A df of patient_id,timestamp where we have ground truth events
    ground_truth_events = df.loc[~df['Nancy Index'].isna() | ~df['UCEIS (0 to 8)'].isna(), ['participant_id', 'response_date']].drop_duplicates()
    ground_truth_events['cutoff_date'] = ground_truth_events['response_date'] - pd.to_timedelta(f'{60} days')

    X = []
    y = []

    for (idx, row) in ground_truth_events.iterrows():
        # Filter main df to just the dates and patient in question
        patient_id, truth_date, start_date = row.values
        local_df = df.loc[
            (df['participant_id'] == patient_id) &
            (df['response_date'] >= start_date) &
            (df['response_date'] <= truth_date)
            ]

        # Compute y
        UCEIS = local_df['UCEIS (0 to 8)'].mean(skipna=True)
        NANCY = local_df['Nancy Index'].mean(skipna=True)
        local_y = (UCEIS >= 3.0) & (NANCY >= 2.0)  # Criterion for active disease

        # Select columns for X
        local_x = local_df[['summary', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']]

        X.append(local_x)
        y.append(local_y)

    return X,y






#
# --------------------------------------------------
# ---------------   Helper Functions ---------------
# --------------------------------------------------
#
def align_nancy_and_uceis_test_dates(pdf):
    '''
    Aligns the dates on Nancy and UCEIS timestamps
    :param pdf: A df for only one patient
    :return: An updated pdf with the Nancy and UCEIS test dates aligned to the same timestamp
    '''
    has_UCEIS_idx = np.argwhere((~pdf['UCEIS (0 to 8)'].isna()).values).flatten()
    has_Nancy_idx = np.argwhere((~pdf['Nancy Index'].isna()).values).flatten()

    if (len(has_UCEIS_idx) == 0) or (len(has_Nancy_idx) == 0):
        # Patient doesn't have any complete test results
        pass
    elif (len(has_UCEIS_idx) != len(has_Nancy_idx)):
        raise ValueError(f'Patient {pid} has an uneven number of UCEIS and NANCY test results')
    else:
        # Patient has an even number of both NANCY & UCEIS tests - let's match them up
        UCEIS_dates = pdf.iloc[has_UCEIS_idx]['response_date'].values
        UCEIS_sorted_by_date_argidx = UCEIS_dates.argsort()
        NANCY_dates = pdf.iloc[has_Nancy_idx]['response_date'].values
        NANCY_sorted_by_date_argidx = NANCY_dates.argsort()

        # Index of resp test results, sorted by time
        has_UCEIS_sorted_idx = has_UCEIS_idx[UCEIS_sorted_by_date_argidx]
        has_NANCY_sorted_idx = has_Nancy_idx[NANCY_sorted_by_date_argidx]

        # Take earliest date for pair of test results (as conduceted at the same time)
        test_date = np.minimum(UCEIS_dates[UCEIS_sorted_by_date_argidx], NANCY_dates[NANCY_sorted_by_date_argidx])

        # Update the test date for both tests (as cba finding which ones is actually the minimum)
        pdf.iloc[has_UCEIS_sorted_idx, 1] = test_date
        pdf.iloc[has_NANCY_sorted_idx, 1] = test_date

        return pdf



if __name__ == '__main__':
    df = load()
    print(df)