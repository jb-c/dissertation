import json, pickle
from truecolours_path import root, task
#from data.truecolours_path import root, task
from tqdm import tqdm
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
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
    # Read data from JSON
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
            p_df['response_date'] = pd.to_datetime(p_df['response_date'],format="%Y-%m-%dT%H:%M:%S.%fZ",errors='coerce',exact=False)
            p_df.response_date = p_df.response_date.dt.floor('d')
        if 'isNoResponse' in p_df.columns:
            p_df = p_df[p_df.isNoResponse != True] # Ignore periods where there was no data

        df = pd.concat([df, p_df])

    print('Formatting the loaded dataframe')

    # Explode the column of dictionaries into many columns
    df.data.fillna({},inplace=True) # Needed so next line doesn't error
    df = pd.concat([df.drop(['data'], axis=1), df['data'].apply(pd.Series)], axis=1)

    # Format test date column (as a floored datetime)
    df['test_date'] = pd.to_datetime(df['test_date'], format="%Y-%m-%dT%H:%M:%S", exact=False, errors='coerce')
    df['test_date'] = df['test_date'].dt.floor('d')

    # Override the response_date with the test date
    df['response_date'] = np.where(df['test_date'].isna(), df['response_date'].values, df['test_date'].values)

    # Sort by patient_id and date
    df.sort_values(by=['participant_id','response_date'],inplace=True)
    df.reset_index(inplace=True,drop=True)

    # Drop unneeded questionaiires
    model_names_to_keep = ['gut_dsccai', 'labresults','gut_hbi','eq5d']
    df = df[df['modelName'].isin(model_names_to_keep)]

    return df


def load_and_parse_by_disease(root_path = root):
    '''
    Loads and splits TC data by disease
    :param root_path:
    :return: UC_df, CD_df
    '''
    df = load(root_path)
    # -----------------------------------------------------------------------------------
    #                            Split data by disease
    # -----------------------------------------------------------------------------------
    UC_df = df.loc[
                    df['modelName'].isin(['gut_dsccai', 'labresults']),
                    ['participant_id', 'response_date', 'summary', 'q1', 'q2','q3', 'q4', 'q5', 'q6', 'UCEIS (0 to 8)', 'Nancy Index']
    ]

    CD_df = df.loc[
                    df['modelName'].isin(['gut_hbi', 'labresults']),
                    ['participant_id', 'response_date', 'summary', 'q1', 'q2','q3', 'UCEIS (0 to 8)', 'Nancy Index']
    ]
    # -----------------------------------------------------------------------------------
    #                            Type handling and aggregating
    # -----------------------------------------------------------------------------------
    # Cast numerical cols to floats
    CD_df = CD_df.astype({c: float for c in CD_df.columns if c not in ['participant_id', 'response_date']})
    UC_df = UC_df.astype({c: float for c in CD_df.columns if c not in ['participant_id', 'response_date']})

    # Fix the q6 col for the SCCAI
    UC_df['q4'] = UC_df['q4'].apply(UCEIS_Q6_PARSER)
    UC_df['q5'] = UC_df['q5'].apply(UCEIS_Q6_PARSER)
    UC_df['q6'] = UC_df['q6'].apply(UCEIS_Q6_PARSER)

    # Aggregate by day - ie one row per day / patient
    UC_df = UC_df.groupby(['participant_id', 'response_date']).agg(np.mean).reset_index()
    CD_df = CD_df.groupby(['participant_id', 'response_date']).agg(np.mean).reset_index()

    # Sort by patient_id and date
    UC_df.sort_values(by=['participant_id','response_date'],inplace=True)
    CD_df.sort_values(by=['participant_id','response_date'],inplace=True)

    # Drop all NAN rows
    UC_df = UC_df[~UC_df[[c for c in UC_df.columns if c not in ['participant_id','response_date']]].isna().all(axis=1)]
    CD_df = CD_df[~CD_df[[c for c in CD_df.columns if c not in ['participant_id','response_date']]].isna().all(axis=1)]

    # Add new cols
    for df, label in zip([UC_df, CD_df], ['UC', 'CD']):
        df['time'] = df.groupby('participant_id').apply(lambda x: (x['response_date'] - x['response_date'].iloc[0]).dt.days).reset_index(0, drop=True)
        df['days_since_last_response'] = df.groupby('participant_id').apply(lambda x:x['response_date'].diff().dt.days).reset_index(0, drop=True)
        df['days_since_last_response'].fillna(0,inplace=True) # Fix the initial nan entry


    return UC_df,CD_df


def load_and_parse_by_disease_from_pickle(path_to_pickle='UC_and_CD_df.p',update_pickle=False):
    '''
    Reads the parsed data from a pickle file, to save time
    :param path_to_pickle: The path to the pickle file
    :param update_pickle: Boolean to update pickle or not
    :return: UC_df,CD_df
    '''

    if update_pickle:
        UC_df, CD_df = load_and_parse_by_disease()
        pickle.dump([UC_df, CD_df], open(path_to_pickle, "wb"))
    else:
        [UC_df, CD_df] = pickle.load(open(path_to_pickle, "rb" ) )

    return UC_df,CD_df














#
# --------------------------------------------------
# ---------------   Helper Functions ---------------
# --------------------------------------------------
#
def impute(df,method):
    if method == 'ffill':
        return df.ffill(axis=0)


def UCEIS_Q6_PARSER(answer):
    if isinstance(answer, list):
        return float(answer[0])
    else:
        return float(answer)




if __name__ == '__main__':
    df = load()
    print(df)