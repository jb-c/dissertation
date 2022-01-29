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
    useful_columns = ['data', 'participant_id', 'modelName','response_date']
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

    # Explode the column of dictionaries into many columns
    df.data.fillna({},inplace=True) # Needed so next line doesn't error
    df = df.join(pd.json_normalize(df.data))
    df.drop(['data'],inplace=True,axis=1)

    df.sort_values(by=['participant_id','response_date'],inplace=True)
    df.reset_index(inplace=True,drop=True)

    return df




if __name__ == '__main__':
    df = load()
    print(df)