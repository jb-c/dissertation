from truecolours_path import root, task
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

def load():
    # Load raw json data
    with open(root) as f:
        data = json.load(f)
    IDS = list(data.keys())
    
    # Loop over each patient
    for ID in tqdm(IDS,desc='Processing Patient : '):
        df = pd.DataFrame(data[ID])
        df = df[df.isNoResponse != True] # Ignore periods where there was no data
        #df[['response_date','scheduleOpenedAt','scheduleClosedAt']] = pd.to_datetime(df[['response_date','scheduleOpenedAt','scheduleClosedAt']])


    
    return df