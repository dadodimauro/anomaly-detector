import numpy as np
import pandas as pd
from datetime import datetime

import torch

def get_default_device():
    """
    Pick GPU if available, else CPU

    Returns
    -------
    torch.device
        torch device
    """

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    """
    Move tensor(s) to chosen device

    Parameters
    ----------
    data : torch.tensor
    device : torch.device

    Returns
    -------
    torch.tensor
        input tensor moved to the chosen device
    """

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

    
def generate_anomalies_intervals(labels, timestamps):
    """
    Generate a DataFrame containing the anomalies intervals

    Parameters
    ----------
    labels : np.array
    timestamps : pandas.DatetimeIndex

    Returns
    -------
    pandas.Dataframe
        DataFrame containing the anomalies intervals
    """

    offset = len(timestamps) - len(labels)
    timestamps = timestamps[offset:]  # the offset is the len of the window

    df = pd.DataFrame(columns=['start', 'end', 'lenght'])

    flag = False
    start, end = None, None
    for i, label in enumerate(labels):
        if label == 1:  # start of the anomaly window
            if flag is False:
                start = timestamps[i]
                flag = True
        else:
            if flag is True: # end of the anomaly window
                end = timestamps[i]
                flag = False
                # print(start, '-', end)
                df.loc[len(df)] = [start, end, end-start]
                start = None
                end = None

    if start is not None and end is None:  # handle case where last anomaly windows doesn't end
        end = timestamps[-1]
        flag = False
        print("Warning: last anomaly windows doesn't end")
        # print(start, '-', end)
        df.loc[len(df)] = [start, end, end-start]
        start = None
        end = None
        
    print(df.to_string(header=True, index=False))
    
    return df


def save_anomalies_intervals(df, save_txt=True, save_csv=True,
                                                path='./reports/anomalies-intervals/', filename='anomalies-intervals'):
    """
    Save a DataFrame containing the anomalies intervals to a file (.txt and/or .csv)

    Parameters
    ----------
    df : pandas.DataFrame
    save_txt : bool
    save_csv : bool
    path : str
        path where to store the intervals
    filename : str
        name of the file
    """

    full_path = path + filename
    
    if save_csv is True:
        df.to_csv(full_path+'.csv', index=False)
    
    if save_txt is True:
        with open(full_path+'.txt', 'w') as f:
            f.write(df.to_string(header=True, index=False))
    

def remove_week_with_anomalies(df, anomalies_intervals_df):
    """
    Remove the weeks containing anomalies from a DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
    anomalies_intervals_df : pandas.DataFrame
        DataFrame containing the anomalies intervals

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        returns 2 DataFrame, one with the 'clean' and one with the 'dirty' data
    """

    anomaly_weeks = set(anomalies_intervals_df['start'].dt.isocalendar().week).union(
                                                            set(anomalies_intervals_df['end'].dt.isocalendar().week))
    
    clean_df = df[~df.index.isocalendar().week.isin(anomaly_weeks)]
    dirty_df = df[df.index.isocalendar().week.isin(anomaly_weeks)]
    return clean_df, dirty_df

