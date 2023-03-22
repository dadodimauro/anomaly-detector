# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import torch

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

    
def generate_anomalies_intervals(labels, timestamps):
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
    
    full_path = path + filename
    
    if save_csv is True:
        df.to_csv(full_path+'.csv', index=False)
    
    if save_txt is True:
        with open(full_path+'.txt', 'w') as f:
            f.write(df.to_string(header=True, index=False))
    
