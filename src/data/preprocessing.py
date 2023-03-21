import numpy as np
import pandas as pd

from sklearn import preprocessing

import torch
import torch.utils.data as data_utils

from src.data.columns import get_columns_list


# Transform all columns into float64
def transform_in_float64(df):
    for i in list(df): 
        df[i]=df[i].apply(lambda x: str(x).replace("," , "."))
    df = df.astype(float)
    return df


# downsampling
def downsampling(df, downsampling_rate):
    # df = df.groupby(np.arange(len(df.index)) // downsampling_rate).mean()
    df = df.groupby(pd.Grouper(freq=f'{downsampling_rate}min')).mean()
    df = df.dropna()
    return df


def downsample_labels(labels, downsampling_rate):
    labels_down=[]

    for i in range(len(labels)//downsampling_rate):
        if labels[5*i:5*(i+1)].count(1.0):
            labels_down.append(1.0) #Attack
        else:
            labels_down.append(0.0) #Normal

    #for the last few labels that are not within a full-length window
    if labels[downsampling_rate*(i+1):].count(1.0):
        labels_down.append(1.0) #Attack
    else:
        labels_down.append(0.0) #Normal
        
    return labels_down


def normalize_data(df, scaler='z-score'):
    if scaler == 'z-score':
        print("normalizing data using Z-Score")
        scaler = preprocessing.StandardScaler()
    elif scaler == 'min-max':
        print("normalizing data using MinMax Scaler")
        scaler = preprocessing.MinMaxScaler()
    elif scaler == 'all':
        scaler1 = preprocessing.StandardScaler()
        scaler2 = preprocessing.MinMaxScaler()
        
        x = df.values
        print("normalizing data using Z-Score")
        x_scaled = scaler1.fit_transform(x)
        print("normalizing data using MinMax Scaler")
        x_scaled = scaler2.fit_transform(x_scaled)
        df = pd.DataFrame(x_scaled)
        
        return df
    else:
        print('Error: wrong Scaler!')
        print('using default (Z-Score scaler)')
        print("normalizing data using Z-Score")
        scaler = preprocessing.StandardScaler()
        
    x = df.values
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    
    return df


def create_windows(df, window_size):
    windows_df = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0]-window_size)[:, None]]
    
    return windows_df


def get_df(db_path, columns_name=None):
    # retrieve columns used
    if columns_name is None:
        print('Warning: columns name not specified, using all metrics')
        df = pd.read_csv(db_path)
        # df = df.drop(['BEGIN_TIME'], axis = 1)
    else:
        columns = get_columns_list(columns_name)
        columns = ['BEGIN_TIME'] + columns
        df = pd.read_csv(db_path, usecols=columns)
    
    df['BEGIN_TIME'] = pd.to_datetime(df['BEGIN_TIME'])
    df = df.set_index('BEGIN_TIME') 
    
    print(f"dataframe shape: {df.shape}")
    return df


def get_db_time(df, PREPROCESSING_PARAMS, INTERVALS_PARAMS=None):
    
    if INTERVALS_PARAMS is None:
        df_len = len(df)
        train_start = 0
        train_end = int(np.floor(0.7 * df_len))  # 70 train, 30 test
        test_start = int(np.floor(0.3 * df_len))
        test_end = df_len
    else:
        train_start = INTERVALS_PARAMS['train_start']
        train_end = INTERVALS_PARAMS['train_end']
        test_start = INTERVALS_PARAMS['test_start']
        test_end = INTERVALS_PARAMS['test_end']
    
    df_train = df[train_start:train_end]  # .reset_index(drop=True)
    df_test = df[test_start:test_end]  # .reset_index(drop=True)
    
    downsampling_rate = PREPROCESSING_PARAMS['downsamplig_rate']

    if downsampling_rate > 0:
        df_train = downsampling(df_train, downsampling_rate)
        df_test = downsampling(df_test, downsampling_rate)
    df_train = transform_in_float64(df_train)
    df_test = transform_in_float64(df_test)
    
    db_time_train = df_train['Database Time Per Sec'].reset_index(drop=True)
    db_time_test = df_test['Database Time Per Sec'].reset_index(drop=True)
    
    return db_time_train, db_time_test


def data_preprocessing(PREPROCESSING_PARAMS, df, INTERVALS_PARAMS=None, scaler='z-score'):
    
    if INTERVALS_PARAMS is None:
        df_len = len(df)
        train_start = 0
        train_end = int(np.floor(0.7 * df_len))  # 70 train, 30 test
        test_start = int(np.floor(0.3 * df_len))
        test_end = df_len
    else:
        train_start = INTERVALS_PARAMS['train_start']
        train_end = INTERVALS_PARAMS['train_end']
        test_start = INTERVALS_PARAMS['test_start']
        test_end = INTERVALS_PARAMS['test_end']
    
    df_train = df[train_start:train_end]  # .reset_index(drop=True)
    df_test = df[test_start:test_end]  # .reset_index(drop=True)
    
    downsampling_rate = PREPROCESSING_PARAMS['downsamplig_rate']
    
    # df = df.drop(['BEGIN_TIME'], axis = 1)
    
    # train dataset 
    # df_train = transform_in_float64(df_train)
    if downsampling_rate > 0:
        df_train = downsampling(df_train, downsampling_rate)
    train_timestamps = df_train.index
    # df_train = df_train.drop(['BEGIN_TIME'], axis = 1).reset_index(drop=True)
    # df_train = transform_in_float64(df_train).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    if scaler is not None:
        df_train = normalize_data(df_train, scaler=scaler)
    
    # test dataset
    # df_test = transform_in_float64(df_test)
    if downsampling_rate > 0:
        df_test = downsampling(df_test, downsampling_rate)
    test_timestamps = df_test.index
    # df_test = df_test.drop(['BEGIN_TIME'], axis = 1).reset_index(drop=True)
    # df_test = transform_in_float64(df_test).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    if scaler is not None:                    
        df_test = normalize_data(df_test, scaler=scaler)
    
    # create windows
    window_size = PREPROCESSING_PARAMS['window_size']
    
    windows_train = create_windows(df_train, window_size)
    windows_test = create_windows(df_test, window_size)
    
    return df_train, df_test, windows_train, windows_test, train_timestamps, test_timestamps


def get_dataloaders(windows_train, windows_test, batch_size, w_size, z_size):
    # 80% train - 20% val
    windows_train = windows_train[:int(np.floor(.8 *  windows_train.shape[0]))]
    windows_val = windows_train[int(np.floor(.8 *  windows_train.shape[0])):int(np.floor(windows_train.shape[0]))]
    
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_train).float().view(([windows_train.shape[0],w_size]))
    ) , batch_size=batch_size, shuffle=False, num_workers=0)
    
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_val).float().view(([windows_val.shape[0],w_size]))
    ) , batch_size=batch_size, shuffle=False, num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_test).float().view(([windows_test.shape[0],w_size]))
    ) , batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader