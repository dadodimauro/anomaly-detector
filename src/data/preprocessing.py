import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as data_utils
from sklearn import preprocessing

from src.data.columns import get_columns_list


# Transform all columns into float64
def transform_in_float64(df):
    """
    Transform all DataFrame values into float64

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame

    Returns
    -------
    pandas.DataFrame
        the input DataFrame with all its values converted to float64
    """

    for i in list(df):
        df[i] = df[i].apply(lambda x: str(x).replace(",", "."))
    df = df.astype(float)
    return df


# downsampling
# TODO add option for df with non DateTime Index
def downsampling(df, downsampling_rate):
    """
    Function that aggregate the data (down-sampling) of the DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame, INDEX MUST BE A DATETIME INDEX
    downsampling_rate : int
        number of minutes of the aggregation (i.e. if 5, aggregate data over a 5 minutes span)

    Returns
    -------
    pandas.DataFrame
        the DataFrame with the down-sampled data

    """
    # df = df.groupby(np.arange(len(df.index)) // downsampling_rate).mean()
    df = df.groupby(pd.Grouper(freq=f'{downsampling_rate}min')).mean()
    df = df.dropna()
    return df


def downsample_labels(labels, downsampling_rate):
    """
    apply the downsample to a set of labels, useful when training in a SUPERVISED scenario

    Parameters
    ----------
    labels : np.array
    downsampling_rate : int

    Returns
    -------
    np.array
    """

    labels_down = []

    for i in range(len(labels) // downsampling_rate):
        if labels[5 * i:5 * (i + 1)].count(1.0):
            labels_down.append(1.0)  # Attack
        else:
            labels_down.append(0.0)  # Normal

    # for the last few labels that are not within a full-length window
    if labels[downsampling_rate * (i + 1):].count(1.0):
        labels_down.append(1.0)  # Attack
    else:
        labels_down.append(0.0)  # Normal

    return labels_down


def normalize_data(df, scaler='z-score'):
    """
    A function for applying transformation to the input DataFrame

    Supports z-score standardization, min-max normalization and the two combined

    Parameters
    ----------
    df : pandas.DataFrame
    scaler : str
        possible values: 'z-score', 'min-max', 'all'

    Returns
    -------
    df : pandas.DataFrame
        the normalized DataFrame
    """

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
    """
    A function that create windows of size window_size
    From shape(len, #features) -> shape(len-window_size, window_size, #features)

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame of shape (len, #features)
    window_size : int
        the size of the window

    Returns
    -------
    pandas.DataFrame
        a pandas DataFrame of shape (len-window_size, window_size, #features)
    """

    windows_df = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]

    return windows_df


def get_df(path, columns_name=None):
    """
    Function to retrieve data from a csv file

    Parameters
    ----------
    path : str
        path of the csv file
    columns_name : :obj:`list` of :obj:`str`
        the metrics to retrieve
    Returns
    -------
    pandas.DataFrame
        a DataFrame containing data of the specified metrics
    """

    # retrieve columns used
    if columns_name is None:
        print('Warning: columns name not specified, using all metrics')
        df = pd.read_csv(path)
        # df = df.drop(['BEGIN_TIME'], axis = 1)
    else:
        columns = get_columns_list(columns_name)
        columns = ['BEGIN_TIME'] + columns
        df = pd.read_csv(path, usecols=columns)

    df['BEGIN_TIME'] = pd.to_datetime(df['BEGIN_TIME'])
    df = df.set_index('BEGIN_TIME')

    print(f"dataframe shape: {df.shape}")
    return df

# TODO: change it to return always the same number of values (maybe set None) to make it safer
def get_db_time(df, PREPROCESSING_PARAMS, INTERVALS_PARAMS=None, multi=True):
    """
    Function to retreive the 'Database Time Per Sec' from an input DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
    PREPROCESSING_PARAMS : dict
        a dictionary containing the preprocessing parameters
    INTERVALS_PARAMS : dict
        a dictionary containing the intervals parameters
    multi : bool
        True if multivariate case, False if univatiate case
        ATTENTION:
            - 2 df are returned if set to True (train and test)
            - 1 df is returned if set to False
    Returns
    -------
    pandas.DataFrame
        if multi is True:
            2 DataFrame containing the 'Database Time Per Sec' for train and test
        if multi is False:
            Only one DataFrame is Returned
    """

    if multi is True:
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

    else:  # if univariate no division for test and split since is always unsupervised
        if INTERVALS_PARAMS is None:
            start = 0
            end = len(df)
        else:
            start = INTERVALS_PARAMS['start']
            end = INTERVALS_PARAMS['end']

        df = df[start:end]
        downsampling_rate = PREPROCESSING_PARAMS['downsamplig_rate']

        if downsampling_rate > 0:
            df = downsampling(df, downsampling_rate)

        df = transform_in_float64(df)

        db_time = df['Database Time Per Sec'].reset_index(drop=True)

        return db_time

# TODO: add function to retreive train and test from 'filtered' folder instead of from INTERVALS_PARAMS
# TODO: change it to return always the same number of values (maybe set None) to make it safer
def data_preprocessing(PREPROCESSING_PARAMS, df, INTERVALS_PARAMS=None, scaler='z-score', multi=True):
    """
    function that handles the data preprocessing

    Parameters
    ----------
    PREPROCESSING_PARAMS : dict
        a dictionary containing the preprocessing parameters
    df : DataFrame
    INTERVALS_PARAMS : dict
        a dictionary containing the intervals parameters
    scaler : str
        possible values: 'z-score', 'min-max', 'all', 'None'
    multi : bool
        True if multivariate case, False if univatiate case
        ATTENTION:
            - 4 df and 2 DatetimeIndex are returned if set to True (train and test)
            - 1 df and 1 DatetimeIndex are returned if set to False
    Returns
    -------
    (pandas.DataFrame, ..., pandas.DatetimeIndex)
        if multi is True:
            - 2 DataFrames containing Train and Test data
            - 2 DataFrames containing the "windowed" Train and Test data
            - 2 DatetimeIndex for the Train and Test data
        if multi is False:
            - 1 DataFrame containing the data
            - 1 DatetimeIndex for the data
    """

    if multi is True:
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

    else:
        if INTERVALS_PARAMS is None:
            start = 0
            end = len(df)
        else:
            start = INTERVALS_PARAMS['start']
            end = INTERVALS_PARAMS['end']

        df = df[start:end]  # .reset_index(drop=True)

        downsampling_rate = PREPROCESSING_PARAMS['downsamplig_rate']

        if downsampling_rate > 0:
            df = downsampling(df, downsampling_rate)
        timestamps = df.index
        df = df.reset_index(drop=True)
        if scaler is not None:
            df = normalize_data(df, scaler=scaler)

        return df, timestamps


# TODO: check if z_size is needed, if not remove it
def get_dataloaders(windows_train, windows_test, batch_size, w_size, z_size):
    """
    Function to get the pytorch dataloaders for training and testing the model
    (used for the USAD model)

    Parameters
    ----------
    windows_train : pandas.Dataframe
        train data (windowed)
    windows_test : pandas.Dataframe
        test data (windowed)
    batch_size : int
    w_size : int
        parameter of the USAD model

    Returns
    -------
    (torch.DataLoader, torch.DataLoader, torch.DataLoader)
        DataLoaders for train, test and validation
    """

    # 80% train - 20% val
    windows_train = windows_train[:int(np.floor(.8 * windows_train.shape[0]))]
    windows_val = windows_train[int(np.floor(.8 * windows_train.shape[0])):int(np.floor(windows_train.shape[0]))]

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_train).float().view(([windows_train.shape[0], w_size]))
    ), batch_size=batch_size, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_val).float().view(([windows_val.shape[0], w_size]))
    ), batch_size=batch_size, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_test).float().view(([windows_test.shape[0], w_size]))
    ), batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def save_df(df, path, name):
    """
    Function to save a DataFrame in a .csv file

    Parameters
    ----------
    df : pandas.DataFrame
    path : str
        path of the csv file
    name : str
        name of the file
    Returns
    -------
    pandas.DataFrame
        a DataFrame containing data of the specified metrics
    """
    
    if not os.path.exists(path):
        os.makedirs(path)
    if '.csv' in name:
        full_path = path + name
    else:
        full_path = path + name + '.csv'
        
    df.to_csv(full_path, index=False)