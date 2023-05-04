import importlib
import numpy as np


# https://github.com/KulikDM/pythresh
def get_th_and_labels(th_algorithm, y_pred, max_contamination=0.1):
    """
    Get the threshold and labels from a prediction score using pythresh

    Parameters
    ----------
    th_algorithm : str or :obj:`list` of :obj:`str`
        name of the threshold algorithm
    y_pred : np.Array
        prediction score
    max_contamination : int
        max contamination use for the COMB algorithm

    Returns
    -------
    (float, np.array)
        threshold and labels
    """

    th_algorithms = []
    if type(th_algorithm) is list:  # COMB
        for element in th_algorithm:
            module = importlib.import_module('.' + th_algorithm, package='pythresh.thresholds')
            th_algorithm = getattr(module, th_algorithm.upper())
            th_algorithms.append(th_algorithm)

        module = importlib.import_module('.comb', package='pythresh.thresholds')
        th_algorithm = getattr(module, 'COMB')

        thres = th_algorithm(th_algorithms)
    elif th_algorithm == 'all':  # all algorithms            
        module = importlib.import_module('.' + th_algorithm, package='pythresh.thresholds')
        th_algorithm = getattr(module, th_algorithm.upper())

        thres = th_algorithm(max_contam=max_contamination)
    #     elif th_algorithm == 'meta':
    #         module = importlib.import_module('.'+th_algorithm, package='pythresh.thresholds')
    #         th_algorithm = getattr(module, th_algorithm.upper())

    #         thres = th_algorithm(method='GNBC')
    else:  # single algorithm
        module = importlib.import_module('.' + th_algorithm, package='pythresh.thresholds')
        th_algorithm = getattr(module, th_algorithm.upper())

        thres = th_algorithm()

    labels = thres.eval(y_pred)
    th = thres.thresh_

    return th, labels


def get_labels(th_algorithm, y_pred, max_contamination=0.1):
    """
    Get the labels from a prediction score using pythresh

    Parameters
    ----------
    th_algorithm : str or :obj:`list` of :obj:`str`
        name of the threshold algorithm
    y_pred : np.Array
        prediction score
    max_contamination : int
        max contamination use for the COMB algorithm

    Returns
    -------
    np.array
        labels
    """

    th, labels = get_th_and_labels(th_algorithm, y_pred, max_contamination=max_contamination)

    return labels


def MinMaxNorm(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm


def get_labels_from_th(th, y_pred):
    labels = np.where(MinMaxNorm(y_pred) > th, 1, 0)
    return labels


def adjust_labels(labels, window_size):
    l = labels.copy()
    for i, label in enumerate(l):
        if label == 1:
            if i-window_size < 0:
                labels[0:i] = 1
            else:
                labels[i-window_size:i] = 1
    
    return labels