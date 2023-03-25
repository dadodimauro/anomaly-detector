import yaml

# TODO: retreive the parameeter 'MULTI' from commad line and/or parameter \
#       instead of from YAML file
# TODO: modify the YAML adding the field 'ALGORITHM' also for the multivariate case
def get_params(verbose=True):
    """
    Function used to retreive and store in dictionaries all the experiment parameters
    saved in the configuration YAML file

    Parameters
    ----------
    verbose : bool
        print the parameter or no

    Returns
    -------
    (dict, ..., str)
        - ALGORITHM: the algorith used in the univariate case or for dividing train and test set
        - PREPROCESSING_PARAMS: preprocessing parameters
        - TRAINING_PARAMS: training parameters
        - INTERVALS_PARAMS: intervals parameters
        - TH_ALGORITHM: threshold algorithm
        - PLOT_PARAMS: plotting parameters
        - df_path: path of the input data
    """

    with open("src/params.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    if params['MULTI']:  # multivatiate
        params = params['MULTIVARIATE']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        PLOT_PARAMS = params['PLOT_PARAMS']
        df_path = params['DF_PATH']
        
        if verbose is True:
            print_params(params)
        
        return PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, df_path
    
    else:
        params = params['UNIVARIATE']
        ALGORITHM = params['ALGORITHM']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        PLOT_PARAMS = params['PLOT_PARAMS']
        df_path = params['DF_PATH']
        
        if verbose is True:
            print_params(params)
    
        return ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, df_path
    
    
def print_params(params):
    """
    print the parameters

    Parameters
    ----------
    params : dict
        dictionary of parameters to be printed
    """

    for k, v in params.items():
        print(f'{k}: {v}')