import yaml

show_plots = False

# TODO: retreive the parameeter 'MULTI' from commad line and/or parameter \
#       instead of from YAML file
# TODO: modify the YAML adding the field 'ALGORITHM' also for the multivariate case
# TODO: update Docstring
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

    from src.utils.parser import args
    
    params_file_path = args.params_path
    with open(params_file_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    if args.show_plots is True:
        global show_plots
        show_plots = True
    
    if args.type == 'multivariate':  # multivatiate
        params = params['MULTIVARIATE']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        
        if args.auto is True:
            PREPROCESSING_PARAMS['single_file'] = False  # if automatic train data selection is true use always 2 files
        else:
            PREPROCESSING_PARAMS['single_file'] = args.single_file
            
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        PLOT_PARAMS = params['PLOT_PARAMS']
        PATH_PARAMS = params['PATH_PARAMS']
        
        if verbose is True:
            print_params(params)
        
        return None, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
    
    elif args.type == 'univariate':
        params = params['UNIVARIATE']
        ALGORITHM = params['ALGORITHM']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        PLOT_PARAMS = params['PLOT_PARAMS']
        PATH_PARAMS = params['PATH_PARAMS']
        
        if verbose is True:
            print_params(params)
    
        return ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
    
    else:
        print('Error: inserted wrong value for parameter --type')
        print(f"\tinserted parameter: <{args.type}>, choose between ['univariate', 'multivariate']")
    

def get_params_jupyter(verbose=True, MULTI=None):
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
            
    global show_plots
    show_plots = True
           
    if MULTI is None:
        MULTI = params['MULTI']
    
    if MULTI is True:  # multivatiate
        params = params['MULTIVARIATE']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        PLOT_PARAMS = params['PLOT_PARAMS']
        PATH_PARAMS = params['PATH_PARAMS']
        
        if verbose is True:
            print_params(params)
        
        return PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
    
    else:
        params = params['UNIVARIATE']
        ALGORITHM = params['ALGORITHM']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        PLOT_PARAMS = params['PLOT_PARAMS']
        PATH_PARAMS = params['PATH_PARAMS']
        
        if verbose is True:
            print_params(params)
    
        return ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
    
    
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