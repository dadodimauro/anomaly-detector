import yaml

def get_params(verbose=True):
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
        df_path = params['DF_PATH']
        
        if verbose is True:
            print_params(params)
        
        return PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, df_path
    
    else:
        params = params['UNIVARIATE']
        ALGORITHM = params['ALGORITHM']
        PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
        TRAINING_PARAMS = params['TRAINING_PARAMS']
        INTERVALS_PARAMS = params['INTERVALS_PARAMS']
        TH_ALGORITHM = params['TH_ALGORITHM']
        df_path = params['DF_PATH']
        
        if verbose is True:
            print_params(params)
    
        return ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, df_path
    
    
def print_params(params):
    for k, v in params.items():
        print(f'{k}: {v}')