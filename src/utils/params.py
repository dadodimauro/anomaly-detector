import yaml

def get_params():
    with open("src/params.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    PREPROCESSING_PARAMS = params['PREPROCESSING_PARAMS']
    TRAINING_PARAMS = params['TRAINING_PARAMS']
    INTERVALS_PARAMS = params['INTERVALS_PARAMS']
    TH_ALGORITHM = params['TH_ALGORITHM']
    df_path = params['DF_PATH']
    
    return PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, df_path