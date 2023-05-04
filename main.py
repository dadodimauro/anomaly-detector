# TO-DO: the update version is on the notebook, the .py version must be changed

from src.multivariate_ad import Multivariate_ad
from src.univariate_ad import Univariate_ad

from src.utils import params
from src.utils.parser import args

ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS = params.get_params()

if args.type == 'multivariate':
    if args.auto is True:
        u = Univariate_ad(
                ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, 
                INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
        )
        u.remove_anomalies()
        
    m = Multivariate_ad(
            PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, 
            TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
        )
    m.fit_predict()
        
else:  # univariate
    u = Univariate_ad(
            ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, 
            INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS
        )
    u.fit_predict()
