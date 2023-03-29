import os

try:
    from src.utils import utils
    from src.utils import params
    from src.utils import thresholds as th
    from src.data import columns
    from src.data import preprocessing  
    from src.visualization import plotter
    
    from pyod.models.lof import LOF
    from pyod.models.iforest import IForest
    from pyod.models.ecod import ECOD
    from pyod.models.knn import KNN
    
except ModuleNotFoundError:
    print("installing requirements..")
    os.system('pip install -r requirements.txt')
    from src.utils import utils
    from src.utils import params
    from src.utils import thresholds as th
    from src.data import columns
    from src.data import preprocessing
    from src.visualization import plotter
    
    from pyod.models.cblof import CBLOF
    from pyod.models.iforest import IForest
    from pyod.models.ecod import ECOD
    from pyod.models.knn import KNN
    
    
class Univariate_ad():
    def __init__(self, ALGORITHM, PREPROCESSING_PARAMS, TRAINING_PARAMS, 
                     INTERVALS_PARAMS, TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS):
        self.ALGORITHM = ALGORITHM
        self.PREPROCESSING_PARAMS = PREPROCESSING_PARAMS
        self.TRAINING_PARAMS = TRAINING_PARAMS
        self.INTERVALS_PARAMS = INTERVALS_PARAMS
        self.TH_ALGORITHM = TH_ALGORITHM
        self.PLOT_PARAMS = PLOT_PARAMS
        self.df_path = PATH_PARAMS['DF_PATH']
        self.ALGORITHMS_LIST = ['lof', 'iforest', 'ecod', 'knn']
        self.contamination = TRAINING_PARAMS['contamination']
        self.scaler = PREPROCESSING_PARAMS['normalization']
        self.metric = PREPROCESSING_PARAMS['metric']
        if len(self.metric) != 1:
            print('ERROR: use only one metric in the univariate case')
        
        self.df = None
        self.labels_ = None
        self.decision_scores_ = None
        self.threshold_ = None
        self.anomalies_intervals_ = None
        
    def fit_predict(self, plot=True):
        self.df = preprocessing.get_df(self.df_path, columns_name=self.metric)
        db_time = preprocessing.get_db_time(self.df, self.PREPROCESSING_PARAMS, INTERVALS_PARAMS=None, multi=False)
        df, timestamps = preprocessing.data_preprocessing(
                                                self.PREPROCESSING_PARAMS, self.df, 
                                                INTERVALS_PARAMS=None, 
                                                scaler=self.scaler,
                                                multi=False
                                        )
        
        if self.ALGORITHM not in self.ALGORITHMS_LIST:
            print('Error: specified algorithm not supported')
            print('using default algorithm (KNN)')
        
        if self.ALGORITHM == 'lof':
            clf = LOF(contamination=self.TRAINING_PARAMS['contamination'])
        elif self.ALGORITHM == 'iforest':
            clf = IForest(contamination=self.TRAINING_PARAMS['contamination'])
        elif self.ALGORITHM == 'ecod':
            clf = ECOD(contamination=self.TRAINING_PARAMS['contamination'])
        else:
            clf = KNN(contamination=self.TRAINING_PARAMS['contamination'])
            
        clf.fit(df, y=None)
        y_pred = clf.decision_scores_
        model_thresh = clf.threshold_
        model_labels = clf.labels_
        self.decision_scores_ = y_pred
        if self.TH_ALGORITHM is None:
            self.threshold_, self.labels_ = model_thresh, model_labels
        else:
            self.threshold_, self.labels_ = th.get_th_and_labels(self.TH_ALGORITHM, y_pred)
        
        print(f'detected {sum(self.labels_)} anomalies')
        
        anomalies_intervals_df = utils.generate_anomalies_intervals(self.labels_, timestamps)
        self.anomalies_intervals_ = anomalies_intervals_df
        utils.save_anomalies_intervals(anomalies_intervals_df, filename='anomalies-intervals-univariate')
        
        if plot is True:
            plotter.plot_res_db_time(y_pred, db_time, timestamps=timestamps, 
                             save_static=self.PLOT_PARAMS['db_time_static'], save_html=self.PLOT_PARAMS['db_time_html']) 
            plotter.plot_labels(y_pred, self.labels_, timestamps=timestamps,
                                    save_static=self.PLOT_PARAMS['labels_static'], save_html=self.PLOT_PARAMS['labels_html'])
            
    def remove_anomalies(self, path='./data/processed/filtered/', clean_name='clean_df.csv', dirty_name='dirty_df.csv'):
        if self.anomalies_intervals_ is None:
            self.fit_predict()
        
        df = preprocessing.get_df(self.df_path, columns_name=None)
        clean_df, dirty_df = utils.remove_week_with_anomalies(df, self.anomalies_intervals_)
        preprocessing.save_df(clean_df, path=path, name=clean_name)
        preprocessing.save_df(dirty_df, path=path, name=dirty_name)