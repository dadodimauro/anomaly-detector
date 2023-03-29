import os

try:
    from src.utils import utils
    from src.utils import params
    from src.utils import thresholds as th
    from src.models import usad
    from src.models import usad_utils
    from src.data import columns
    from src.data import preprocessing
    from src.visualization import plotter
except ModuleNotFoundError:
    print("installing requirements..")
    os.system('pip install -r requirements.txt')
    from src.utils import utils
    from src.utils import params
    from src.utils import thresholds as th
    from src.models import usad
    from src.models import usad_utils
    from src.data import columns
    from src.data import preprocessing
    from src.visualization import plotter
    
    
# TODO: add Docstrings
class Multivariate_ad():
    def __init__(self, PREPROCESSING_PARAMS, TRAINING_PARAMS, INTERVALS_PARAMS, 
                         TH_ALGORITHM, PLOT_PARAMS, PATH_PARAMS):
        
        self.PREPROCESSING_PARAMS = PREPROCESSING_PARAMS
        self.TRAINING_PARAMS = TRAINING_PARAMS
        self.INTERVALS_PARAMS = INTERVALS_PARAMS
        self.TH_ALGORITHM = TH_ALGORITHM
        self.PLOT_PARAMS = PLOT_PARAMS

        self.df_path = PATH_PARAMS['DF_PATH']
        self.df_train_path = PATH_PARAMS['DF_TRAIN_PATH']
        self.df_test_path = PATH_PARAMS['DF_TEST_PATH']
        self.batch_size = TRAINING_PARAMS['batch_size']
        self.epochs = TRAINING_PARAMS['epochs']
        self.hidden_size = TRAINING_PARAMS['hidden_size']
        self.single_file = PREPROCESSING_PARAMS['single_file']
        self.scaler = PREPROCESSING_PARAMS['normalization']
        self.columns_name = PREPROCESSING_PARAMS['metrics']
        self.alpha =  TRAINING_PARAMS['alpha']
        self.beta =  TRAINING_PARAMS['beta']
        
        self.df = None
        self.train_df = None
        self.test_df = None

        # self.w_size = windows_train.shape[1] * windows_train.shape[2]
        # self.z_size = windows_train.shape[1] * hidden_size
        self.device = utils.get_default_device()
        
        self.labels_ = None
        self.decision_scores_ = None
        self.anomalies_intervals_ = None
        
        
    def get_data(self, prep=True):
        if prep: scaler = self.scaler
        else: scaler = None
        
        if self.single_file is True:
            self.df = preprocessing.get_df(self.df_path, columns_name=self.columns_name)
            return preprocessing.data_preprocessing(
                                                    self.PREPROCESSING_PARAMS, self.df, 
                                                    INTERVALS_PARAMS=self.INTERVALS_PARAMS, 
                                                    scaler=scaler, multi=True, single_file=self.single_file
                                                )

        else:
            self.train_df = preprocessing.get_df(self.df_train_path, columns_name=self.columns_name)
            self.test_df = preprocessing.get_df(self.df_test_path, columns_name=self.columns_name)
            return preprocessing.data_preprocessing(
                                                    self.PREPROCESSING_PARAMS, df=None, 
                                                    INTERVALS_PARAMS=self.INTERVALS_PARAMS, 
                                                    scaler=scaler, multi=True, single_file=self.single_file, 
                                                    df_train=self.train_df, df_test=self.test_df
                                                )
        
    def get_db_time(self):
        if self.single_file is True:
            _, db_time = preprocessing.get_db_time(self.df, self.PREPROCESSING_PARAMS,
                                                           INTERVALS_PARAMS=self.INTERVALS_PARAMS, multi=True)
        else:
             db_time = preprocessing.get_db_time(self.test_df, self.PREPROCESSING_PARAMS, 
                                                    INTERVALS_PARAMS=None, multi=False)
                
        return db_time
        
    # TODO: add fit and predict
    def fit_predict(self, plot=True):
        df_train, df_test, windows_train, windows_test, train_timestamps, test_timestamps = self.get_data()
        w_size = windows_train.shape[1] * windows_train.shape[2]
        z_size = windows_train.shape[1] * self.hidden_size
        
        db_time = self.get_db_time()
        _, original_test_data, _, _, _, _ = self.get_data(prep=False)
        train_loader, val_loader, test_loader = preprocessing.get_dataloaders(
                                                                    windows_train, windows_test, 
                                                                    self.batch_size, w_size, z_size
                                                                )
        model = usad.UsadModel(w_size, z_size)
        model = utils.to_device(model, self.device)
        history = usad_utils.training(self.epochs, model, self.device, train_loader, val_loader)
        usad_utils.save_model(model)
        model = usad_utils.load_checkpoint(model)
        
        results = usad_utils.test_model(model, self.device, test_loader, alpha=self.alpha, beta=self.beta)
        y_pred = usad_utils.get_prediction_score(results)
        labels = th.get_labels(self.TH_ALGORITHM, y_pred) 
        
        anomalies_intervals_df = utils.generate_anomalies_intervals(labels, test_timestamps)
        utils.save_anomalies_intervals(anomalies_intervals_df)
        
        # update public variables
        self.decision_scores_ = y_pred
        self.labels_ = labels
        self.anomalies_intervals_ = anomalies_intervals_df
        
        if plot is True:
            plotter.plot_history(history, save_static=self.PLOT_PARAMS['history_static'], save_html=self.PLOT_PARAMS['history_html'])
            plotter.plot_res_db_time(y_pred, db_time, timestamps=test_timestamps, 
                                        save_static=self.PLOT_PARAMS['db_time_static'], save_html=self.PLOT_PARAMS['db_time_html'])
            plotter.plot_labels(y_pred, labels, timestamps=test_timestamps, 
                                    save_static=self.PLOT_PARAMS['labels_static'], save_html=self.PLOT_PARAMS['labels_html'])
            plotter.plot_thresholds(original_test_data, labels, timestamps=test_timestamps,
                                        save_static=self.PLOT_PARAMS['thresholds_static'], save_html=self.PLOT_PARAMS['thresholds_html'])