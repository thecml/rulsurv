from pathlib import Path
ROOT_DIR = Path(__file__).absolute().parent.parent

N_SIGNALS_XJTU = 3
N_REAL_BEARING_XJTU = 5
BASE_DYNAMIC_LOAD_XJTU = 12.82
BEARING_IDS = [1, 2, 3, 4, 5]
CONDITIONS = [0, 1, 2]
CENSORING_LEVELS = [0.25, 0.5, 0.75]
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
PLOTS_DIR = Path.joinpath(ROOT_DIR, 'plots')
DATASET_PATH_XJTU = Path.joinpath(ROOT_DIR, 'data/XJTU-SY/csv')
RAW_DATA_PATH_XJTU = [Path.joinpath(ROOT_DIR, 'data/XJTU-SY/35Hz12kN'),
                      Path.joinpath(ROOT_DIR, 'data/XJTU-SY/37.5Hz11kN'),
                      Path.joinpath(ROOT_DIR, 'data/XJTU-SY/40Hz10kN')]
                      
FREQUENCY_BANDS1 = {'xjtu_start': [12, 34, 71, 107, 171],
                    'xjtu_stop': [14, 36, 73, 109, 173]}
FREQUENCY_BANDS2 = {'xjtu_start': [13, 36, 76, 114, 183],
                    'xjtu_stop': [15, 38, 78, 116, 185]}
FREQUENCY_BANDS3 = {'xjtu_start': [14, 39, 82, 122, 195],
                    'xjtu_stop': [15, 41, 84, 124, 197]}

FREQUENCY_FTS = ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH']
NOISE_FT = ['noise']
NON_PH_FTS = {'xjtu_c1': ['skew', 'p2p'],
              'xjtu_c2': ['mean', 'skew', 'kurtosis', 'entropy', 'rms', 'p2p', 'clearence', 'shape'],
              'xjtu_c3': ['std', 'skew', 'kurtosis', 'rms', 'crest', 'shape', 'impulse']}
                
PARAMS_CPH = {'alpha': 0.0001,
              'tol': 0.1, 
              'n_iter': 50}

PARAMS_RSF = {'n_estimators': 100, 
              'min_samples_split': 2, 
              'min_samples_leaf': 4, 
              'max_depth': 7}

PARAMS_DEEPSURV = {'batch_size' : 16,
                   'learning_rate' : 0.01,
                   'iters': 50,
                   'layers': [32]}

PARAMS_DSM = {'batch_size' : 16,
              'learning_rate' : 0.01,
              'iters': 50,
              'layers': [32]}

PARAMS_BNN = {'layers' : [32],
              'regularization_pen': 0.001}

PARAMS_MTLR = {
    'hidden_size': 64,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'num_epochs': 1000,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 50}

PARAMS_COXBOOST = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'loss': 'coxph',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'dropout_rate': 0.0,
    'subsample': 1.0}

PARAMS_CPH_LASSO = {
    'l1_ratio': 1.0,
    'alpha_min_ratio': 0.5,
    'fit_baseline_model': True,
    'normalize': False,
    'tol': 1e-7,
    'max_iter': 100000
}