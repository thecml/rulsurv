import numpy as np
import pandas as pd
import torch
import config as cfg
from sksurv.util import Surv
from utility.survival import Survival
from tools.regressors import RSF
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
from sklearn.preprocessing import StandardScaler
from utility.survival import make_event_times, make_time_bins
from tools.data_loader import DataLoader
import tensorflow as tf
import random
import warnings

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
random.seed(0)

tf.config.set_visible_devices([], 'GPU') # use CPU

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

# Setup TF logging
import logging
tf.get_logger().setLevel(logging.ERROR)

DATASET_NAME = "xjtu"
AXIS = "X"
N_POST_SAMPLES = 100
BEARING_IDS = [1, 2, 3, 4, 5]
K = 1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if __name__ == "__main__":
    for condition in cfg.CONDITIONS:
        dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
        
        for test_bearing_id in BEARING_IDS:
            train_ids = [x for x in BEARING_IDS if x != test_bearing_id]
            
            # Load train data
            train_data = pd.DataFrame()
            for train_bearing_id in train_ids:
                df = dl.make_moving_average(train_bearing_id)
                train_data = pd.concat([train_data, df], axis=0)
            
            # Load test data
            test_data = dl.make_moving_average(test_bearing_id)
            
            # Reset index
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

            # Select Tk observations
            if K == 1:
                test_samples = test_data[test_data['Survival_time'] == test_data['Survival_time'].max()-5] \
                               .drop_duplicates(subset="Survival_time") # skip first 5
            else:
                surv_times = range(1, int(test_data['Survival_time'].max()+1))
                test_samples = pd.DataFrame()
                for k in range(1, K+1):
                    tk = int(np.quantile(surv_times, 1 / k))
                    tk_nearest = find_nearest(surv_times, tk)
                    test_sample = test_data[test_data['Survival_time'] == tk_nearest]
                    test_samples = pd.concat([test_samples, test_sample], axis=0)
                test_samples = test_samples.loc[test_samples['Event'] == True]
            
            if test_samples.empty:
                test_samples = test_data[test_data['Survival_time'] == test_data['Survival_time'].max()] \
                               .drop_duplicates(subset="Survival_time")
                
            X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            X_test = test_samples.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_samples)

            # Set event times for models
            continuous_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            discrete_times = make_time_bins(y_train['Survival_time'].copy(), event=y_train['Event'].copy())
            
            # Scale data
            features = list(X_train.columns)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
            
            # Set up RSF
            rsf_model = RSF().make_model(RSF().get_hyperparams(condition))
            
            # Train models
            rsf_model.fit(X_train_scaled, y_train)

            # Predict RSF
            rsf_surv_preds = Survival.predict_survival_function(rsf_model, X_test_scaled, continuous_times)
            
            # Calculate TTE
            for surv_preds in [rsf_surv_preds]:
                # Sanitize
                surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
                bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
                sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
                sanitized_y_test = np.delete(y_test, bad_idx)
                
                lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_y_test['Survival_time'], sanitized_y_test['Event'],
                                                    y_train['Survival_time'], y_train['Event'])
                median_survs = lifelines_eval.predict_time_from_curve(predict_median_survival_time)
                    
                # Calculate CRA
                cra = 0
                n_preds = len(median_survs)
                for k in range(1, n_preds+1):
                    wk = k/sum(range(n_preds+1))
                    ra_tk = 1 - (abs(sanitized_y_test['Survival_time'][k-1]-median_survs[k-1])/
                                sanitized_y_test['Survival_time'][k-1])
                    cra += wk*ra_tk
                print(f'CRA for Bearing {condition+1}_{test_bearing_id}: {round(cra, 4)}')
            print()