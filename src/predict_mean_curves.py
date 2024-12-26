import numpy as np
import pandas as pd
import config as cfg
from sksurv.util import Surv
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times, make_time_bins
from sklearn.preprocessing import StandardScaler
from tools.regressors import CoxPH, CoxBoost, RSF, MTLR, BNNSurv
from utility.survival import Survival
from utility.mtlr import train_mtlr_model, make_mtlr_prediction
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
from tools.data_loader import DataLoader
from utility.survival import make_stratified_split
import warnings

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'medium',
                     'axes.titlesize': 'medium',
                     'font.size': 12.0,
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})

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
PCT_CENSORING = 0.25
N_POST_SAMPLES = 100
BEARING_IDS = cfg.BEARING_IDS

if __name__ == "__main__":
    for condition in [0, 1, 2]:
        dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
        data = pd.DataFrame()
        for bearing_id in BEARING_IDS:
            df = dl.make_moving_average(bearing_id)
            df = Formatter.add_random_censoring(df, PCT_CENSORING)
            df = df.sample(frac=1, random_state=0)
            data = pd.concat([data, df], axis=0)
        
        train_data, _, test_data = make_stratified_split(data, stratify_colname='both', frac_train=0.7, frac_test=0.3, random_state=0)

        X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
        y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
        X_test = test_data.drop(['Event', 'Survival_time'], axis=1)
        y_test = Surv.from_dataframe("Event", "Survival_time", test_data)
        
        # Make times
        continuous_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
        discrete_times = make_time_bins(y_train['Survival_time'].copy(), event=y_train['Event'].copy())
        
        # Scale data
        features = list(X_train.columns)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

        # Format training data for BNN model
        X_train_arr = X_train.to_numpy()
        X_test_arr = X_test.to_numpy()
        t_train = y_train['Survival_time']
        e_train = y_train['Event']
        
        # Format data for MTLR model
        X_train_mtlr, X_valid_mtlr, y_train_mtlr, y_valid_mtlr = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
        X_train_mtlr = X_train_mtlr.reset_index(drop=True)
        X_valid_mtlr = X_valid_mtlr.reset_index(drop=True)
        data_train = X_train_mtlr.copy()
        data_train["Survival_time"] = pd.Series(y_train_mtlr['Survival_time'])
        data_train["Event"] = pd.Series(y_train_mtlr['Event']).astype(int)
        data_valid = X_valid_mtlr.copy()
        data_valid["Survival_time"] = pd.Series(y_valid_mtlr['Survival_time'])
        data_valid["Event"] = pd.Series(y_valid_mtlr['Event']).astype(int)
        
        # Make models
        cph_model = CoxPH().make_model(CoxPH().get_hyperparams(condition))
        coxboost_model = CoxBoost().make_model(CoxBoost().get_hyperparams(condition))
        rsf_model = RSF().make_model(RSF().get_hyperparams(condition))
        bnn_model = BNNSurv().make_model(BNNSurv().get_hyperparams(condition))
        config = dotdict(cfg.PARAMS_MTLR)
        best_params = MTLR().get_hyperparams(condition)
        config['batch_size'] = best_params['batch_size']
        config['dropout'] = best_params['dropout']
        config['lr'] = best_params['lr']
        config['c1'] = best_params['c1']
        config['num_epochs'] = best_params['num_epochs']
        config['hidden_size'] = best_params['hidden_size']
        num_features = X_train_mtlr.shape[1]
        num_time_bins = len(discrete_times)
        mtlr_model = MTLR().make_model(num_features=num_features,
                                       num_time_bins=num_time_bins,
                                       config=config)
    
        # Train models
        cph_model.fit(X_train, y_train)
        coxboost_model.fit(X_train, y_train)
        rsf_model.fit(X_train, y_train)
        bnn_model.fit(X_train_arr, t_train, e_train)
        mtlr_model = train_mtlr_model(mtlr_model, data_train, data_valid, discrete_times,
                                      config, random_state=0, reset_model=True, device=device)
        
        # Predict
        km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Survival_time'].reshape(1,-1),
                                                               y_test['Event'].reshape(1,-1),
                                                               continuous_times)
        cph_surv_func = Survival().predict_survival_function(cph_model, X_test, continuous_times)
        coxboost_surv_func = Survival().predict_survival_function(coxboost_model, X_test, continuous_times)
        rsf_surv_func = Survival().predict_survival_function(rsf_model, X_test, continuous_times)
        bnn_surv_func = Survival().predict_survival_function(bnn_model, X_test_arr, continuous_times,
                                                             n_post_samples=N_POST_SAMPLES)
        data_test = X_test.copy()
        data_test["Survival_time"] = pd.Series(y_test['Survival_time'])
        data_test["Event"] = pd.Series(y_test['Event']).astype(int)
        baycox_test_data = torch.tensor(data_test.drop(["Survival_time", "Event"], axis=1).values,
                                        dtype=torch.float, device=device)
        survival_outputs, _, _ = make_mtlr_prediction(mtlr_model, baycox_test_data, discrete_times, config)
        surv_preds = survival_outputs.numpy()
        discrete_times = torch.cat([torch.tensor([0]).to(discrete_times.device), discrete_times], 0)
        mtlr_surv_func = pd.DataFrame(surv_preds, columns=discrete_times.numpy())

        fig = plt.figure(figsize=(6, 4))
        plt.plot(km_mean.columns, km_mean.iloc[0], 'k--', linewidth=2, alpha=0.5, label=r"$\mathbb{E}[S(t)]$ Kaplan-Meier", color="black")
        plt.plot(continuous_times, np.mean(cph_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ CoxPH", linewidth=2)
        plt.plot(continuous_times, np.mean(coxboost_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ GBSA", linewidth=2)
        plt.plot(continuous_times, np.mean(rsf_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ RSF", linewidth=2)
        plt.plot(discrete_times, np.mean(mtlr_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ MTLR", linewidth=2)
        plt.plot(continuous_times, np.mean(bnn_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ BNNSurv", linewidth=2)
        plt.xlabel("Time (min)")
        plt.ylabel("Event probability S(t)")
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.savefig(f'{cfg.PLOTS_DIR}/mean_survival_C{condition+1}.pdf', format='pdf', bbox_inches="tight")
        plt.close()
        