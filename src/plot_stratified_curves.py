import numpy as np
import pandas as pd
import config as cfg
import tensorflow as tf
import random
import warnings
import torch
from sksurv.util import Surv
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times
from sklearn.preprocessing import StandardScaler
from tools.regressors import RSF
from utility.survival import Survival
from tools.data_loader import DataLoader
from utility.survival import make_stratified_split

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
    for condition in [1]: # Use C1
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
        
        # Make unique times
        continuous_times = make_event_times(y_train['Survival_time'].copy(), y_train['Event'].copy()).astype(int)
        
        #Format the data
        t_train = y_train['Survival_time']
        e_train = y_train['Event']
        t_test = y_test['Survival_time']
        e_test = y_test['Event']

        # Predict per feature
        features_to_split = X_train.columns
        for feature in features_to_split:
            X_train_feature, X_test_feature = pd.DataFrame(X_train[feature]), pd.DataFrame(X_test[feature])
        
            # Scale train data
            scaler = StandardScaler()
            scaler.fit(X_train_feature)
            X_train_scaled = scaler.transform(X_train_feature)
            
            model = RSF().make_model(RSF().get_hyperparams(condition))
            model.fit(X_train_scaled, y_train)
            
            # Split data
            split_thresholds = []
            for qct in [0.25, 0.5, 0.75]:
                split_thresholds.append(round(X_test_feature[feature].quantile(qct), 2))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
            for i, st in enumerate(split_thresholds):
                g1_idx = X_test_feature.loc[X_test_feature[feature] < st].index
                g2_idx = X_test_feature.loc[X_test_feature[feature] >= st].index
                X_test_g1 = X_test_feature.loc[g1_idx]
                X_test_g2 = X_test_feature.loc[g2_idx]
                y_test_g1 = y_test[g1_idx]
                y_test_g2 = y_test[g2_idx]
        
                # Scale splitted data
                X_test_scaled = scaler.transform(X_test_feature)
                X_test_g1_scaled = scaler.transform(X_test_g1)
                X_test_g2_scaled = scaler.transform(X_test_g2)
        
                # Predict for mean and two groups
                surv_probs = Survival().predict_survival_function(model, X_test_scaled, continuous_times)
                km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Survival_time'].reshape(1,-1),
                                                                       y_test['Event'].reshape(1,-1),
                                                                       continuous_times)
                surv_probs_g1 = Survival().predict_survival_function(model, X_test_g1_scaled, continuous_times)
                km_mean_g1, km_high_g1, km_low_g1 = calculate_kaplan_vectorized(y_test_g1['Survival_time'].reshape(1,-1),
                                                                                y_test_g1['Event'].reshape(1,-1),
                                                                                continuous_times)
                surv_probs_g2 = Survival().predict_survival_function(model, X_test_g2_scaled, continuous_times)
                
                surv_probs_mean = np.mean(surv_probs, axis=0)
                surv_probs_g1_mean = np.mean(surv_probs_g1, axis=0)
                surv_probs_g2_mean = np.mean(surv_probs_g2, axis=0)
                
                # Plot
                axes[i].plot(surv_probs_mean, linewidth=2, label=r"$\mathbb{E}[S(t|$" + f"{feature}" + r"$)]$", color="black")
                axes[i].plot(surv_probs_g1_mean, linewidth=2, label=r"$S(t|$" + f"{feature} $<{st})$", color="C0")
                axes[i].plot(surv_probs_g2_mean, linewidth=2, label=r"$S(t|$" + f"{feature} $\geq{st})$", color="C1")
                axes[i].set_xlabel("Time (min)")
                axes[i].set_ylabel("Survival probability S(t)")
                axes[i].legend(loc='upper right')
                axes[i].grid(True)
            plt.savefig(f'{cfg.PLOTS_DIR}/group_survival_{feature}_C{condition+1}.pdf',
                        format='pdf', bbox_inches="tight")
            plt.close()
        