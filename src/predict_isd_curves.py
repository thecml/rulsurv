import numpy as np
import pandas as pd
import config as cfg
from sksurv.util import Surv
from tools.formatter import Formatter
from sklearn.preprocessing import StandardScaler
from tools.regressors import BNNSurv
import torch
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
from matplotlib.patches import Rectangle
import math
import tensorflow as tf
import random
from tools.data_loader import DataLoader
import warnings
from utility.survival import make_event_times

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
BEARING_IDS = [1, 2, 3, 4, 5]
PLOT_INDICIES = [0, 1, 2, 3, 4]
PCT_CENSORING = 0.25
N_POST_SAMPLES = 100

if __name__ == "__main__":
    fig, axes = plt.subplots(5, 3, figsize=(12, 16), sharey=True, layout='constrained')
    for condition in cfg.CONDITIONS:
        dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
        for test_bearing_id, plot_idx in zip(BEARING_IDS, PLOT_INDICIES):
            # Define train ids
            train_ids = [x for x in BEARING_IDS if x != test_bearing_id]
            
            # Load train data
            train_data = pd.DataFrame()            
            for bearing_id in train_ids:
                df = dl.make_moving_average(bearing_id)
                df = Formatter.add_random_censoring(df, PCT_CENSORING)
                df = df.sample(frac=1, random_state=0)
                train_data = pd.concat([train_data, df], axis=0)
            
            # Load test data
            test_data = dl.make_moving_average(test_bearing_id)
            
            # Select first observation
            test_sample = test_data[test_data['Survival_time'] == test_data['Survival_time'].max()-5] \
                                                                  .drop_duplicates(subset="Survival_time") # skip first 5

            X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            X_test = test_sample.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_sample)
            
            # Set event times for models
            continuous_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            
            # Scale data
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Format data
            t_train = y_train['Survival_time']
            e_train = y_train['Event']
            t_test = y_test['Survival_time']
            e_test = y_test['Event']
            
            # Define test event time
            test_event_time = int(t_test[0])
            
            #Set up the models on test
            model = BNNSurv().make_model(BNNSurv().get_hyperparams(condition))

            # Train the model
            model.fit(X_train_scaled, t_train, e_train)
            
            # Predict
            surv_probs = model.predict_survival(X_test_scaled, continuous_times, N_POST_SAMPLES)
            median_outputs = pd.DataFrame(np.mean(surv_probs, axis=0), columns=continuous_times)
            
            # Calculate TTE
            lifelines_eval = LifelinesEvaluator(median_outputs.T, t_test, e_test, t_train, e_train)
            pred_survival_time = int(lifelines_eval.predict_time_from_curve(predict_median_survival_time))
            
            # Plot
            p1 = axes[plot_idx, condition].plot(np.mean(median_outputs, axis=0).T, linewidth=2, label=r"$\mathbb{E}[S(t|\bm{X})]$", color="black")
            drop_num = math.floor(0.5 * N_POST_SAMPLES * (1 - 0.9))
            lower_outputs = torch.kthvalue(torch.from_numpy(surv_probs), k=1+drop_num, dim=0)[0]
            upper_outputs = torch.kthvalue(torch.from_numpy(surv_probs), k=N_POST_SAMPLES-drop_num, dim=0)[0]
            axes[plot_idx, condition].fill_between(continuous_times, upper_outputs[0,:], lower_outputs[0,:], color="gray", alpha=0.25)
            p2 = axes[plot_idx, condition].axhline(y=0.5, linestyle= "dashed", color='blue', linewidth=1, label='$\hat{t}_{i}$ = ' + f'{pred_survival_time}')        
            p3 = axes[plot_idx, condition].axvline(x=test_event_time, linestyle= "dashed",
                                                   color='green', linewidth=2.0, label=f'$t_i$ = {int(test_event_time)}')
            axes[plot_idx, condition].axvline(x=int(pred_survival_time), linestyle= "dashed", color='blue', linewidth=2.0)
            axes[plot_idx, condition].set_title(f'Bearing {condition+1}_{plot_idx+1}')
            axes[plot_idx, condition].set_xlabel("Time (min)")
            text = f'Error = {int(pred_survival_time-test_event_time)}'
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            axes[plot_idx, condition].legend([p1[0], p2, p3, extra], [p1[0].get_label(), p2.get_label(), p3.get_label(), text], loc='upper right')
            if condition == 0:
                axes[plot_idx, condition].set_ylabel("Survival probability S(t)")
            axes[plot_idx, condition].grid(True)
    plt.savefig(f'{cfg.PLOTS_DIR}/isd.pdf', format='pdf', bbox_inches="tight")
    plt.close()