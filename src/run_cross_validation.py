import numpy as np
import pandas as pd
import warnings
import torch
import math
import config as cfg
import tensorflow as tf
import random
from sksurv.util import Surv
from tools.regressors import CoxPH, CoxBoost, RSF, MTLR, BNNSurv
from tools.formatter import Formatter
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import make_monotonic
from utility.survival import Survival, make_event_times, coverage, make_time_bins, make_stratification_label
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats._stats_py import chisquare
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from tools.data_loader import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tools.Evaluations.MeanError import mean_error
from tools.Evaluations.util import predict_median_survival_time
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.simplefilter(action='ignore', category=FutureWarning)

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
N_ITER = 10
N_SPLITS = 5
N_POST_SAMPLES = 100
BEARING_IDS = [1, 2, 3, 4, 5]
N_CONDITION = len(cfg.RAW_DATA_PATH_XJTU)

def main():
    models = [CoxPH, CoxBoost, RSF, MTLR, BNNSurv]
    model_results = pd.DataFrame()
    
    # Run cross-validation per condition and censoring level
    for condition in cfg.CONDITIONS:
        dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
        for pct in cfg.CENSORING_LEVELS:
            data = pd.DataFrame()
            for bearing_id in BEARING_IDS:
                df = dl.make_moving_average(bearing_id)
                df['TrueTime'] = df['Survival_time']
                df = Formatter.add_random_censoring(df, pct)
                df = df.sample(frac=1, random_state=0)
                data = pd.concat([data, df], axis=0)
            
            # Make stratified split
            stra_lab = make_stratification_label(data)
            mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
            for i, (train_index, test_index) in enumerate(mskf.split(data, stra_lab)):
                train_data = data.iloc[train_index]
                test_data = data.iloc[test_index]
                
                # Evaluate each model
                for model_builder in models:
                    model_name = model_builder.__name__
                    
                    # Format data
                    train_x = train_data.drop(['Event', 'Survival_time', 'TrueTime'], axis=1)
                    train_y = Surv.from_dataframe("Event", "Survival_time", train_data)
                    test_x = test_data.drop(['Event', 'Survival_time', 'TrueTime'], axis=1)
                    test_y = Surv.from_dataframe("Event", "Survival_time", test_data)
                    
                    # Scale data
                    features = list(train_x.columns)
                    scaler = StandardScaler()
                    scaler.fit(train_x)
                    ti = (pd.DataFrame(scaler.transform(train_x), columns=features), train_y)
                    cvi = (pd.DataFrame(scaler.transform(test_x), columns=features), test_y)
                    
                    # Make event times
                    continuous_times = make_event_times(ti[1]['Survival_time'].copy(), ti[1]['Event'].copy()).astype(int)
                    discrete_times = make_time_bins(ti[1]['Survival_time'].copy(), event=ti[1]['Event'].copy())

                    # Get hyperparams
                    params = model_builder().get_hyperparams(condition)
                    
                    # Train on train set TI with new parameters
                    x = ti[0].to_numpy()
                    t = ti[1]["Survival_time"]
                    e = ti[1]["Event"]
                    if model_name == "MTLR":
                        X_train, X_valid, y_train, y_valid = train_test_split(ti[0], ti[1], test_size=0.3, random_state=0)
                        X_train = X_train.reset_index(drop=True)
                        X_valid = X_valid.reset_index(drop=True)
                        data_train = X_train.copy()
                        data_train["Survival_time"] = pd.Series(y_train['Survival_time'])
                        data_train["Event"] = pd.Series(y_train['Event']).astype(int)
                        data_valid = X_valid.copy()
                        data_valid["Survival_time"] = pd.Series(y_valid['Survival_time'])
                        data_valid["Event"] = pd.Series(y_valid['Event']).astype(int)
                        config = dotdict(cfg.PARAMS_MTLR)
                        config['batch_size'] = params['batch_size']
                        config['dropout'] = params['dropout']
                        config['lr'] = params['lr']
                        config['c1'] = params['c1']
                        config['num_epochs'] = params['num_epochs']
                        config['hidden_size'] = params['hidden_size']
                        num_features = ti[0].shape[1]
                        num_time_bins = len(discrete_times)
                        model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
                        model = train_mtlr_model(model, data_train, data_valid, discrete_times,
                                                 config, random_state=0, reset_model=True, device=device)
                    elif model_name == "BNNSurv":
                        model = model_builder().make_model(params)
                        model.fit(x, t, e)
                    else:
                        model = model_builder().make_model(params)
                        model.fit(ti[0], ti[1])
              
                    # Get survival predictions for CVI
                    if model_name == "DeepSurv" or model_name == "DSM" or model_name == "BNNSurv":
                        xte = cvi[0].to_numpy()
                        surv_preds = Survival.predict_survival_function(model, xte, continuous_times, n_post_samples=N_POST_SAMPLES)
                    elif model_name == "MTLR":
                        data_test = cvi[0].copy()
                        data_test["Survival_time"] = pd.Series(cvi[1]['Survival_time'])
                        data_test["Event"] = pd.Series(cvi[1]['Event']).astype(int)
                        mtlr_test_data = torch.tensor(data_test.drop(["Survival_time", "Event"], axis=1).values,
                                                      dtype=torch.float, device=device)
                        survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, discrete_times, config)
                        surv_preds = survival_outputs.numpy()
                        discrete_times = torch.cat([torch.tensor([0]).to(discrete_times.device), discrete_times], 0)
                        surv_preds = pd.DataFrame(surv_preds, columns=discrete_times.numpy())
                    else:
                        surv_preds = Survival.predict_survival_function(model, cvi[0], continuous_times)
                        
                    # Ensure proper survival curve
                    surv_preds[0] = 1
                    surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
                        
                    # Fit LASSO for comparasion
                    ls = Lasso(alpha=1.0, random_state=0, max_iter=5000)
                    x_train_ls = train_data.loc[train_data['Event'] == True].drop(['Event', 'Survival_time', 'TrueTime'], axis=1)
                    x_test_ls = test_data.drop(['Event', 'Survival_time', 'TrueTime'], axis=1)
                    y_train_ls = train_data.loc[train_data['Event'] == True]['Survival_time']
                    y_test_ls = test_data['Survival_time']
                    ls.fit(x_train_ls, y_train_ls)
                    y_pred_ls = ls.predict(x_test_ls)
                    ls_mae = mean_absolute_error(y_test_ls, y_pred_ls)
                    
                    # Calculate scores
                    try:
                        lifelines_eval = LifelinesEvaluator(surv_preds.T, cvi[1]['Survival_time'], cvi[1]['Event'],
                                                            ti[1]['Survival_time'], ti[1]['Event'])
                        mae_hinge = lifelines_eval.mae(method="Hinge")
                        mae_margin = lifelines_eval.mae(method="Margin")
                        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                        median_survs = lifelines_eval.predict_time_from_curve(predict_median_survival_time)
                        d_calib = lifelines_eval.d_calibration()[0]
                    except:
                        mae_hinge = np.nan
                        mae_margin = np.nan
                        mae_pseudo = np.nan
                        d_calib = np.nan

                    if mae_hinge > 1000:
                        mae_hinge = np.nan
                    if mae_margin > 1000:
                        mae_margin = np.nan
                    if mae_pseudo > 1000:
                        mae_pseudo = np.nan
                    
                    if condition == 0:
                        cond_name = "C1"
                    elif condition == 1:
                        cond_name = "C2"
                    else:
                        cond_name = "C3"
                        
                    # Calculate true MAE
                    true_times = np.array(test_data['TrueTime'])
                    event_indicators = np.array([1] * len(true_times))
                    true_mae = mean_error(median_survs, true_times, event_indicators, method='Uncensored')
                    
                    # Calucate C-cal for BNN model
                    if model_name == "BNNSurv":
                        xte = cvi[0].to_numpy()
                        surv_probs = model.predict_survival(xte, event_times=continuous_times,
                                                            n_post_samples=N_POST_SAMPLES)
                        credible_region_sizes = np.arange(0.1, 1, 0.1)
                        surv_times = torch.from_numpy(surv_probs)
                        coverage_stats = {}
                        for percentage in credible_region_sizes:
                            drop_num = math.floor(0.5 * N_POST_SAMPLES * (1 - percentage))
                            lower_outputs = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
                            upper_outputs = torch.kthvalue(surv_times, k=N_POST_SAMPLES - drop_num, dim=0)[0]
                            coverage_stats[percentage] = coverage(continuous_times, upper_outputs, lower_outputs,
                                                                  cvi[1]["Survival_time"], cvi[1]["Event"])
                        expected_percentages = coverage_stats.keys()
                        observed_percentages = coverage_stats.values()
                        expected = [x / sum(expected_percentages) * 100 for x in expected_percentages] # normalize
                        observed = [x / sum(observed_percentages) * 100 for x in observed_percentages]
                        _, p_value = chisquare(f_obs=observed, f_exp=expected)
                        c_calib = p_value
                    else:
                        c_calib = 0

                    try:
                        print(f"Evaluated {cond_name} - {model_name} - {pct} - {round(mae_hinge)} - {round(mae_margin)}" +
                              f" - {round(mae_pseudo)} - {round(true_mae)} - {round(ls_mae)}")
                    except:
                        print("Print failed, probably has NaN in results...")
                        
                    res_sr = pd.Series([cond_name, model_name, pct, mae_hinge, mae_margin,
                                        mae_pseudo, true_mae, ls_mae, d_calib, c_calib],
                                        index=["Condition", "ModelName", "CensoringLevel", "MAEHinge", "MAEMargin",
                                               "MAEPseudo", "MAETrue", "LSMAE", "DCalib", "CCalib"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")

if __name__ == "__main__":
    main()