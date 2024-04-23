import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utility.printer import Suppressor
from utility.survival import make_event_times, convert_to_structured, make_time_bins
from pycox.evaluation import EvalSurv
from tools.evaluator import LifelinesEvaluator
import config as cfg
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def run_cross_validation(model_builder, data, param_list, device, n_internal_splits):
    sample_results = pd.DataFrame()
    model_name = model_builder.__name__
    for sample in param_list:
        param_results = pd.DataFrame()
        kf = KFold(n_splits=n_internal_splits, random_state=0, shuffle=True)
        for split_idx, (train_idx, test_idx) in enumerate(kf.split(data[0], data[1])):
            t_train = np.array(data[1][train_idx]["Survival_time"])
            e_train = np.array(data[1][train_idx]["Event"])
            t_test = np.array(data[1][test_idx]["Survival_time"])
            e_test = np.array(data[1][test_idx]["Event"])
            x_train =  np.array(data[0].iloc[train_idx])
            x_test =  np.array(data[0].iloc[test_idx])
            event_horizon = make_event_times(t_train, e_train).astype(int)
            mtlr_times = make_time_bins(t_train, event=e_train)
            if model_name in ["DeepSurv", "DSM"]:
                model = model_builder().make_model(sample)
                model = model.fit(x_train, t_train, e_train, vsize=0.3,
                                  iters=sample['iters'],
                                  learning_rate=sample['learning_rate'],
                                  batch_size=sample['batch_size'])
                preds = pd.DataFrame(model.predict_survival(x_test, t=list(event_horizon)), columns=event_horizon)
            elif model_name == "MTLR":
                X_train, X_valid, y_train, y_valid = train_test_split(data[0].iloc[train_idx], data[1][train_idx],
                                                                      test_size=0.3, random_state=0)
                X_train = X_train.reset_index(drop=True)
                X_valid = X_valid.reset_index(drop=True)
                data_train = X_train.copy()
                data_train["Survival_time"] = pd.Series(y_train['Survival_time'])
                data_train["Event"] = pd.Series(y_train['Event']).astype(int)
                data_valid = X_valid.copy()
                data_valid["Survival_time"] = pd.Series(y_valid['Survival_time'])
                data_valid["Event"] = pd.Series(y_valid['Event']).astype(int)
                data_test = data[0].iloc[test_idx].copy()
                data_test["Survival_time"] = pd.Series(t_test)
                data_test["Event"] = pd.Series(e_test).astype(int)
                config = dotdict(cfg.PARAMS_MTLR)
                config['batch_size'] = sample['batch_size']
                config['dropout'] = sample['dropout']
                config['lr'] = sample['lr']
                config['c1'] = sample['c1']
                config['num_epochs'] = sample['num_epochs']
                config['hidden_size'] = sample['hidden_size']
                num_features = x_train.shape[1]
                num_time_bins = len(mtlr_times)
                model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
                model = train_mtlr_model(model, data_train, data_valid, mtlr_times,
                                         config, random_state=0, reset_model=True, device=device)
                baycox_test_data = torch.tensor(data_test.drop(["Survival_time", "Event"], axis=1).values,
                                                dtype=torch.float, device=device)
                survival_outputs, _, _ = make_mtlr_prediction(model, baycox_test_data, mtlr_times, config)
                surv_preds = survival_outputs.numpy()
                mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
                preds = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
            elif model_name == "BNNSurv":
                model = model_builder().make_model(sample)
                model.fit(x_train, t_train, e_train)
                preds = pd.DataFrame(np.mean(model.predict_survival(x_test, event_horizon), axis=0), columns=event_horizon)
            else:
                model = model_builder().make_model(sample)
                y_train = convert_to_structured(t_train, e_train)
                model.fit(x_train, y_train)
                preds = pd.DataFrame(np.row_stack([fn(event_horizon)
                                                   for fn in model.predict_survival_function(x_test)]), columns=event_horizon)
            preds = preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
            bad_idx = preds[preds.iloc[:,0] < 0.5].index # check we have a median
            sanitized_preds = preds.drop(bad_idx).reset_index(drop=True)
            sanitized_t_test = np.delete(t_test, bad_idx)
            sanitized_e_test = np.delete(e_test, bad_idx)
            try:
                lifelines_eval = LifelinesEvaluator(sanitized_preds.T, sanitized_t_test, sanitized_e_test, t_train, e_train)
                mae = lifelines_eval.mae(method="Hinge")
            except:
                mae = np.nan
            res_sr = pd.Series([str(model_name), split_idx, sample, mae],
                            index=["ModelName", "SplitIdx", "Params", "MAE"])
            param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
        mean_mae = param_results['MAE'].mean()
        res_sr = pd.Series([str(model_name), param_results.iloc[0]["Params"], mean_mae],
                        index=["ModelName", "Params", "MAE"])
        sample_results = pd.concat([sample_results, res_sr.to_frame().T], ignore_index=True)
    best_params = sample_results.loc[sample_results['MAE'].astype(float).idxmin()]['Params']
    return best_params