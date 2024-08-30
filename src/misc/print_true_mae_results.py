import sys, os
sys.path.append(os.getcwd() + "\\src")

import pandas as pd
import config as cfg
import numpy as np

N_DECIMALS = 1

def calculate_improvement(metric, baseline):
    mae_improvement = round(((float(metric) - float(baseline)) / float(baseline)) * 100, N_DECIMALS)
    mae_sign = "+" if mae_improvement > 0 else ""
    mae_color = "red" if mae_improvement > 0 else "green"
    mae_improvement_text = f"({mae_sign}{mae_improvement})"
    mae_improvement_text = "\\textcolor{" + f"{mae_color}" + "}" + "{" + f"{mae_improvement_text}" + "}"
    return mae_improvement_text

if __name__ == "__main__":
    path = cfg.RESULTS_DIR
    results = pd.read_csv(f'{path}/model_results.csv', index_col=0)
    conditions = ["C1", "C2", "C3"]
    censoring = cfg.CENSORING_LEVELS
    model_names = ["CoxPH", "CoxBoost", "RSF", "MTLR", "BNNSurv"]
    for cond in conditions:
        for index, model_name in enumerate(model_names):
            text = ""
            text += f"& {model_name} & "
            for cens in censoring:
                ls_mae = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens)]['LSMAE']
                mae_true = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAETrue']
                
                mean_ls_mae = f"%.{N_DECIMALS}f" % round(np.mean(ls_mae.dropna()), N_DECIMALS)
                mean_mae_true = f"%.{N_DECIMALS}f" % round(np.mean(mae_true.dropna()), N_DECIMALS)
                std_mae_true = f"%.{N_DECIMALS}f" % round(np.std(mae_true.dropna()), N_DECIMALS)
                
                improvement_text = calculate_improvement(mean_mae_true, mean_ls_mae)
                
                text += f"{mean_mae_true}$\pm${std_mae_true} {improvement_text}"
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += " & "
            print(text)
        print()
        