import pandas as pd
import config as cfg
import numpy as np

N_DECIMALS = 1

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
                mae_true = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAETrue']
                
                mean_mae_true = f"%.{N_DECIMALS}f" % round(np.mean(mae_true.dropna()), N_DECIMALS)
                std_mae_true = f"%.{N_DECIMALS}f" % round(np.std(mae_true.dropna()), N_DECIMALS)
                
                text += f"{mean_mae_true}$\pm${std_mae_true}"
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += " & "
            print(text)
        print()
        