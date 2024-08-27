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
                mae_hinge = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEHinge']
                mae_margin = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEMargin']
                mae_pseudo = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEPseudo']
                mae_true = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAETrue']
                
                error_mae_hinge = mae_true.dropna() - mae_hinge.dropna()
                error_mae_margin = mae_true.dropna() - mae_margin.dropna()
                error_mae_pseudo = mae_true.dropna() - mae_pseudo.dropna()
                
                mean_error_mae_hinge = f"%.{N_DECIMALS}f" % round(np.mean(error_mae_hinge), N_DECIMALS)
                mean_error_mae_margin = f"%.{N_DECIMALS}f" % round(np.mean(error_mae_margin), N_DECIMALS)
                mean_error_mae_pseudo = f"%.{N_DECIMALS}f" % round(np.mean(error_mae_pseudo), N_DECIMALS)
                std_error_mae_hinge = f"%.{N_DECIMALS}f" % round(np.std(error_mae_hinge), N_DECIMALS)
                std_error_mae_margin = f"%.{N_DECIMALS}f" % round(np.std(error_mae_margin), N_DECIMALS)
                std_error_mae_pseudo = f"%.{N_DECIMALS}f" % round(np.std(error_mae_pseudo), N_DECIMALS)
                
                text += f"{mean_error_mae_hinge}$\pm${std_error_mae_hinge} & {mean_error_mae_margin}$\pm${std_error_mae_margin}" + \
                        f"& {mean_error_mae_pseudo}$\pm${std_error_mae_pseudo}"
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += " & "
            print(text)
        print()
        
        mean_error_mae_margin = round(np.mean(error_mae_margin), N_DECIMALS)