import pandas as pd
import config as cfg
import numpy as np

N_DECIMALS = 0

if __name__ == "__main__":
    path = cfg.RESULTS_DIR
    results = pd.read_csv(f'{path}/model_results.csv', index_col=0)
    conditions = ["C1", "C2", "C3"]
    censoring = cfg.CENSORING_LEVELS
    for cond in conditions:
        text = ""            
        text += f"& LASSO & "
        for cens in censoring:
            ls_mae = mae_true = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens)]['LSMAE']

            mean_ls_mae = f"%.{N_DECIMALS}f" % round(np.mean(ls_mae.dropna()), N_DECIMALS)
            std_ls_mae = f"%.{N_DECIMALS}f" % round(np.std(ls_mae.dropna()), N_DECIMALS)
            
            text += f"{mean_ls_mae}$\pm${std_ls_mae}"
            if cens == 0.75:
                text += "\\\\"
            else:
                text += " & "
        print(text)
    print()
        