import pandas as pd
import config as cfg
import numpy as np

ALPHA = 0.05

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
                d_cal = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['DCalib']
                c_cal = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['CCalib']
                sum_d_cal = sum(1 for value in d_cal if value > ALPHA)
                sum_c_cal = sum(1 for value in c_cal if value > ALPHA)
                if model_name in ["CoxPH", "CoxBoost", "RSF", "MTLR"]:
                    text += f"{sum_d_cal}/5 & - "
                else:
                    text += f"{sum_d_cal}/5 & {sum_c_cal}/5 "
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += "& "
            print(text)
        print()