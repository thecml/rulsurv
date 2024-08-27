import pandas as pd
import config as cfg
from glob import glob
import numpy as np
import re

if __name__ == "__main__":
    path = cfg.RESULTS_PATH
    results = pd.read_csv(f'{path}/model_results.csv', index_col=0)
    conditions = ["C1", "C2", "C3"]
    censoring = [0.25]
    model_names = ["CoxPH", "CoxBoost", "RSF", "MTLR", "BNNSurv"]
    for cond in conditions:
        for index, model_name in enumerate(model_names):
            text = ""            
            text += f"& {model_name} & "
            for cens in censoring:
                best_params =  results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['BestParams']
                text += str(best_params.values)
            print(text)
        print()