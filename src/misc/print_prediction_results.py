import sys, os
sys.path.append(os.getcwd() + "\\src")

import pandas as pd
import config as cfg
import numpy as np

N_DECIMALS = 1

def calculate_improvement(metric, baseline):
    mae_improvement = round(((float(metric) - float(baseline)) / float(baseline)) * 100, N_DECIMALS)
    mae_sign = "+" if mae_improvement > 0 else ""
    mae_color = "dimRed" if mae_improvement > 0 else "dimGreen"
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
                mae_hinge_bl = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == "CoxPH")]['MAEHinge']
                mae_margin_bl = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == "CoxPH")]['MAEMargin']
                mae_pseudo_bl = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == "CoxPH")]['MAEPseudo']
                mae_hinge = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEHinge']
                mae_margin = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEMargin']
                mae_pseudo = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEPseudo']
                mean_mae_hinge = f"%.{N_DECIMALS}f" % round(np.mean(mae_hinge.dropna()), N_DECIMALS)
                mean_mae_margin = f"%.{N_DECIMALS}f" % round(np.mean(mae_margin.dropna()), N_DECIMALS)
                mean_mae_pseudo = f"%.{N_DECIMALS}f" % round(np.mean(mae_pseudo.dropna()), N_DECIMALS)
                bl_mean_mae_hinge = f"%.{N_DECIMALS}f" % round(np.mean(mae_hinge_bl.dropna()), N_DECIMALS)
                bl_mean_mae_margin = f"%.{N_DECIMALS}f" % round(np.mean(mae_margin_bl.dropna()), N_DECIMALS)
                bl_mean_mae_pseudo = f"%.{N_DECIMALS}f" % round(np.mean(mae_pseudo_bl.dropna()), N_DECIMALS)
                std_mae_hinge = f"%.{N_DECIMALS}f" % round(np.std(mae_hinge.dropna()), N_DECIMALS)
                std_mae_margin = f"%.{N_DECIMALS}f" %round(np.std(mae_margin.dropna()), N_DECIMALS)
                std_mae_pseudo = f"%.{N_DECIMALS}f" %round(np.std(mae_pseudo.dropna()), N_DECIMALS)
                improvement_text_hinge = calculate_improvement(mean_mae_hinge, bl_mean_mae_hinge)
                improvement_text_margin = calculate_improvement(mean_mae_margin, bl_mean_mae_margin)
                improvement_text_pseudo = calculate_improvement(mean_mae_pseudo, bl_mean_mae_pseudo)
                if model_name == "CoxPH":
                    text += f"{mean_mae_hinge}$\pm${std_mae_hinge} & {mean_mae_margin}$\pm${std_mae_margin} & {mean_mae_pseudo}$\pm${std_mae_pseudo}"
                else:
                    text += f"{mean_mae_hinge}$\pm${std_mae_hinge} {improvement_text_hinge} & {mean_mae_margin}$\pm${std_mae_margin} {improvement_text_margin} & {mean_mae_pseudo}$\pm${std_mae_pseudo} {improvement_text_pseudo}"
                if cens == 0.75:
                    text += "\\\\"
                else:
                    text += " & "
            print(text)
        print()
        