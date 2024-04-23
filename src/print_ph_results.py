import pandas as pd
import config as cfg
import numpy as np
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import EventManager
from tools.data_ETL import DataETL
from utility.data import get_window_size, get_lag, get_lmd
from lifelines import CoxPHFitter
from textwrap import fill
import numpy as np
from lifelines.statistics import proportional_hazard_test, TimeTransformers
from lifelines.utils import format_p_value
from lifelines.utils.lowess import lowess
from tools.data_loader import DataLoader

NEW_DATASET = False

pd.set_option('display.float_format', lambda x: '%.3f' % x)

if __name__ == "__main__":
    dataset_name = "xjtu"
    axis = "X"
    bearing_ids = [1, 2, 3, 4, 5]
    n_boot = 0
    
    results = pd.DataFrame()
    covariate_names = ["mean", "std", "skew", "kurtosis", "entropy", "rms",
                       "max", "p2p", "crest", "clearence", "shape", "impulse"]
    results['Covariates'] = covariate_names
    for condition in [0, 1, 2]:
        dl = DataLoader(dataset_name, axis, condition).load_data()
        data = pd.DataFrame()
        for bearing_id in bearing_ids:
            data = pd.concat([data, dl.make_moving_average(bearing_id)], axis=0)
        df = data.reset_index(drop=True)
        cph = CoxPHFitter()
        cph.fit(df, duration_col="Survival_time", event_col= "Event")
        residuals = cph.compute_residuals(df, kind="scaled_schoenfeld")
        test_results = proportional_hazard_test(cph, df, time_transform=["rank"],
                                                precomputed_residuals=residuals)
        results[f'C{condition+1}'] = test_results.p_value
    print(results)
        