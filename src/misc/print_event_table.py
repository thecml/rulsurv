import pandas as pd
import config as cfg
import numpy as np
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import EventManager
from utility.data import get_lmd

NEW_DATASET = False

if __name__ == "__main__":
    dataset = "xjtu"
    axis = "X"
    n_boot = 0
    dataset_path = cfg.DATASET_PATH_XJTU
    n_condition = len(cfg.RAW_DATA_PATH_XJTU)
    bearing_ids = cfg.BEARING_IDS
    
    if NEW_DATASET == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
    
    for test_condition in cfg.CONDITIONS:
        pct_error_list = list()
        _, analytic = FileReader(dataset, dataset_path).read_data(test_condition, axis=axis)
        event_manager = EventManager(dataset)
        event_times = event_manager.get_event_times(analytic, test_condition,
                                                    get_lmd(test_condition))
        failure_times = event_manager.get_eol_times(analytic)
        for bearing_id in bearing_ids:
            event_time = event_times[bearing_id-1]
            failure_time = failure_times[bearing_id-1]
            error = event_time - failure_time
            pct_error = ((event_time - failure_time)/ failure_time) * 100
            pct_error_list.append(pct_error)
            bearing_name = f'& Bearing {test_condition+1}\_{bearing_id}'
            print(f"{bearing_name} & {event_time} & {failure_time} & {error} & {abs(round(pct_error, 1))} \\\\")
        print()