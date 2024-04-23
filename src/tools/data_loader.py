import numpy as np
import pandas as pd
import warnings
import torch
import math
import config as cfg
from sksurv.util import Surv
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold
from tools.regressors import CoxPH, RSF, MTLR, BNNSurv, CoxBoost
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from tools.formatter import Formatter
from tools.evaluator import LifelinesEvaluator
from utility.survival import Survival, make_event_times, coverage, make_time_bins
from tools.cross_validator import run_cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from utility.data import get_window_size, get_lag, get_lmd
from scipy.stats._stats_py import chisquare
from utility.event import EventManager
import tensorflow as tf
import random
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction

import config as cfg

class DataLoader:
    def __init__(self, dataset_name, axis, condition):
        self.dataset_name = dataset_name
        self.axis = axis
        self.condition = condition
        
        # XJTU-SY
        if dataset_name == "xjtu":
            self.dataset_path = cfg.DATASET_PATH_XJTU
        else:
            raise NotImplementedError()
    
        self.df_timeseries = None
        self.df_frequency = None
        self.event_times = None
        
    def load_data(self):
        self.df_timeseries, self.df_frequency = FileReader(self.dataset_name, self.dataset_path).read_data(self.condition, axis=self.axis)
        self.event_times = EventManager(self.dataset_name).get_event_times(self.df_frequency, self.condition, lmd=get_lmd(self.condition))
        return self
    
    def make_moving_average(self, bearing_id, drop_non_ph_fts=False):
        data_util = DataETL(self.dataset_name)
        event_time = self.event_times[bearing_id-1]
        transformed_data = data_util.make_moving_average(self.df_timeseries, event_time, bearing_id,
                                                         get_window_size(self.condition),
                                                         get_lag(self.condition))
        if drop_non_ph_fts:
            non_ph_fts = cfg.NON_PH_FTS[f'{self.dataset_name}_c{self.condition+1}']
            transformed_data.drop(non_ph_fts, axis=1, inplace=True)
        return transformed_data
