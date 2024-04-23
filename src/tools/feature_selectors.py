import pandas as pd
import numpy as np
import config as cfg
from abc import ABC, abstractmethod
from typing import List
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest

class SelectAllFeatures ():
    def fit(self, X, y=None):
        X = X.loc[:, ~X.columns.isin(['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise'])] #,'FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise'
        self.features = X.columns
        return self

    def get_feature_names_out (self):
        return self.features
    
class SelectPHFeatures ():
    def __init__ (self, X, y, data_info):
        self.x = X
        self.y = y
        self.dataset = data_info[0]
        self.cond = data_info[1]
        self.features = []

    def fit (self, X, y=None):
        if self.cond == 0:
            cond = "c1"
        elif self.cond == 1:
            cond = "c2"
        else:
            cond = "c3" 
        data_type = f"{self.dataset}_{cond}"
        
        exclusion_list = cfg.NON_PH_FTS[data_type]
        for feature in exclusion_list:
            X = X.drop(feature, axis=1)

        self.features = X.columns
        return self

    def get_feature_names_out (self):
        return self.features

def fit_and_score_features (X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha= 0.4)
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

class BaseFeatureSelector (ABC):
    """
    Base class for feature selectors.
    """
    def __init__ (self, X, y, estimator):
        """Initilizes inputs and targets variables."""
        self.X = X
        self.y = y
        self.estimator = estimator

    @abstractmethod
    def make_model (self):
        """
        """

    def get_features (self) -> List:
        ft_selector = self.make_model()
        if ft_selector.__class__.__name__ == "UMAP":
            self.fit(ft_selector, self.X)
            new_features = self.get_feature_names_out()
        else:
            ft_selector.fit(self.X, self.y)
            new_features = ft_selector.get_feature_names_out()
        return new_features

class NoneSelector (BaseFeatureSelector):
    def make_model (self):
        return SelectAllFeatures()
    
class PHSelector (BaseFeatureSelector):
    def make_model (self):
        return SelectPHFeatures(X=self.X, y=self.y, data_info=self.estimator)

class LowVar (BaseFeatureSelector):
    def make_model (self):
        return VarianceThreshold(threshold=0.001)

class SelectKBest4 (BaseFeatureSelector):
    def make_model (self):
        return SelectKBest(fit_and_score_features, k=4)

class SelectKBest8 (BaseFeatureSelector):
    def make_model (self):
        return SelectKBest(fit_and_score_features, k=8)