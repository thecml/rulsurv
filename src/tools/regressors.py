from abc import ABC, abstractmethod
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import config as cfg
from bnnsurv import models
from utility.mtlr import mtlr

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class BaseRegressor (ABC):
    """
    Base class for regressors.
    """
    def __init__ (self):
        """Initilizes inputs and targets variables."""

    @abstractmethod
    def make_model (self, params=None):
        """
        This method is an abstract method to be implemented
        by a concrete classifier. Must return a sklearn-compatible
        estimator object implementing 'fit'.
        """

    @abstractmethod
    def get_hyperparams (self):
        """Method"""

    @abstractmethod
    def get_hyperparams (self, condition):
        """Method"""

    def get_tuneable_params (self):
        return self.get_hyperparams()

    def get_best_params (self):
        return self.get_hyperparams()

    def get_estimator (self, params=None):
        return self.make_model(params)
    
class CoxPH(BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    """
    def get_hyperparams (self):
        return {'n_iter': [100000],
                'tol': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]}
    """
    def get_hyperparams (self, condition):
        if condition == 0:
            return {'tol': 1e-09, 'n_iter': 100}
        elif condition == 1:
            return {'tol': 1e-09, 'n_iter': 100}
        elif condition == 2:
            return {'tol': 1e-09, 'n_iter': 100}
        else:
            raise ValueError("Invalid condition for XJTU-SY dataset, choose {0, 1, 2}")

class CoxPHLasso(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_CPH_LASSO
        if params:
            model_params.update(params)
        return CoxnetSurvivalAnalysis(**model_params)
    """
    def get_hyperparams(self):
        return {
            'n_alphas': [25, 50, 100, 150, 200],
            'normalize': [False],
            'tol': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            'max_iter': [100000]
        }
    """
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'tol': 1e-05, 'normalize': False,
                    'n_alphas': 50, 'max_iter': 100000}
        elif condition == 1:
            return {'tol': 1e-05, 'normalize': False,
                    'n_alphas': 50, 'max_iter': 100000}
        elif condition == 2:
            return {'tol': 1e-05, 'normalize': False,
                    'n_alphas': 50, 'max_iter': 100000}
        else:
            raise ValueError("Invalid condition for XJTU-SY dataset, choose {0, 1, 2}")
                
class CoxBoost (BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_COXBOOST
        if params:
            model_params.update(params)
        return GradientBoostingSurvivalAnalysis(**model_params)
    """
    def get_hyperparams(self):
        return {'learning_rate': [0.1, 0.5, 1.0],
                'n_estimators': [100, 200, 400],
                'max_depth': [3, 5, 7],
                'min_samples_split': [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)],
                'min_samples_leaf': [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)],
                'max_features': [None, "log2", "sqrt"],
                'dropout_rate': [float(x) for x in np.linspace(0.0, 0.9, 10, endpoint=True)],
                'subsample': [float(x) for x in np.linspace(0.1, 1.0, 10, endpoint=True)]}
    """  
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'learning_rate': 0.1, 'n_estimators': 100, 'loss': 'coxph',
                    'dropout_rate': 0.4, 'max_features': 'log2', 'max_depth': 3,
                    'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 1.0}
        elif condition == 1:
            return {'learning_rate': 0.1, 'n_estimators': 200, 'loss': 'coxph',
                    'dropout_rate': 0.4, 'max_features': 'log2', 'max_depth': 5,
                    'min_samples_split': 10, 'min_samples_leaf': 5, 'subsample': 1.0}
        elif condition == 2:
            return {'learning_rate': 0.1, 'n_estimators': 400, 'loss': 'coxph',
                    'dropout_rate': 0.4, 'max_features': 'log2', 'max_depth': 7,
                    'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 1.0}
        else:
            raise ValueError("Invalid condition for XJTU-SY dataset, choose {0, 1, 2}")
        
class RSF(BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_RSF
        if params:
            model_params.update(params)
        return RandomSurvivalForest(**model_params)
    """
    def get_hyperparams (self):
        return {'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 400],
                'min_samples_split': [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)],
                'min_samples_leaf': [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]}
    """
    def get_hyperparams (self, n_condition):
        if n_condition == 0:
            return  {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 3,  'max_depth': 3}
        elif n_condition == 1:
            return  {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 5,  'max_depth': 5}
        elif n_condition == 2:
            return  {'n_estimators': 400, 'min_samples_split': 20, 'min_samples_leaf': 10,  'max_depth': 7}
        else:
            raise ValueError("Invalid condition for XJTU-SY dataset, choose {0, 1, 2}")
    
class MTLR(BaseRegressor):
    def make_model(self, num_features, num_time_bins, config):
        model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
        return model
    """
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'dropout' : [0.25, 0.5, 0.6],
                'lr' : [0.00008],
                'c1': [0.01],
                'num_epochs': [100, 500, 1000, 5000],
                'hidden_size': [32, 50, 60, 128]}
    """
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'batch_size': 32, 'dropout': 0.25, 'lr': 0.00008, 'c1': 0.01, 'num_epochs': 5000, 'hidden_size': 16}
        elif condition == 1:
            return {'batch_size': 64, 'dropout': 0.25, 'lr': 0.00008, 'c1': 0.01, 'num_epochs': 5000, 'hidden_size': 32}
        elif condition == 2:
            return {'batch_size': 128, 'dropout': 0.25, 'lr': 0.00008, 'c1': 0.01, 'num_epochs': 5000, 'hidden_size': 64}
        else:
            raise ValueError("Invalid condition for XJTU-SY dataset, choose {0, 1, 2}")

class DSM(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_DSM
        if params:
            model_params.update(params)
        return DeepSurvivalMachines(layers=model_params['layers'])
    """
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'learning_rate' : [1e-2, 5e-3, 1e-3],
                'iters': [100, 300, 500, 1000],
                'layers': [[16], [32], [64]]}
    """
    def get_hyperparams(self, condition):
        raise NotImplementedError()
    
class BNNSurv(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_BNN
        if params:
            model_params.update(params)
        return models.MCD(**model_params)
    """
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'learning_rate' : [1e-2, 5e-3, 1e-3],
                'num_epochs': [10, 15, 20],
                'layers': [[16], [32], [64], [128]]}
    """
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'batch_size' : 32, 'learning_rate' : 0.01, 'num_epochs': 100, 'layers': [16]}
        elif condition == 1:
            return {'batch_size' : 64, 'learning_rate' : 0.01, 'num_epochs': 100, 'layers': [32]}
        elif condition == 2:
            return {'batch_size' : 128, 'learning_rate' : 0.01, 'num_epochs': 100, 'layers': [64]}
        else:
            raise ValueError("Invalid condition for XJTU-SY dataset, choose {0, 1, 2}")