import pickle
from pathlib import Path
import pandas as pd
from utility.event import EventManager
import config as cfg

class FileReader:

    def __init__ (self, dataset, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = dataset

    def map_column_names(self, df):
        column_names_mapping = {}
        for old_col_name in df.columns:
            b_number = int(old_col_name.split('_')[0][1:])
            feature = old_col_name.split('_')[1]
            new_b_number = (b_number + 1) // 2
            new_col_name = f'B{new_b_number}_{feature}'
            column_names_mapping[old_col_name] = new_col_name
        return column_names_mapping

    def read_data_kaggle(self):
        """
        Deprecated Kaggle data extractor

        Returns:
        - set1: DataFrame containing data from 'set1_timefeatures.csv'
        - set2: DataFrame containing data from 'set2_timefeatures.csv'
        - set3: DataFrame containing data from 'set3_timefeatures.csv'
        """
        set1 = pd.read_csv("src/dataset/set1_timefeatures.csv")
        set2 = pd.read_csv("src/dataset/set2_timefeatures.csv")
        set3 = pd.read_csv("src/dataset/set3_timefeatures.csv")
        set1 = set1.rename(columns={'Unnamed: 0':'time'})
        set1.set_index('time')
        set2 = set2.rename(columns={'Unnamed: 0':'time'})
        set2.set_index('time')
        set3 = set3.rename(columns={'Unnamed: 0':'time'})
        set3.set_index('time')

        return [set1, set2, set3]
    
    def read_data(self, 
            test_condition: int, 
            axis: str, 
            from_pickle: bool = False
        ) -> (pd.DataFrame, pd.DataFrame, dict):
        dp = str(Path(self.dataset_path))
        covariates = pd.read_csv(dp + '/' + 'covariates_' + str(test_condition) + '.csv')
        analytic = pd.read_csv(dp + '/' + 'analytic_' + str(test_condition) + '.csv')
        if axis == "X":
            covariates_cols = [col for col in covariates.columns if int(col.split('_')[0][1:]) % 2 != 0]
            analytic_cols = [col for col in analytic.columns if int(col.split('_')[0][1:]) % 2 != 0]
            covariates_X = covariates[covariates_cols].copy(deep=True)
            analytic_X = analytic[analytic_cols].copy(deep=True)
            covariates_X.rename(columns=self.map_column_names(covariates_X), inplace=True)
            analytic_X.rename(columns=self.map_column_names(analytic_X), inplace=True)
        else:
            raise NotImplementedError()
        return covariates_X, analytic_X

    def read_pickle (self, 
        path: str
        ):

        file_handler = open(path, 'rb')
        obj = pickle.load(file_handler)
        file_handler.close()
        
        return obj
    