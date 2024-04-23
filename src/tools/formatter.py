import pandas as pd
import numpy as np
import statistics
import random
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import config as cfg

class Formatter:

    def __init__ (self):
        pass

    @staticmethod
    def format_main_data_Kfold (
            T1: tuple, 
            train: list, 
            test: list
        ) -> (tuple, tuple, tuple, tuple):

        """
        Formats the main data for K-fold cross-validation.

        Args:
        - T1 (tuple): A tuple containing the main data and labels.
        - train (list): List of indices for the training data.
        - test (list): List of indices for the test data.

        Returns:
        - ti, cvi, ti_NN, cvi_NN (tuple): A tuple containing the formatted training and test data for both the main data and labels.
        """

        #Extract the y as labels
        ti_y_df= T1[0].iloc[train, -2:]
        cvi_y_df= T1[0].iloc[test, -2:]

        #Change the name of the labels
        ti_y_df.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)
        cvi_y_df.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)

        #Create the test/train data
        ti_y_df.event = ti_y_df.event.replace({True: 1, False: 0})
        cvi_y_df.event = cvi_y_df.event.replace({True: 1, False: 0})
        ti_X = T1[0].iloc[train, :-2]
        ti_y = T1[1][train]
        cvi_X = T1[0].iloc[test, :-2]
        cvi_y = T1[1][test]
        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)
        ti_y_df.reset_index(inplace= True, drop=True)
        cvi_y_df.reset_index(inplace= True, drop=True)

        # Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)
        ti_NN = (ti_X, ti_y_df)
        cvi_NN = (cvi_X, cvi_y_df)

        return ti, cvi, ti_NN, cvi_NN

    @staticmethod
    def format_main_data (
            T1: tuple, 
            T2: tuple
        ) -> (tuple, tuple, tuple, tuple):

        """
        Formats the main data for training and testing.

        Args:
        - T1 (tuple): Training data tuple containing X_train and y_train.
        - T2 (tuple): Testing data tuple containing X_test and y_test.

        Returns:
        - X_tr, X_te, y_tr_NN, y_te_NN (tuple): A tuple containing formatted training and testing data.

        """

        #Extract the y as labels
        y_train_NN = T1[0].iloc[:, -2:]
        y_test_NN = T2[0].iloc[:, -2:]

        #Change the name of the labels
        y_train_NN.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)
        y_test_NN.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)
        
        #Create the test/train data
        y_train_NN.event = y_train_NN.event.replace({True: 1, False: 0})
        y_test_NN.event = y_test_NN.event.replace({True: 1, False: 0})
        X_train = T1[0].iloc[:, :-2]
        y_train = T1[1]
        X_test = T2[0].iloc[:, :-2]
        y_test = T2[1]
        y_train_NN.reset_index(inplace= True, drop=True)
        y_test_NN.reset_index(inplace= True, drop=True)
        X_train.reset_index(inplace= True, drop=True)
        X_test.reset_index(inplace= True, drop=True)

        # Collect splits
        X_tr = (X_train, y_train)
        X_te = (X_test, y_test)
        y_tr_NN = (X_train, y_train_NN)
        y_te_NN = (X_test, y_test_NN)

        return X_tr, X_te, y_tr_NN, y_te_NN
    
    @staticmethod
    def centering_data (
            ti: tuple, 
            cvi: tuple
        ) -> (tuple, tuple):

        """
        Center the train and test data using standard scaling.

        Parameters:
        - ti (tuple): Tuple containing the train data (X, y).
        - cvi (tuple): Tuple containing the test data (X, y).

        Returns:
        - ti, cv (tuple): Tuple containing the centered train data (X, y) and centered test data (X, y).
        """

        #Change name of the train/test data
        ti_X = ti[0]
        ti_y = ti[1]
        cvi_X = cvi[0]
        cvi_y = cvi[1]
        features = list(ti_X.columns)

        #Choose scaling
        scaler = StandardScaler()

        #Apply scaling
        scaler.fit(ti_X)
        ti_X = pd.DataFrame(scaler.transform(ti_X), columns=features)
        cvi_X = pd.DataFrame(scaler.transform(cvi_X), columns=features)
        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)

        #Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)

        return ti, cvi
    
    @staticmethod
    def centering_main_data (
            ti: tuple, 
            cvi: tuple, 
            ti_NN: tuple, 
            cvi_NN: tuple
        ) -> (tuple, tuple, tuple, tuple):

        """
        Center the main data by applying scaling using StandardScaler.

        Parameters:
        - ti (tuple): Tuple containing train data X and y.
        - cvi (tuple): Tuple containing cross-validation data X and y.
        - ti_NN (tuple): Tuple containing train data for neural network X and y.
        - cvi_NN (tuple): Tuple containing cross-validation data for neural network X and y.

        Returns:
        - ti (tuple): Tuple containing centered train data X and y.
        - cvi (tuple): Tuple containing centered cross-validation data X and y.
        - ti_NN (tuple): Tuple containing centered train data for neural network X and y.
        - cvi_NN (tuple): Tuple containing centered cross-validation data for neural network X and y.
        """

        #Change name of the train/test data
        ti_X = ti[0]
        ti_y = ti[1]
        cvi_X = cvi[0]
        cvi_y = cvi[1]
        ti_X_NN = ti_NN[0]
        ti_y_NN = ti_NN[1]
        cvi_X_NN = cvi_NN[0]
        cvi_y_NN = cvi_NN[1]
        features = list(ti_X.columns)

        #Choose scaling
        scaler = StandardScaler()
        
        #Apply scaling
        scaler.fit(ti_X)
        ti_X = pd.DataFrame(scaler.transform(ti_X), columns=features)
        cvi_X = pd.DataFrame(scaler.transform(cvi_X), columns=features)
        ti_X_NN = pd.DataFrame(scaler.transform(ti_X_NN), columns=features)
        cvi_X_NN = pd.DataFrame(scaler.transform(cvi_X_NN), columns=features)
        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)
        ti_X_NN.reset_index(inplace= True, drop=True)
        cvi_X_NN.reset_index(inplace= True, drop=True)

        #Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)
        ti_NN = (ti_X_NN, ti_y_NN)
        cvi_NN = (cvi_X_NN, cvi_y_NN)

        return ti, cvi, ti_NN, cvi_NN
    
    @staticmethod
    def format_centering_NN_data (
            T1NN: tuple, 
            T2NN: tuple, 
            ti_y_df: tuple, 
            cvi_y_df: tuple, 
            TvalNN: tuple
        ) -> (tuple, tuple, tuple, tuple, tuple, tuple):

        """
        Formats and preprocesses the data for training a neural network model.

        Args:
        - T1NN (tuple): Tuple containing the training data for T1NN.
        - T2NN (tuple): Tuple containing the training data for T2NN.
        - ti_y_df (pandas.DataFrame): DataFrame containing the target variable for T1NN.
        - cvi_y_df (pandas.DataFrame): DataFrame containing the target variable for T2NN.
        - TvalNN (tuple): Tuple containing the validation data.

        Returns:
        tuple: A tuple containing the formatted and preprocessed data for training the neural network model.
            - ti_NN (numpy.ndarray): Formatted training data for T1NN.
            - y_ti_NN (tuple): Tuple containing the target variable for T1NN.
            - cvi_NN (numpy.ndarray): Formatted training data for T2NN.
            - durations_test (numpy.ndarray): Array containing the durations for the test data.
            - events_test (numpy.ndarray): Array containing the events for the test data.
            - val_NN (tuple): Tuple containing the formatted validation data and target variable.
        """

        #Change name of the train/test data
        ti_X_NN = pd.concat([T1NN[0], ti_y_df], axis=1)
        cvi_X_NN = pd.concat([T2NN[0], cvi_y_df], axis=1)
        ti_X_val_NN= TvalNN[0]
        features = T1NN[0].columns
        cols_standardize = list(features)
        cols_leave = []
        
        #Choose scaling
        scaler= StandardScaler()

        #Apply scaling
        standardize = [([col], scaler) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_train_ti = x_mapper.fit_transform(ti_X_NN).astype('float32')
        x_train_cvi = x_mapper.fit_transform(cvi_X_NN).astype('float32')       
        x_val = x_mapper.transform(ti_X_val_NN).astype('float32')
        get_target = lambda df: (df['Survival_time'].values, df['Event'].values)
        y_ti_NN = get_target(ti_X_NN)
        y_val = get_target(ti_X_val_NN)
        durations_test, events_test = get_target(cvi_X_NN)       
        
        #Change name of the train/test data
        ti_NN = x_train_ti
        cvi_NN = x_train_cvi
        val_NN= x_val, y_val

        return ti_NN , y_ti_NN, cvi_NN, durations_test, events_test, val_NN

    @staticmethod
    def add_random_censoring (X: pd.DataFrame, pct: float) -> pd.DataFrame:
        samples_to_censor = X.sample(frac=pct, random_state=0)
        df_rest = X.loc[~X.index.isin(samples_to_censor.index)]
        samples_to_censor['Survival_time'] = samples_to_censor.apply(
            lambda x: np.random.randint(0, x['Survival_time']), axis=1)
        samples_to_censor.loc[samples_to_censor['Survival_time'].eq(0),
                              'Survival_time'] = 1 # avoid zero time
        samples_to_censor['Event'] = False
        new_dataset = pd.concat([samples_to_censor, df_rest])
        new_dataset = new_dataset.reset_index(drop=True)
        return new_dataset
        
    @staticmethod
    def control_censored_data (
            X: pd.DataFrame, 
            percentage: float
        ) -> pd.DataFrame:

        """
        Adjusts the censored data in the given DataFrame to match the desired percentage.

        Args:
        - X (pandas.DataFrame): The input DataFrame containing the data.
        - percentage (float): The desired percentage of censored data.

        Returns:
        - X (pandas.DataFrame): The modified DataFrame with adjusted censored data.
        """

        # Take the indexes about actual censored data status
        censored_actual_idx = X.loc[X['Event'] == 0].index
        not_censored_actual_idx = X.loc[X['Event'] == 1].index
        
        # Take the information about actual censored data status
        num_censored_actual = len(X.iloc[censored_actual_idx])       
        num_censored_required = int(np.floor(len(X) * percentage))
        
        # If needed less censored data
        if num_censored_actual > num_censored_required:
            censored_indexes = np.random.choice(censored_actual_idx, size=num_censored_actual - num_censored_required, replace=False)
            X.loc[censored_indexes, "Event"] = True

        # If needed more censored data
        elif num_censored_actual < num_censored_required:
            not_censored_indexes = np.random.choice(not_censored_actual_idx, size=num_censored_required - num_censored_actual, replace=False)
            X.loc[not_censored_indexes, "Event"] = False
            # for index in not_censored_indexes:
            #     rnd = random.randint(1, 3)
            #     X.loc[index, "Survival_time"] = X.loc[index, "Survival_time"] / rnd         
        X.reset_index(drop=True)

        return X

    @staticmethod
    def calculate_positions_percentages (
            dataframe: pd.DataFrame, 
            column_name: str, 
            values: list
        ) -> list:

        """
        Calculate the positions of values in a DataFrame column as percentages.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column.
        - column_name (str): The name of the numerical column.
        - values (list): List of numerical values for which you want to calculate the positions.

        Returns:
        - positions_percentages (list): List of positions of the values in percentages.
        """
        if not values:
            return []
        
        #Take the columns name and sort the column in ascending order
        column = dataframe[column_name].tolist()
        column.sort()  
        total_values = len(column)

        positions_percentages = []
        
        #Find the value end manage special cases
        for value in values:
            if value < column[0]:
                position_percent = 0.0
            elif value > column[-1]:
                position_percent = 100.0
            else:
                # Find the index where the value would be inserted to maintain the order
                index = 0
                while index < total_values and column[index] < value:
                    index += 1
                
                # Calculate the position in percentage based on index and total values
                position_percent = (index / total_values) * 100
            
            positions_percentages.append(position_percent)
        
        return positions_percentages
    
    @staticmethod
    def find_values_by_percentages (
            dataframe: pd.DataFrame, 
            column_name: str, 
            target_percentages: list
        ) -> list:

        """
        Find values in a DataFrame column based on a list of target percentage values within the column.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column.
        - column_name (str): The name of the numerical column.
        - target_percentages (list): List of target percentage values.

        Returns:
        - values_by_percentages (list): List of values from the column corresponding to the target percentage values.
        """

        if not target_percentages:
            return []
        
        #Take the columns name and sort the column in ascending order
        column = dataframe[column_name].tolist()
        column.sort()  
        total_values = len(column)
        
        values_by_percentages = []
        
        #Find the point in the dataframe
        for target_percentage in target_percentages:
            target_position = int((target_percentage / 100) * total_values)
            values_by_percentages.append(column[target_position])
        
        return values_by_percentages
    
    def transform_column_with_cutpoints (
            dataset: pd.DataFrame, 
            cut_points: list, 
            column_name: str
        ) -> pd.DataFrame:

        new_df = dataset.copy()
        for cut_point in cut_points:
            new_column_name = f"{column_name}_<=_{cut_point}"
            new_df[new_column_name] = dataset[column_name] >= cut_point
        
        return new_df