import os
import pandas as pd
import re
from tools.featuring import Featuring
import config as cfg
from pathlib import Path

class Builder:
    def __init__ (self, dataset, bootstrap):
        self.real_bearing = cfg.N_REAL_BEARING_XJTU
        boot_folder_size = (2 + bootstrap) * 2
        self.total_bearings = self.real_bearing * boot_folder_size
        self.raw_main_path = cfg.RAW_DATA_PATH_XJTU
        self.aggregate_main_path = cfg.DATASET_PATH_XJTU
        self.dataset = dataset 

    def build_new_dataset(self): 
        self.from_raw_to_csv()
        self.aggregate_and_refine()

    def from_raw_to_csv(self):

        """
        Create from raw data of two axis vibration data into timeseries data.
        The timeseries is structured to have a row of feature for each file of raw data and save into CSV format.

        Returns:
        - None
        """

        # For each type of test condition start to create the timeseries data
        for TYPE_TEST, group in enumerate(self.raw_main_path):
            bearing_channel_number = 1

            # For each real bearing create the timeseries data
            for bearing in range(1, self.real_bearing + 1, 1):
                dataset_path = str(Path(group)) + "/" + "Bearing1_" + str(bearing)
                datasets = Featuring.time_features_xjtu(dataset_path)

                # Create the label for each column and the filename for each dataset, hardcoded for the two axis vibration data bootstrap
                bearing_number = 1
                for dataset in datasets:
                    dataset.columns = ['B' + str(bearing_channel_number) + '_mean', 'B' + str(bearing_channel_number) + '_std', 'B' + str(bearing_channel_number) + '_skew', 'B' + str(bearing_channel_number) + '_kurtosis', 'B' + str(bearing_channel_number) + '_entropy',
                                       'B' + str(bearing_channel_number) + '_rms', 'B' + str(bearing_channel_number) + '_max', 'B' + str(bearing_channel_number) + '_p2p', 'B' + str(bearing_channel_number) + '_crest', 'B' + str(bearing_channel_number) + '_clearence',
                                       'B' + str(bearing_channel_number) + '_shape', 'B' + str(bearing_channel_number) + '_impulse', 'B' + str(bearing_channel_number) + '_FoH', 'B' + str(bearing_channel_number) + '_FiH', 'B' + str(bearing_channel_number) + '_FrH',
                                       'B' + str(bearing_channel_number) + '_FrpH', 'B' + str(bearing_channel_number) + '_FcaH', 'B' + str(bearing_channel_number) + '_Fo', 'B' + str(bearing_channel_number) + '_Fi', 'B' + str(bearing_channel_number) + '_Fr', 'B' + str(bearing_channel_number) + '_Frp', 'B' + str(bearing_channel_number) + '_Fca',
                                       'B' + str(bearing_channel_number) + '_noise', 'B' + str(bearing_channel_number) + '_Event', 'B' + str(bearing_channel_number) + '_Survival_time',
                                       'B' + str(bearing_channel_number + 1) + '_mean', 'B' + str(bearing_channel_number + 1) + '_std', 'B' + str(bearing_channel_number + 1) + '_skew', 'B' + str(bearing_channel_number + 1) + '_kurtosis', 'B' + str(bearing_channel_number + 1) + '_entropy',
                                       'B' + str(bearing_channel_number + 1) + '_rms', 'B' + str(bearing_channel_number + 1) + '_max', 'B' + str(bearing_channel_number + 1) + '_p2p', 'B' + str(bearing_channel_number + 1) + '_crest', 'B' + str(bearing_channel_number + 1) + '_clearence',
                                       'B' + str(bearing_channel_number + 1) + '_shape', 'B' + str(bearing_channel_number + 1) + '_impulse', 'B' + str(bearing_channel_number + 1) + '_FoH', 'B' + str(bearing_channel_number + 1) + '_FiH', 'B' + str(bearing_channel_number + 1) + '_FrH',
                                       'B' + str(bearing_channel_number + 1) + '_FrpH', 'B' + str(bearing_channel_number + 1) + '_FcaH', 'B' + str(bearing_channel_number + 1) + '_Fo', 'B' + str(bearing_channel_number + 1) + '_Fi', 'B' + str(bearing_channel_number + 1) + '_Fr', 'B' + str(bearing_channel_number + 1) + '_Frp', 'B' + str(bearing_channel_number + 1) + '_Fca',
                                       'B' + str(bearing_channel_number + 1) + '_noise', 'B' + str(bearing_channel_number + 1) + '_Event', 'B' + str(bearing_channel_number + 1) + '_Survival_time']
                    
                    dataname = str(Path(self.aggregate_main_path)) + "/" + "Bearing1_" + str(bearing) + "_" + str(bearing_number) + "_timefeature" + "_" + str(TYPE_TEST) + ".csv"
                    if not os.path.exists(self.aggregate_main_path):
                        os.mkdir(self.aggregate_main_path)
                    dataset.to_csv(dataname, index=False)
                    bearing_number += 1
                    bearing_channel_number += 2
                
    def aggregate_and_refine (self):

        """
        Final step that aggregate and refine the data for event detection and survival analysis after being processed from prerocessed CSV files.
        It saves the aggregated data in separate CSV files for each type of data and information.

        Returns:
        - None
        """

        # Data used for the event detector
        set_analytic = pd.DataFrame()
        # Covariates used for survival analysis
        set_covariates = pd.DataFrame()

        # For each type of test condition, aggregate the data
        for TYPE_TEST, group in enumerate(self.raw_main_path):

            # For each CSV file of timeseries data, using regex condense the information into a single dataframe
            for filename in os.listdir(self.aggregate_main_path):
                if re.search('^Bearing.*timefeature_' + str(TYPE_TEST), filename):
                    #From the dataframe of the csv file, select the columns of interest like time features, frequency features
                    datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename))
                    set_analytic_aux = datafile.iloc[:,12:17] # 5 bands
                    set_cov_aux = datafile.iloc[:,:12] # 12 features
                    set_analytic = pd.concat([set_analytic, set_analytic_aux], axis=1)
                    set_covariates = pd.concat([set_covariates, set_cov_aux], axis=1)

                    set_analytic_aux = datafile.iloc[:,37: 42] # 5 bands
                    set_cov_aux = datafile.iloc[:,25:25+12] # 12 features
                    set_analytic = pd.concat([set_analytic, set_analytic_aux], axis=1)
                    set_covariates = pd.concat([set_covariates, set_cov_aux], axis=1)
                    
            # Save the aggregated data in separate CSV files
            set_analytic.to_csv(str(Path(self.aggregate_main_path)) + "/" + 'analytic_' + str(TYPE_TEST) + '.csv', index=False)
            set_covariates.to_csv(str(Path(self.aggregate_main_path)) + "/" + 'covariates_' + str(TYPE_TEST) + '.csv', index=False)
        
            # Clean the variables for the next batch of data
            set_analytic = pd.DataFrame()
            set_covariates = pd.DataFrame()
