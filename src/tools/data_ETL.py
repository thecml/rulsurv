import pandas as pd
import numpy as np
import statistics
import random
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import config as cfg

class DataETL:

    def __init__ (self, dataset, bootstrap=0):
        # 2 real bearings from x and y channel + 2 bootstrapped bearings from x and y channel multiplied by the bootstrap value
        self.fixed_time_split = 1
        self.boot_folder_size = (bootstrap * 2) + 2
        self.lag = 1
        self.post = 2
        self.real_bearings = cfg.N_REAL_BEARING_XJTU
        self.total_bearings = self.real_bearings * self.boot_folder_size
        self.total_signals = cfg.N_SIGNALS_XJTU
        self.folder_size = self.total_bearings * self.fixed_time_split

    def make_moving_average(self, timeseries_data: pd.DataFrame, event_time: int,
                            bearing_id: int, window_size: int, lag: int):
        bearing_cols = [col for col in timeseries_data if col.startswith(f'B{bearing_id}_')]
        df = timeseries_data.loc[:,bearing_cols].dropna()
        df.columns = df.columns.str.replace(r'^B\d+_','', regex=True)
        df = df.loc[:event_time, :] # select data up to event
        df['Survival_time'] = range(int(event_time)+1, 0, -1) # annotate the event
        cols = list(df.columns)
        total_df = pd.DataFrame()
        for ft_col in cols: # Compute moving average
            roll = df[ft_col].rolling(window_size)
            ma = roll.mean().shift(lag).reset_index(0, drop=True)
            if ft_col == "Survival_time":
                total_df[ft_col] = ma.transform(np.floor)
            else:
                total_df[ft_col] = ma
        total_df = total_df.dropna().reset_index(drop=True)
        total_df['Event'] = True
        total_df['Survival_time'] = total_df['Survival_time'].astype(int)
        total_df['Event'] = total_df['Event'].astype(bool)
        return total_df
                
    def make_surv_data_bootstrap (self,
            covariates: dict, 
            set_boot: pd.DataFrame, 
            info_pack: dict, 
            bootstrap: int
        ) -> (list, dict):

        """
        Transform the timeseries data into survival data.
        Prepare the survival data only for bootstrap upsampling strategy and avoid information leaking train/test.

        Args:
        - covariates (dict): Dictionary containing the covariates data.
        - set_boot (DataFrame): Number of bootstrap sets.
        - info_pack (dict): Dictionary containing information about the time-to-event.
        - bootstrap (int): The value of bootstrapped multiplier.

        Returns:
        - survival_covariates (list): List of dataframes containing the upsampled data for each group of bearings.
        - reference_value_TtE_abs (dict): Dictionary containing the reference values for each bearing of the event time with absolute values of time-to-event.
        """

        #Prepare the empty data and folders of grouped bearings to fill for avoid leaking
        row = pd.DataFrame()
        reference_value_TtE = {}
        survival_covariates = []
        type_foldering = []
        
        # Prepare the structured empty data and for grouped bearings to fill for avoid leaking
        survival_covariates, type_foldering = self.prepare_data()

        # Extract the info about the time-to-event
        for bear_num in range (1, self.total_bearings + 1, self.boot_folder_size):
            val = self.event_analyzer (bear_num, info_pack)
            reference_value_TtE.update({bear_num : val})

        # For each covariates, event or survival time take the all the time values in a column 
        for column in covariates:
            columnSeriesObj = covariates[column]
            columnSeriesObj = columnSeriesObj.dropna()

            bear_num = int(re.findall("\d?\d?\d", column)[0])
            temp_label_cov= ""
            
            # Assign the label to the covariate or take the event or survival time value
            temp_label_cov = re.split("\d_", column)[1]

            if re.findall(r"Event\b", column):
                reference_bearing_num = self.select_reference(bear_num, self.boot_folder_size)
                
                # If the event is negative means that event detector has no detection and need to force the censoring status
                if reference_value_TtE[reference_bearing_num] < 0:        
                    columnSeriesObj = self.event_manager(bear_num, bootstrap, force_censoring = True)
                else:
                    columnSeriesObj = self.event_manager(bear_num, bootstrap, force_censoring = False)
            elif re.findall(r"Survival_time\b", column):
                reference_bearing_num = self.select_reference(bear_num, self.boot_folder_size)
                
                # After use the information about censoring transform the negative value of TtE into positive for the survival data requirements
                if reference_value_TtE[reference_bearing_num] < 0:
                    reference_value_TtE_abs = reference_value_TtE
                    reference_value_TtE_abs[reference_bearing_num] = abs(reference_value_TtE[reference_bearing_num])
                    columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
                else:
                    columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE)
                    reference_value_TtE_abs = reference_value_TtE
            
            # Actual information in subject that convert timeseries into survival data
            label= temp_label_cov
            
            # If survival time or event type take the values as is, instead for the others take the mean of time-series             
            if label == "Event" or label == "Survival_time":
                row [label]= pd.Series(columnSeriesObj).T   
            else:
                row [label]= pd.Series(np.mean(columnSeriesObj.values)).T

            # If survival time column close the row for survival data and go to the next
            if label == "Survival_time":
                for FOLDER, bearings_in_folder in enumerate(type_foldering):
                    if bear_num in bearings_in_folder:
                        survival_covariates[FOLDER] = pd.concat([survival_covariates[FOLDER], row], ignore_index= True) 

        return survival_covariates, reference_value_TtE
            
    def make_surv_data_transform_ma(self, 
            covariates: dict, 
            set_boot: pd.DataFrame, 
            info_pack: dict, 
            bootstrap: int, 
            type_correlation: str
        ) -> (list, dict):

        """
        Transform the timeseries data into survival data using boostrap and transform MA methododology.
        Upsamples the data from given time split and prepares the empty data and folders of grouped bearings to fill for avoid information leaking train/test.

        Parameters:
        - covariates (dict): Dictionary containing the covariates data.
        - set_boot (pd.DataFrame): Number of bootstrap sets.
        - info_pack (dict): Dictionary containing the information about the time-to-event.
        - bootstrap (int): The value of bootstrapped multiplier.
        - type_correlation (str): Type of data correlation.

        Returns:
        - survival_covariates (list): List of dataframes containing the upsampled data for each group of bearings.
        - reference_value_TtE_abs (dict): Dictionary containing the reference values for each bearing of the event time with absolute values of time-to-event.
        """

        #Prepare the empty data
        row = pd.DataFrame()
        reference_value_TtE = {}
        
        # Prepare the structured empty data and for grouped bearings to fill for avoid leaking
        survival_covariates, type_foldering = self.prepare_data()

        # Extract the info about the time-to-event
        for bear_num in range (1, self.total_bearings + 1, self.boot_folder_size):
            val = self.event_analyzer (bear_num, info_pack)
            reference_value_TtE.update({bear_num : val})
        
        # Set up the time slices
        moving_window = 0
        time_split = self.lag + self.post
        time_moving = self.fixed_time_split

        # Upsample the data from given time split
        while moving_window < time_moving:
        
            # For each covariates, event or survival time take the all the time values in a column 
            for column in covariates:
                columnSeriesObj = covariates[column]
                columnSeriesObj= columnSeriesObj.dropna()

                bear_num = int(re.findall("\d?\d?\d", column)[0])
                temp_label_cov= ""                    
                
                # Setup the time window size from the information of the time-to-event
                timepoints = math.floor(abs(self.survival_time_manager(bear_num, set_boot, reference_value_TtE)))
                if time_moving < timepoints:
                    time_moving = timepoints

                # If the end of the lifetime is reached, stop to slice the time-series for this bearing
                if timepoints - 1 < moving_window:
                    continue

                # Transform the time window size into bounduaries for slicing and manage particular cases                    
                low = moving_window - self.lag
                if low < 0:
                    low = 0
                high = self.post + moving_window
                if high > timepoints:
                    high = timepoints 

                # Take the second part that contain the name of the covariate from the column name as label
                temp_label_cov = re.split("\d_", column)[1]

                if re.findall(r"Event\b", column):
                    reference_bearing_num = self.select_reference(bear_num, self.boot_folder_size)
                    
                    # If the event is negative means that event detector has no detection and need to force the censoring status
                    if reference_value_TtE[reference_bearing_num] < 0:        
                        columnSeriesObj = self.event_manager(bear_num, bootstrap, force_censoring = True)
                    else:
                        columnSeriesObj = self.event_manager(bear_num, bootstrap, force_censoring = False)
                        
                elif re.findall(r"Survival_time\b", column):
                    reference_bearing_num = self.select_reference(bear_num, self.boot_folder_size)
                    
                    # After use the information about censoring transform the negative value of TtE into positive for the survival data requirements
                    if reference_value_TtE[reference_bearing_num] < 0:
                        reference_value_TtE_abs = reference_value_TtE
                        reference_value_TtE_abs[reference_bearing_num] = abs(reference_value_TtE[reference_bearing_num])
                        columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
                    else:
                        columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE)
                        reference_value_TtE_abs = reference_value_TtE
                
                # Actual information in subject that convert timeseries into survival data
                label= temp_label_cov
                
                # If survival time or event type start to slice the time-series, instead for the others take the mean of time-series                     
                if label == "Event" or label == "Survival_time":
                    if label == "Survival_time":
                        
                        # At the last window take the event within the first slice, otherwise take the mean of the lasts windows near the event 
                        if timepoints - 2 < moving_window:
                            proportional_value= np.mean([0, self.post])                                     
                        else:
                            proportional_value= columnSeriesObj - np.mean([low, high])

                        row [label] = pd.Series(proportional_value).T 
                    else:
                        row [label] = pd.Series(columnSeriesObj).T   
                else:
                    # Take the covariates slice near to the event or the last                                 
                    time_slice = columnSeriesObj.iloc[low:high]
                    
                    # Average the covariates value to transform time-series to survival data
                    row [label] = pd.Series(np.mean(time_slice.values)).T
                
                # If survival time column close the row and save it into a proper folder for avoid leaking
                if label == "Survival_time":
                    for FOLDER, bearings_in_folder in enumerate(type_foldering):
                        if bear_num in bearings_in_folder:
                            survival_covariates[FOLDER] = pd.concat([survival_covariates[FOLDER], row], ignore_index= True)

            # Go for next window                
            moving_window += 1  
                  
        return survival_covariates, reference_value_TtE_abs

    def make_surv_data_transform_ama(self, 
            covariates: dict, 
            set_boot: pd.DataFrame, 
            info_pack: dict, 
            bootstrap: int, 
            type_correlation: str
        ) -> (list, dict):

        """
        Transform the timeseries data into survival data using boostrap and transform AMA methododology.
        Upsamples the data from given time split and prepares the empty data and folders of grouped bearings to fill for avoid information leaking train/test.

        Parameters:
        - covariates (dict): Dictionary containing the covariates data.
        - set_boot (pd.DataFrame): Number of bootstrap sets.
        - info_pack (dict): Dictionary containing the information about the time-to-event.
        - bootstrap (int): The value of bootstrapped multiplier.
        - type_correlation (str): Type of data correlation.

        Returns:
        - survival_covariates (list): List of dataframes containing the upsampled data for each group of bearings.
        - reference_value_TtE_abs (dict): Dictionary containing the reference values for each bearing of the event time with absolute values of time-to-event.
        """

        #Prepare the empty data
        row = pd.DataFrame()
        reference_value_TtE = {}
        
        # Prepare the structured empty data and for grouped bearings to fill for avoid leaking
        survival_covariates, type_foldering = self.prepare_data()

        # Extract the info about the time-to-event
        for bear_num in range (1, self.total_bearings + 1, self.boot_folder_size):
            val = self.event_analyzer (bear_num, info_pack)
            reference_value_TtE.update({bear_num : val})
        
        # Set up the time slices
        moving_window = 0
        time_split = self.fixed_time_split

        # Upsample the data from given time split
        while moving_window < time_split:
        
            # For each covariates, event or survival time take the all the time values in a column 
            for column in covariates:
                columnSeriesObj = covariates[column]
                columnSeriesObj= columnSeriesObj.dropna()

                bear_num = int(re.findall("\d?\d?\d", column)[0])
                temp_label_cov= ""                    
                
                # Setup the time window size from the information of the time-to-event
                timepoints = math.floor(abs(self.survival_time_manager(bear_num, set_boot, reference_value_TtE)))
                if time_split < timepoints:
                    time_split = timepoints

                # If the end of the lifetime is reached, stop to slice the time-series for this bearing
                if timepoints - 1 < moving_window:
                    continue

                # Take the second part that contain the name of the covariate from the column name as label
                temp_label_cov = re.split("\d_", column)[1]

                if re.findall(r"Event\b", column):
                    reference_bearing_num = self.select_reference(bear_num, self.boot_folder_size)
                    
                    # If the event is negative means that event detector has no detection and need to force the censoring status
                    if reference_value_TtE[reference_bearing_num] < 0:        
                        columnSeriesObj = self.event_manager(bear_num, bootstrap, force_censoring = True)
                    else:
                        columnSeriesObj = self.event_manager(bear_num, bootstrap, force_censoring = False)
                elif re.findall(r"Survival_time\b", column):
                    reference_bearing_num = self.select_reference(bear_num, self.boot_folder_size)
                    
                    # After use the information about censoring transform the negative value of TtE into positive for the survival data requirements
                    if reference_value_TtE[reference_bearing_num] < 0:
                        reference_value_TtE_abs = reference_value_TtE
                        reference_value_TtE_abs[reference_bearing_num] = abs(reference_value_TtE[reference_bearing_num])
                        columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
                    else:
                        columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE)
                        reference_value_TtE_abs = reference_value_TtE
                
                # Actual information in subject that convert timeseries into survival data
                label = temp_label_cov
                
                # If survival time or event type start to slice the time-series, instead for the others take the mean of time-series                     
                if label == "Event" or label == "Survival_time":
                    if label == "Survival_time":

                        # For the last window take the time-to-event as is, otherwise take value of the moving window
                        if timepoints - 2 < moving_window:
                            proportional_value = columnSeriesObj                                                                       
                        else:
                            proportional_value = moving_window
                        row [label] = pd.Series(proportional_value).T 
                    else:
                        row [label] = pd.Series(columnSeriesObj).T   
                else:
                    # For the last window take the covariates as is, otherwise take lasts time slices        
                    if timepoints - 2 < moving_window: 
                        time_slice = columnSeriesObj
                    else:
                        time_slice = columnSeriesObj.iloc[- (self.fixed_time_split * (moving_window + 1)):] 

                    # Average the covariates value to transform time-series to survival data
                    row [label] = pd.Series(np.mean(time_slice.values)).T
                
                # If survival time column close the row and save it into a proper folder for avoid leaking
                if label == "Survival_time":
                    for FOLDER, bearings_in_folder in enumerate(type_foldering):
                        if bear_num in bearings_in_folder:
                            survival_covariates[FOLDER] = pd.concat([survival_covariates[FOLDER], row], ignore_index= True)

            # Go for next window                
            moving_window += 1  
                  
        return survival_covariates, reference_value_TtE_abs

    def event_manager(self, 
            num: int, 
            bootstrap: int,  
            force_censoring: bool
        ) -> (bool):
        """
        Determines whether an event is censored or not

        Parameters:
        - num (int): The current bearing number.
        - bootstrap (int): The value of bootstrapped multiplier.
        - force_censoring (bool): Flag indicating whether to force censoring or not.

        Returns:
        - (bool): True if the event is not censored, False otherwise.
        """
        
        # The event is not determined by KL and SD so it is censored
        if force_censoring == True:
            return False

        # Only the last bootstrapped bearing will be censored        
        censor_pointer = self.boot_folder_size
        total_number_bearings = self.total_bearings
        if bootstrap > 0:  
            for check in range(censor_pointer, total_number_bearings + 1, self.boot_folder_size):
                if check == num: 
                    return False
                else:
                    return True
        
        # If no bootstrapping will be always not censored
        else:
            return True

    def survival_time_manager (self, 
            num: int, 
            bootref: pd.DataFrame, 
            ref: dict
        ) -> (float):

        """
        Calculates the survival time for a given bearing number adding a last randomizer from the special reference bearing.

        Parameters:
        - num (int): The bearing number.
        - bootref (DataFrame): The bootstrapped reference values.
        - ref (dict): The reference values for special bearings.

        Returns:
        - (float): The calculated survival time for the given bearing number.
        """

        # Prepare random value for the special reference bearing
        for key, value in ref.items():
            if key == num or key + 1 == num:
                return value  
        
        # The size of a batch of bootstrapped sample from a single bearing
        batch_bootstrap_size = self.boot_folder_size
       
        # Bootstrapping for the other normal bootstrapped/upsampled bearings
        for BEARING, batch_level in enumerate(range(batch_bootstrap_size, self.total_bearings + 1, batch_bootstrap_size)): 
            
            # Transform the real number of the bearing into a index of a bootstrap batch
            if num >= batch_bootstrap_size:
                relative_num = num - batch_level
            else:
                # The index substract the number 2 of the real bearing and 1 to adapt to an pandas index
                relative_num = num -2 -1

            # Take the value from the batch of bootstrapped and apply the normal randomized value
            if num <= batch_level and num > batch_level - batch_bootstrap_size:
                return bootref.iat[relative_num,BEARING] + ref[batch_level - batch_bootstrap_size + 1]         

    def event_analyzer(self, 
        bear_num: int, 
        info_pack: dict
        ) -> (float):
        """
        Analyzes the event data for a specific bearing.

        Parameters:
        - bear_num (int): The number of the bearing.
        - info_pack (dict): A dictionary containing information about time-to-event for each bearing.

        Returns:
        - result (float): The analyzed event data.
        """
        
        # Transform the bearing number into the index of the info_pack
        bear_num -= 1 

        data_kl = []
        data_sd = []

        # From info pack take the information KL and SD about time-to-event for each bearing
        for INFO_TYPE in info_pack:
            for bear_info in info_pack[INFO_TYPE][bear_num]:
                # Multiply by 10 for scaling
                cross = bear_info[2] * 10
                # Multiply by 10 for scaling
                tot_length = bear_info[3] * 10
                
                # Do not record the event if the time-to-event is 0 because there is not a breakpoint or not determined a breakline
                if INFO_TYPE == "KL" and cross != 0.0:
                    data_kl.append(cross)
                    
                # Do not record the event if the time-to-event is 0 because there is not a breakpoint or not determined a breakline
                if INFO_TYPE == "SD" and cross != 0.0:
                    data_sd.append(cross)

        # Assign a time-to-event if exist from KL or SD or end of recording if necessary
        if not data_sd:
            # if data_kl:
            #     data_sd = data_kl
            # else:
            data_sd = [-tot_length]
            print("For bearing #{}, event considered at the end of the recording".format(bear_num))
        if not data_kl:
            if data_sd:
                data_kl = data_sd

        # #Take the maximum from the evaluations
        # res = [max(data_kl), max(data_sd)]
        # res = round(statistics.mean(res), 1)

        # Take the SD as first choice and KL as second choice
        if self.event_detector_goal == "predictive_maintenance_sensitive":
            result = round(data_sd[0], 1)
        elif self.event_detector_goal == "predictive_maintenance_robust":
            if len(data_sd) > 1:
                result = round(data_sd[1], 1)
            else:
                print("For bearing #{}, event considered at the end of the recording".format(bear_num))
                result = -tot_length
        elif self.event_detector_goal == "labeler_median":
            result = round(np.median(data_sd), 1)
        elif self.event_detector_goal == "labeler_mean":
            result= round(np.mean(data_sd), 1)
        elif self.event_detector_goal == "labeler_max":
            result = round(np.max(data_sd), 1)

        return result

    def select_reference(self, 
            bearing_num: int, 
            folder_size: int
        ) -> (int):

        """
        Selects the reference bearing for a given bearing number.

        Parameters:
        - bearing_num (int): The bearing number.
        - folder_size (int): The size of the folder.

        Returns:
        - reference (int): The reference bearing number.
        """

        folder_no = folder_size
        reference = 1

        while folder_no / bearing_num < 1.0:
            folder_no += folder_size
            reference += folder_size
        
        return reference

    def prepare_data(self
    ) -> (list, list):

        """
        Prepare the data for survival analysis.

        Returns:
        - survival_covariates (list): A list of empty DataFrames for each bearing.
        - type_foldering (list): A list of ranges representing the folders of grouped bootstrapped bearings.
        """

        survival_covariates = []
        type_foldering = []

        # Prepare the empty data and for all the bearings
        for _ in range(0, self.real_bearings, 1):
            survival_covariates.append(pd.DataFrame())

        # Prepare the folders of grouped bootstrapped bearings to fill for avoid leaking depending on the bootstrap value and consequently the folder size
        for FOLDER in range(1, self.boot_folder_size * self.real_bearings, self.boot_folder_size):
            type_foldering.append(range(FOLDER, (FOLDER + self.boot_folder_size), 1))

        return survival_covariates, type_foldering


            





