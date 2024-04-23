import numpy as np
import re
import os

def estimate_target_rul_xjtu(
        data_path: list, 
        test: list, 
        test_condition: int
    ) -> (float):
    """
    Estimates the target Remaining Useful Life (RUL) using the XJTU-SY dataset.

    Args:
    - data_path (list): The path to the data.
    - test (list): A list of indices representing the test data.
    - test_condition (int): The actual test condition in the main pipeline script.

    Returns:
    - datasheet_target (float): The estimated target RUL.
    """
    # Prepare settings for calculating the target datasheet TtE
    dataset_TtE = "XJTU-SY"
    raw_type_test = data_path[test_condition]
    index = re.search(r"\d\d", raw_type_test)
    type_test = raw_type_test[index.start():-1]
    
    # Prepare path and a iterator for calculate the target datasheet TtE
    multiple_TtE = []
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    path_TtE = os.walk(os.path.join(parent_dir, "data", dataset_TtE, type_test))
    next(path_TtE)
    ITERATOR_TTE = 0
    
    # Count the number of files in the test data
    for next_root, next_dirs, next_files in path_TtE:
        if ITERATOR_TTE in test:
            multiple_TtE.append(len([f for f in os.listdir(next_root) if os.path.isfile(os.path.join(next_root, f))]))
        ITERATOR_TTE += 1
    
    # Make a unique value of TtE for the test data
    datasheet_target = np.median(multiple_TtE)

    return datasheet_target

def estimate_target_rul_pronostia(
        data_path: str, 
        test: list, 
        test_condition: int
    ) -> (float):
    """
    Estimates the target Remaining Useful Life (RUL) using the PRONOSTIA dataset.

    Args:
    - data_path (str): The path to the data.
    - test (list): A list of indices representing the test data.
    - test_condition (int): The actual test condition in the main pipeline script.

    Returns:
    - datasheet_target (float): The estimated target RUL.

    """

    # Prepare settings for calculate the target datasheet TtE
    dataset_TtE = "PRONOSTIA"
    raw_type_test = data_path[test_condition]
    index = re.search(r"\d\d", raw_type_test)
    type_test = raw_type_test[index.start():-1]
    
    # Prepare path and a iterator for calculate the target datasheet TtE
    multiple_TtE = []
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    path_TtE = os.walk(os.path.join(parent_dir, "data", dataset_TtE, type_test))
    next(path_TtE)
    ITERATOR_TTE = 0

    # Count the number of files in the test data
    for next_root, next_dirs, next_files in path_TtE:
        if ITERATOR_TTE in test:
            multiple_TtE.append(len([f for f in os.listdir(next_root) if os.path.isfile(os.path.join(next_root, f))]))
        ITERATOR_TTE += 1
    
    # Makes a unique value of TtE for the test data using median
    datasheet_target = np.median(multiple_TtE)

    return datasheet_target