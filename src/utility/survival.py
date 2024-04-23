import numpy as np
import pandas as pd
import math
import torch
from typing import List, Tuple, Optional, Union
import rpy2.robjects as robjects
import scipy.integrate as integrate
from sklearn.utils import shuffle
from dataclasses import InitVar, dataclass, field
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

def make_stratified_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'Event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'Survival_time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["Survival_time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["Event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

def make_stratification_label(df):
    t = df["Survival_time"]
    bins = np.linspace(start=t.min(), stop=t.max(), num=20)
    t = np.digitize(t, bins, right=True)
    e = df["Event"]
    stra_lab = np.stack([t, e], axis=1)
    return stra_lab

def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike
) -> (torch.Tensor, torch.Tensor):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    x = torch.tensor(dataset.drop(["Survival_time", "Event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["Survival_time"].values, dataset["Event"].values, time_bins)
    return x, y

def coverage(time_bins, upper, lower, true_times, true_indicator) -> float:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    time_bins = check_and_convert(time_bins)
    upper, lower = check_and_convert(upper, lower)
    true_times, true_indicator = check_and_convert(true_times, true_indicator)
    true_indicator = true_indicator.astype(bool)
    covered = 0
    upper_median_times = predict_median_survival_times(upper, time_bins, round_up=True)
    lower_median_times = predict_median_survival_times(lower, time_bins, round_up=False)
    covered += 2 * np.logical_and(upper_median_times[true_indicator] >= true_times[true_indicator],
                                  lower_median_times[true_indicator] <= true_times[true_indicator]).sum()
    covered += np.sum(upper_median_times[~true_indicator] >= true_times[~true_indicator])
    total = 2 * true_indicator.sum() + (~true_indicator).sum()
    return covered / total

def predict_median_survival_times(
        survival_curves: np.ndarray,
        times_coordinate: np.ndarray,
        round_up: bool = True
):
    median_probability_times = np.zeros(survival_curves.shape[0])
    max_time = times_coordinate[-1]
    slopes = (1 - survival_curves[:, -1]) / (0 - max_time)

    if round_up:
        # Find the first index in each row that are smaller or equal than 0.5
        times_indices = np.where(survival_curves <= 0.5, survival_curves, -np.inf).argmax(axis=1)
    else:
        # Find the last index in each row that are larger or equal than 0.5
        times_indices = np.where(survival_curves >= 0.5, survival_curves, np.inf).argmin(axis=1)

    need_extend = survival_curves[:, -1] > 0.5
    median_probability_times[~need_extend] = times_coordinate[times_indices][~need_extend]
    median_probability_times[need_extend] = (max_time + (0.5 - survival_curves[:, -1]) / slopes)[need_extend]

    return median_probability_times

def convert_to_structured (T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def make_event_times (t_train, e_train):
    unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
    if 0 not in unique_times:
        unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
    return unique_times.numpy() 

def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None
) -> torch.Tensor:
    """
    Courtesy of https://ieeexplore.ieee.org/document/10158019
    
    Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins

def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

def check_and_convert(*args):
    """ Makes sure that the given inputs are numpy arrays, list,
        tuple, panda Series, pandas DataFrames, or torch Tensors.

        Also makes sure that the given inputs have the same shape.

        Then convert the inputs to numpy array.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    last_length = ()
    for i, arg in enumerate(args):

        if len(arg) == 0:
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)

        else:

            if isinstance(arg, np.ndarray):
                x = (arg.astype(np.double),)
            elif isinstance(arg, list):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, tuple):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, pd.Series):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, pd.DataFrame):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, torch.Tensor):
                x = (arg.cpu().numpy().astype(np.double),)
            else:
                error = """{arg} is not a valid data format. Only use 'list', 'tuple', 'np.ndarray', 'torch.Tensor', 
                        'pd.Series', 'pd.DataFrame'""".format(arg=type(arg))
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, """Shapes between {}-th input array and 
                    {}-th input array are not consistent""".format(i - 1, i)
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result

class Survival:

    def __init__(self):
        pass
    
    @staticmethod
    def sanitize_survival_data (
            surv_preds: pd.DataFrame, 
            cvi: np.ndarray, 
            upper: float, 
            fix_ending: bool = False
        ) -> (pd.DataFrame, np.ndarray):

        """
        Sanitizes the survival data by fixing the ending of the survival function,
        replacing infs with 0, and removing rows where the first prediction is less than 0.5.

        Parameters:
        - surv_preds (pandas.DataFrame): The survival predictions.
        - cvi (numpy.ndarray): The cross-validation indices.
        - upper (float): The upper limit for fixing the ending of the survival function.
        - fix_ending (bool): Whether to fix the ending of the survival function. Default is False.

        Returns:
        - sanitized_surv_preds (pandas.DataFrame): The sanitized survival predictions.
        - sanitized_cvi (numpy.ndarray): The sanitized cross-validation indices.
        """

        # Fix ending of surv function
        if fix_ending:
            surv_preds.replace(np.nan, 1e-1000, inplace=True)
            surv_preds[math.ceil(upper)] = 1e-1000
            surv_preds.reset_index(drop=True, inplace=True)
        
        # Replace infs with 0
        surv_preds[~np.isfinite(surv_preds)] = 0

        # Remove rows where first pred is <0.5
        bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index
        sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
        sanitized_cvi = np.delete(cvi, bad_idx)
        
        return sanitized_surv_preds, sanitized_cvi
    
    @staticmethod
    def predict_survival_function(
            model, 
            X_test: pd.DataFrame, 
            times: np.ndarray,
            n_post_samples=100 # for MCD
        ) -> (pd.DataFrame):

        """
        Predicts the survival function for given test data using the specified model.

        Parameters:
        - model (pd.DataFrame): The survival model used for prediction.
        - X_test (pd.DataFrame): The test data.
        - times (np.ndarray): The time points at which to predict the survival function.

        Returns:
        - surv_prob (pd.DataFrame): The predicted survival probabilities at each time point.
        """

        # lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        # times = np.arange(np.ceil(lower + 1), np.floor(upper - 1), dtype=int)
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_survival_function(X_test).T
            return surv_prob
        elif model.__class__.__name__ == 'MCD':
            surv_prob = pd.DataFrame(np.mean(model.predict_survival(X_test, event_times=times, n_post_samples=n_post_samples), axis=0))
            return surv_prob
        else:
            surv_prob = pd.DataFrame(np.row_stack([fn(times) for fn in model.predict_survival_function(X_test)]), columns=times)
            return surv_prob
    
    @staticmethod
    def predict_hazard_function (
            model, 
            X_test: pd.DataFrame, 
            times: np.ndarray
        ) -> (pd.DataFrame):

        """
        Predicts the hazard function for a given model and test data.

        Parameters:
        - model (pd.DataFrame): The survival model used for prediction.
        - X_test (pd.DataFrame): The test data.
        - times (np.ndarray): The time points at which to predict the hazard function.

        Returns:
        - risk_pred (pd.DataFrame): The predicted hazards at each time point.
        """

        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_cumulative_hazard(X_test)
            return surv_prob
        elif model.__class__.__name__ == 'MCD':
            risk_pred = np.mean(model.predict_risk(X_test, event_times= times), axis= 0).flatten()
            return risk_pred             
        else:
            surv_prob = np.row_stack([fn(times) for fn in model.predict_cumulative_hazard_function(X_test)])
            return pd.DataFrame(surv_prob, columns=times)
    
    @staticmethod
    def sanitize_surv_functions_and_test_data (
            model, 
            X_test: pd.DataFrame, 
            times: np.ndarray, 
            y_test: np.ndarray
        ) -> (pd.DataFrame, pd.DataFrame):

        """
        Sanitizes the survival functions and test data by excluding certain elements based on the predicted survival probabilities.

        Parameters:
        - self (object): The instance of the class.
        - model (object): The predictive model.
        - X_test (array-like): The test data.
        - times (array-like): The event times.
        - y_test (array-like): The test labels.

        Returns:
        - sanitized_PDF_survival_probabilities (DataFrame): The sanitized survival probabilities.
        - sanitized_y_test (DataFrame): The sanitized data test.
        """

        # Init the element to sanitize
        PDF_survival_probabilities = model.predict_survival(X_test, event_times= times)
        sanitized_PDF_survival_probabilities = []
        sanitized_y_test = y_test

        # Find the bearings to exclude
        excluded_indexes = self.find_bearing_to_exclude (PDF_survival_probabilities)

        # Sanitize the bearings in a second moment to keep the array shape consistent (same size for each dimension element)
        sanitized_PDF_survival_probabilities = self.sanitize_PDF_survival_probabilities (PDF_survival_probabilities, excluded_indexes)

        # Sanitize the y_test
        for idx in excluded_indexes:
            sanitized_y_test = np.delete(sanitized_y_test, idx)

        return sanitized_PDF_survival_probabilities, sanitized_y_test

    def find_bearing_to_exclude (
            PDF_survival_probabilities: list
        ) -> list:

        """
        Finds the bearings to exclude based on the given survival probabilities.

        Parameters:
        - PDF_survival_probabilities (list): A list of survival probabilities for each bearing.

        Returns:
        - excluded_indexes (list): A list of bearing indexes to exclude after a a given condition.
        """

        excluded_indexes = []
        CONDITION = 0.5

        for PDF_survival_probability in PDF_survival_probabilities:
            for bearing_num, survival_probability in enumerate(PDF_survival_probability):
                if survival_probability [0] < CONDITION and bearing_num not in excluded_indexes:
                    excluded_indexes.append(bearing_num)

        return excluded_indexes

    def sanitize_PDF_survival_probabilities(
            PDF_survival_probabilities: list, 
            excluded_indexes: list
        ) -> list:

        """
        Sanitizes the PDF_survival_probabilities list by excluding the survival probabilities at the specified excluded_indexes.

        Args:
        - PDF_survival_probabilities (list): A list of lists representing the PDF survival probabilities.
        - excluded_indexes (list): A list of indexes to be excluded from the PDF_survival_probabilities.

        Returns:
        - sanitized_PDF_survival_probabilities (list): A sanitized version of the PDF_survival_probabilities list, where the survival probabilities at the excluded_indexes are removed.
        """

        sanitized_PDF_survival_probabilities = []

        for PDF_survival_probability in PDF_survival_probabilities:
            sanitized_survival_probabilities = []
            for bearing_num, survival_probability in enumerate(PDF_survival_probability):
                if bearing_num not in excluded_indexes:
                    sanitized_survival_probabilities.append(survival_probability)
            sanitized_PDF_survival_probabilities.append(sanitized_survival_probabilities)

        return sanitized_PDF_survival_probabilities
