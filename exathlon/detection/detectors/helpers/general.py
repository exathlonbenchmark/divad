"""General detector helpers."""
import copy
import logging
from typing import Optional, Union

import numpy as np
from scipy.stats import multivariate_normal
from imblearn.over_sampling import RandomOverSampler

from utils.guarding import check_is_percentage


def get_parsed_integer_list_str(input_: str) -> list:
    """Returns the integer list for the provided string of space-separated integers."""
    try:
        parsed = list(map(int, input_.split(" ")))
    except ValueError:
        if len(input_) == 0:
            parsed = []
        else:
            raise ValueError(
                f"`input_` can only be empty or space-separated "
                f"integers: received {input_}."
            )
    return parsed


def get_parsed_list_argument(arg: Union[int, str] = "") -> list:
    """Returns as a list `arg` provided as either an int or a space-separated string of ints."""
    if isinstance(arg, int):
        arg = str(arg)
    return get_parsed_integer_list_str(arg)


def get_parsed_float_list_str(input_: str) -> list:
    """Returns the float list for the provided string of space-separated floats."""
    try:
        parsed = list(map(float, input_.split(" ")))
    except ValueError:
        if len(input_) == 0:
            parsed = []
        else:
            raise ValueError(
                f"`input_` can only be empty or space-separated "
                f"floats: received {input_}."
            )
    return parsed


def get_flattened_windows(X: np.array) -> np.array:
    """Returns flattened windows from 3d array `X`.

    Args:
        X: 3d array of windows, of shape `(n_windows, window_size, n_features)`.

    Returns:
        Corresponding 2d array of flattened windows, of shape `(n_windows, window_size * n_features)`.
    """
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))


def get_packed_windows(X: np.array, window_size: int = 1) -> np.array:
    """Returns packed windows from 2d array `X`.

    Args:
        X: 2d array of flattened windows, of shape `(n_windows, window_size * n_features)`.
        window_size: window size used to pack windows.

    Returns:
        Corresponding 3d array of windows, of shape `(n_windows, window_size, n_features)`.
    """
    return X.reshape((X.shape[0], window_size, int(X.shape[-1] / window_size)))


def get_nll(x: np.array, mean: np.array, cov: np.array) -> Union[float, np.array]:
    """Returns the negative log-likelihood of `x` wrt a multivariate normal N(`mean`, `cov`).

    Args:
        x: input vector(s) to compute the NLL of, either of shape `(batch_size, n_features)`
          or of shape `(n_features,)`.
        mean: mean vector of the multivariate normal distribution, of shape `(n_features,)`.
        cov: covariance matrix of the multivariate normal distribution, of shape `(n_features, n_features)`.

    Returns:
        The negative log-likelihood of `x`, either as a scalar or a vector of shape `(batch_size,)`.
    """
    try:
        nll = -multivariate_normal.logpdf(x, mean=mean, cov=cov)
    except np.linalg.LinAlgError:
        logging.warning("Ill-conditioned covariance matrix when computing NLL.")
        nll = -multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=True)
    return nll


def get_normal_windows(
    X_train: np.array,
    y_train: np.array = None,
    X_val: np.array = None,
    y_val: np.array = None,
    train_info: np.array = None,
    val_info: np.array = None,
) -> (np.array, np.array, np.array, np.array):
    """Returns training and validation windows with removed anomalies if relevant."""
    returned_train_info = None
    returned_val_info = None
    if y_train is not None:
        train_normal_mask = y_train == 0
        X_train = X_train[train_normal_mask]
        y_train = y_train[train_normal_mask]
        assert len(np.unique(y_train)) == 1 and y_train[0] == 0
        if train_info is not None:
            returned_train_info = copy.deepcopy(train_info)
            for k, v in train_info.items():
                returned_train_info[k] = v[train_normal_mask]
    if X_val is not None and y_val is not None:
        val_normal_mask = y_val == 0
        X_val = X_val[val_normal_mask]
        y_val = y_val[val_normal_mask]
        assert len(np.unique(y_val)) == 1 and y_val[0] == 0
        if val_info is not None:
            returned_val_info = copy.deepcopy(val_info)
            for k, v in val_info.items():
                returned_val_info[k] = v[val_normal_mask]
    if returned_train_info is not None or returned_val_info is not None:
        return X_train, y_train, X_val, y_val, returned_train_info, returned_val_info
    return X_train, y_train, X_val, y_val


def get_memory_used(X: np.array) -> (float, str):
    """Returns the memory used by `X`, along with the memory unit ("GiB", "MiB" or "kiB")."""
    memory = None
    memory_unit = None
    n_bytes = X.nbytes
    for i, unit in enumerate(["GiB", "MiB", "kiB"]):
        memory = n_bytes / (1024 ** (3 - i))
        if memory >= 1 or unit == "kiB":
            memory_unit = unit
            break
    return memory, memory_unit


def log_windows_memory(X_train: np.array, X_val: Optional[np.array] = None) -> None:
    """Logs the memory used by `X_train` and `X_val`."""
    train_mem, train_mem_unit = get_memory_used(X_train)
    logging.info(f"X_train: {round(train_mem, 2)} {train_mem_unit}")
    if X_val is not None:
        val_mem, val_mem_unit = get_memory_used(X_val)
        logging.info(f"X_val: {round(val_mem, 2)} {val_mem_unit}")


def get_balanced_samples(
    X: np.array, y: np.array, n_ano_per_normal: float, random_seed: int
) -> (np.array, np.array):
    """Returns `X` and `y` with `n_ano_per_normal` anomalous samples per normal sample.

    Args:
        X: input samples of shape `(n_samples, window_size, n_features)`.
        y: corresponding multiclass labels of shape `(n_samples,)`.
        n_ano_per_normal: number of anomalous samples per normal sample, always balanced by type.
         Should be between 0 and 1. For example, 0.2 corresponds to a 5:1 normal/anomaly ratio.
        random_seed: random seed used for random oversampling and shuffling.

    Returns:
        `X` and `y` with `n_ano_per_normal` anomalous samples per normal sample, and anomaly types
         balanced within anomalous samples if relevant.
    """
    check_is_percentage(n_ano_per_normal, "n_ano_per_normal")
    np.random.seed(random_seed)
    ano_mask = y > 0
    X_normal = X[~ano_mask]
    y_normal = y[~ano_mask]
    X_ano = X[ano_mask]
    y_ano = y[ano_mask]
    n_normal, window_size, n_features = X_normal.shape
    n_ano = X_ano.shape[0]
    ano_classes = np.unique(y_ano)
    n_ano_classes = len(ano_classes)
    new_n_ano = int(n_ano_per_normal * n_normal)
    if n_ano_per_normal == 1.0:
        # make the number of normal and anomalous samples match exactly
        # remove instead of oversample to make sure there is no normal duplicate later in a batch
        n_removed_normal = new_n_ano % n_ano_classes
        if n_removed_normal > 0:
            X_normal = X_normal[:-n_removed_normal]
            y_normal = y_normal[:-n_removed_normal]
            n_normal = X_normal.shape[0]
            X = np.concatenate([X_normal, X_ano])
            y = np.concatenate([y_normal, y_ano])
    n_per_ano_class = new_n_ano // n_ano_classes
    for c in ano_classes:
        class_ano_mask = y == c
        n_class_ano = np.sum(class_ano_mask)
        n_removed_class_ano = n_class_ano - n_per_ano_class
        # more anomalies originally than requested: undersample instead oversampling
        if n_removed_class_ano > 0:
            X_class_ano = X[class_ano_mask]
            y_class_ano = y[class_ano_mask]
            other_X = X[~class_ano_mask]
            other_y = y[~class_ano_mask]
            X_class_ano = X_class_ano[:-n_removed_class_ano]
            y_class_ano = y_class_ano[:-n_removed_class_ano]
            X = np.concatenate([other_X, X_class_ano])
            y = np.concatenate([other_y, y_class_ano])
            n_ano -= n_removed_class_ano
    sampling_strategy = {c: n_per_ano_class for c in ano_classes}
    sampling_strategy[0] = n_normal
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy, random_state=random_seed
    )
    X_resampled, y_resampled = ros.fit_resample(
        X.reshape((n_normal + n_ano, window_size * n_features)), y
    )
    X_resampled = X_resampled.reshape((X_resampled.shape[0], window_size, n_features))
    return X_resampled, y_resampled
