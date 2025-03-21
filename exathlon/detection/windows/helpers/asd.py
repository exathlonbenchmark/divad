import math
from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_balanced_asd(data: dict, set_name: str, key: str, random_seed: int = 0) -> dict:
    """Returns windows and information of the provided set names balanced by `key`.

    This ensures an equal number of windows per key value, but keeps the same proportion
    of files within each key.

    Args:
        data: input data.
        set_name: set name.
        key: balancing key: only "file_name" is supported.
        random_seed: random seed to use for reproducibility across calls.

    Returns:
        Balanced windows and information (with just the keys containing "set_name").
    """
    np.random.seed(random_seed)
    balanced_data = {
        f"X_{set_name}": None,
        f"y_{set_name}": None,
        f"{set_name}_info": dict(),
    }
    X = data[f"X_{set_name}"]
    y = data[f"y_{set_name}"]
    data_info = data[f"{set_name}_info"]
    if key == "file_name":
        unique_keys = np.unique(data_info["file_name"])
    else:
        raise NotImplementedError(f'Only key="file_name" is supported. Received {key}.')
    file_names = np.unique(data_info["file_name"])
    n_windows = X.shape[0]
    n_unique_keys = unique_keys.shape[0]
    if n_unique_keys > 1:
        n_per_key = n_windows // n_unique_keys
        file_to_window_ids = dict()
        file_to_n_resampled_windows = dict()
        for key_value in unique_keys:
            key_file_names = list(
                filter(lambda fn: file_name_key_equals(fn, key, key_value), file_names)
            )
            n_key_windows = len(
                np.where(data_info_key_equals(data_info, key, key_value))[0]
            )
            for file_name in key_file_names:
                file_to_window_ids[file_name] = np.where(
                    data_info["file_name"] == file_name
                )[0]
                n_file_windows = len(file_to_window_ids[file_name])
                # preserve distribution of file names within key windows
                file_window_prop = n_file_windows / n_key_windows
                file_to_n_resampled_windows[file_name] = math.ceil(
                    file_window_prop * n_per_key
                )

        resampled_windows_ids = []
        for file_name, file_window_ids in file_to_window_ids.items():
            n_file_windows = len(file_window_ids)
            n_resampled_windows = file_to_n_resampled_windows[file_name]
            resampled_windows_ids.append(
                np.random.choice(
                    file_window_ids,
                    n_resampled_windows,
                    replace=n_resampled_windows > n_file_windows,
                )
            )
        resampled_windows_ids = np.concatenate(resampled_windows_ids)
    else:
        # single key: keep all input windows
        resampled_windows_ids = np.arange(n_windows)
    for k, v in zip(["X", "y"], [X, y]):
        balanced_data[f"{k}_{set_name}"] = v[resampled_windows_ids]
    for info_k, info_v in data_info.items():
        balanced_data[f"{set_name}_info"][info_k] = info_v[resampled_windows_ids]
    return balanced_data


def file_name_key_equals(file_name: str, key: str, key_value: Union[str, int]) -> bool:
    if key == "file_name":
        return file_name == key_value
    raise ValueError


def data_info_key_equals(
    data_info: NDArray[str], key: str, key_value: Union[str, int]
) -> NDArray[bool]:
    if key == "file_name":
        return data_info["file_name"] == key_value
    raise ValueError
