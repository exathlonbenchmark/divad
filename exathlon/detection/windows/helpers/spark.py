import math
from typing import Union

import numpy as np
from numpy.typing import NDArray

from utils.guarding import check_value_in_choices


def get_balanced(data: dict, set_name: str, key: str, random_seed: int = 0) -> dict:
    """Returns windows and information of the provided set names balanced by `key`.

    This ensures an equal number of windows per key value, but keeps the same proportion
    of files within each key.

    Args:
        data: input data.
        set_name: set name.
        key: balancing key: either "app", "rate", "type-rate" or "app-type-rate".
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
    if key == "app":
        unique_keys = np.unique(data_info["app_id"])
    elif key == "rate":
        unique_keys = np.unique(data_info["input_rate"])
    elif key == "type-rate":
        trace_types = data_info["trace_type"]
        input_rates = data_info["input_rate"]
        type_rates = [f"{t}_{r}" for t, r in zip(trace_types, input_rates)]
        unique_keys = np.unique(type_rates)
    elif key == "settings-rate":
        settings = data_info["settings"]
        input_rates = data_info["input_rate"]
        settings_rates = [f"{s}_{r}" for s, r in zip(settings, input_rates)]
        unique_keys = np.unique(settings_rates)
    elif key == "app-type-rate":
        app_ids = data_info["app_id"]
        trace_types = data_info["trace_type"]
        input_rates = data_info["input_rate"]
        app_type_rates = [
            f"{a}_{t}_{r}" for a, t, r in zip(app_ids, trace_types, input_rates)
        ]
        unique_keys = np.unique(app_type_rates)
    elif key == "app-settings-rate":
        app_ids = data_info["app_id"]
        settings = data_info["settings"]
        input_rates = data_info["input_rate"]
        app_settings_rates = [
            f"{a}_{s}_{r}" for a, s, r in zip(app_ids, settings, input_rates)
        ]
        unique_keys = np.unique(app_settings_rates)
    else:
        raise ValueError
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
    if key == "app":
        return int(file_name.split("_")[0]) == key_value
    elif key == "rate":
        return int(file_name.split("_")[2]) == key_value
    elif key == "type-rate":
        type_, rate = key_value.split("_")
        return file_name.split("_")[1] == type_ and file_name.split("_")[2] == rate
    elif key == "settings-rate":
        settings, rate = key_value.split("_")
        return (
            "-".join(file_name.split("_")[3:6]) == settings
            and file_name.split("_")[2] == rate
        )
    elif key == "app-type-rate":
        app, type_, rate = key_value.split("_")
        return (
            file_name.split("_")[0] == app
            and file_name.split("_")[1] == type_
            and file_name.split("_")[2] == rate
        )
    elif key == "app-settings-rate":
        app, settings, rate = key_value.split("_")
        return (
            file_name.split("_")[0] == app
            and "-".join(file_name.split("_")[3:6]) == settings
            and file_name.split("_")[2] == rate
        )
    raise ValueError


def data_info_key_equals(
    data_info: NDArray[str], key: str, key_value: Union[str, int]
) -> NDArray[bool]:
    if key == "app":
        return data_info["app_id"] == key_value
    elif key == "rate":
        return data_info["input_rate"] == key_value
    elif key == "type-rate":
        type_, rate = list(map(int, key_value.split("_")))
        return (data_info["trace_type"] == type_) & (data_info["input_rate"] == rate)
    elif key == "settings-rate":
        settings, rate = key_value.split("_")
        rate = int(rate)
        return (data_info["settings"] == settings) & (data_info["input_rate"] == rate)
    elif key == "app-type-rate":
        app, type_, rate = list(map(int, key_value.split("_")))
        return (
            (data_info["app_id"] == app)
            & (data_info["trace_type"] == type_)
            & (data_info["input_rate"] == rate)
        )
    elif key == "app-settings-rate":
        app, settings, rate = key_value.split("_")
        app = int(app)
        rate = int(rate)
        return (
            (data_info["app_id"] == app)
            & (data_info["settings"] == settings)
            & (data_info["input_rate"] == rate)
        )
    raise ValueError


def get_balanced_keys_by_app(
    data: dict, set_name: str, key: str = "file_name", random_seed: int = 0
) -> dict:
    """For each application, return windows and information balanced by the provided `key`.

    Args:
        data: input data.
        set_name: set name.
        key: information key on which to balance windows for each application (either
          "file_name" or "input_rate").
        random_seed: random seed to use for reproducibility across calls.

    Returns:
        Balanced windows and information (with just the keys containing `set_name`).
    """
    np.random.seed(random_seed)
    check_value_in_choices(key, "key", ["file_name", "input_rate"])
    balanced_data = {
        f"X_{set_name}": None,
        f"y_{set_name}": None,
        f"{set_name}_info": dict(),
    }
    X = data[f"X_{set_name}"]
    y = data[f"y_{set_name}"]
    data_info = data[f"{set_name}_info"]
    app_ids = np.unique(data_info["app_id"])
    app_to_window_ids = {
        app_id: np.where(data_info["app_id"] == app_id)[0] for app_id in app_ids
    }
    resampled_windows_ids = []
    for app_id, app_window_ids in app_to_window_ids.items():
        app_keys = np.unique(data_info[key][app_window_ids])
        n_app_keys = app_keys.shape[0]
        app_key_to_window_ids = {
            k: np.intersect1d(app_window_ids, np.where(data_info[key] == k)[0])
            for k in app_keys
        }
        if n_app_keys > 1:
            n_app_windows = app_window_ids.shape[0]
            n_per_app_key = n_app_windows // n_app_keys
            for app_key, app_key_window_ids in app_key_to_window_ids.items():
                n_app_key_windows = app_key_window_ids.shape[0]
                resampled_windows_ids.append(
                    np.random.choice(
                        app_key_window_ids,
                        n_per_app_key,
                        replace=n_app_key_windows < n_per_app_key,
                    )
                )
        else:
            # single key value for the application: keep all its windows
            resampled_windows_ids.append(app_window_ids)
    resampled_windows_ids = np.concatenate(resampled_windows_ids)
    for k, v in zip(["X", "y"], [X, y]):
        balanced_data[f"{k}_{set_name}"] = v[resampled_windows_ids]
    for info_k, info_v in data_info.items():
        balanced_data[f"{set_name}_info"][info_k] = info_v[resampled_windows_ids]
    return balanced_data


def get_balanced_files_by_app_rate(
    data: dict, set_name: str, random_seed: int = 0
) -> dict:
    """Returns windows and information of the provided set names balanced by
      files for each (application, input rate).

    Args:
        data: input data.
        set_name: set name.
        random_seed: random seed to use for reproducibility across calls.

    Returns:
        Balanced windows and information (with just the keys containing `set_name`).
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
    app_ids = np.unique(data_info["app_id"])
    app_to_window_ids = {
        app_id: np.where(data_info["app_id"] == app_id)[0] for app_id in app_ids
    }
    resampled_windows_ids = []
    for app_id, app_window_ids in app_to_window_ids.items():
        app_rates = np.unique(data_info["input_rate"][app_window_ids])
        app_rate_to_window_ids = {
            rate: np.intersect1d(
                app_window_ids, np.where(data_info["input_rate"] == rate)[0]
            )
            for rate in app_rates
        }
        for app_rate, app_rate_window_ids in app_rate_to_window_ids.items():
            app_rate_files = np.unique(data_info["file_name"][app_rate_window_ids])
            n_app_rate_files = app_rate_files.shape[0]
            app_rate_file_to_window_ids = {
                file: np.intersect1d(
                    app_rate_window_ids, np.where(data_info["file_name"] == file)[0]
                )
                for file in app_rate_files
            }
            if n_app_rate_files > 1:
                n_app_rate_windows = app_rate_window_ids.shape[0]
                n_per_app_rate_file = n_app_rate_windows // n_app_rate_files
                for (
                    app_rate_file,
                    app_rate_file_window_ids,
                ) in app_rate_file_to_window_ids.items():
                    n_app_rate_file_windows = app_rate_file_window_ids.shape[0]
                    resampled_windows_ids.append(
                        np.random.choice(
                            app_rate_file_window_ids,
                            n_per_app_rate_file,
                            replace=n_app_rate_file_windows < n_per_app_rate_file,
                        )
                    )
            else:
                # a single file for (application, input rate) pair: keep all windows for the pair
                resampled_windows_ids.append(app_rate_window_ids)
    resampled_windows_ids = np.concatenate(resampled_windows_ids)
    for k, v in zip(["X", "y"], [X, y]):
        balanced_data[f"{k}_{set_name}"] = v[resampled_windows_ids]
    for info_k, info_v in data_info.items():
        balanced_data[f"{set_name}_info"][info_k] = info_v[resampled_windows_ids]
    return balanced_data
