"""Window-related helpers."""
import os
from typing import Optional

from utils.logging_ import get_verbose_print
from data.helpers import load_mixed_formats, get_concatenated


def load_window_datasets(
    input_path: str,
    set_names: Optional[list] = None,
    return_labels: bool = False,
    return_info: bool = False,
    concat_datasets: bool = True,
    normal_only: bool = True,
    verbose: bool = False,
):
    """Loads and returns window datasets at `input_path`.

    Args:
        input_path: path from which to load sample elements.
        set_names: dataset names for which to load sample elements.
        return_labels: whether to include sample labels to the returned dictionary.
        return_info: whether to include samples information to the returned dictionary.
        concat_datasets (bool): whether to return dataset elements concatenated, removing set
            names from the keys of the returned dictionary.
        normal_only: whether to only return "normal" sample elements (labeled as 0).
        verbose: whether to print progress texts to the console.

    Returns:
        dict: modeling sample ndarrays, as described in `WindowManager.get_modeling_split()`. If
            `concat_datasets` is True, elements from the considered datasets will be returned
            concatenated, and set names removed from the dictionary keys.
    """
    # get printing behavior from verbose
    input_set_names = [fn[2:-4] for fn in os.listdir(input_path) if fn[:2] == "X_"]
    if set_names is None:
        set_names = input_set_names
    else:
        set_names = [sn for sn in set_names if sn in input_set_names]
    v_print = get_verbose_print(verbose)
    window_datasets = dict()
    v_print(f"loading modeling samples from {input_path}:")
    for sn in set_names:
        file_names = [f"X_{sn}"]
        file_formats = ["numpy"]
        if return_labels or normal_only:
            file_names.append(f"y_{sn}")
            file_formats.append("numpy")
        if return_info:
            file_names.append(f"{sn}_info")
            file_formats.append("pickle")
        v_print(f"loading {sn} sample elements...", end=" ", flush=True)
        set_data = load_mixed_formats(
            len(file_names) * [input_path], file_names, file_formats
        )
        v_print("done.")
        if normal_only:
            v_print(f"removing {sn} anomalous samples...", end=" ", flush=True)
            samples_mask = set_data[f"y_{sn}"] == 0
            for k, v in set_data.items():
                if k != f"{sn}_info":
                    set_data[k] = v[samples_mask]
                else:
                    for info_key, info_value in set_data[f"{sn}_info"].items():
                        set_data[f"{sn}_info"][info_key] = info_value[samples_mask]
            v_print("done.")
        if not return_labels:
            set_data.pop(f"y_{sn}", None)
        window_datasets = dict(window_datasets, **set_data)
    if not concat_datasets:
        return window_datasets
    # concatenate windows into a single dataset, removing set names from keys
    concat_window_datasets = {"info": dict()} if return_info else dict()
    for sn in set_names:
        # data-related sample elements
        for k in ["X", "y"]:
            if f"{k}_{sn}" in window_datasets:
                if k not in concat_window_datasets:
                    concat_window_datasets[k] = []
                concat_window_datasets[k] = get_concatenated(
                    concat_window_datasets[k], window_datasets[f"{k}_{sn}"]
                )
        # samples information if specified
        if return_info:
            # information keys are assumed to be the same for each dataset
            for info_key in window_datasets[f"{set_names[0]}_info"]:
                if info_key not in concat_window_datasets["info"]:
                    concat_window_datasets["info"][info_key] = []
                concat_window_datasets["info"][info_key] = get_concatenated(
                    concat_window_datasets["info"][info_key],
                    window_datasets[f"{sn}_info"][info_key],
                )
    return concat_window_datasets
