"""Data helpers module.

Gathers functions for loading and manipulating sequence DataFrames and numpy arrays.
"""
import os
import math
import copy
import json
import pickle
import logging
from typing import Optional, List

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tensorflow as tf


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder used to serialize numpy ndarrays."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyJSONEncoder, self).default(o)


def is_string_type(str_, type_=float):
    """Returns True if `str_` can be converted to type `type_`, False otherwise.

    Args:
        str_ (str): input string whose type to test.
        type_ (type): standard python type (e.g. `int` or `float`).

    Returns:
        bool: True if `str_` can be converted to type `type_`, False otherwise.
    """
    try:
        type_(str_)
        return True
    except ValueError:
        return False


def resample_sequences(
    seq_dfs: List[pd.DataFrame],
    sampling_period: str,
    agg: str = "mean",
    anomaly_col: bool = False,
    original_sampling_period: Optional[str] = None,
) -> None:
    """Resamples sequences inplace to `sampling_period` using the provided aggregation function.

    If `anomaly_col` is False, sequence indices will also be reset to start from a round date,
    in order to keep them aligned with any subsequently resampled labels.

    Args:
        seq_dfs: the sequence DataFrames to resample.
        sampling_period: the new sampling period, either as a valid argument to `pd.Timedelta` or "na".
            If `sampling_period` is "na", the provided periods are returned without resampling.
        agg: the aggregation function defining the resampling (e.g. "mean", "median" or "max").
        anomaly_col (bool): whether the provided DataFrames have an `Anomaly` column, to resample differently.
        original_sampling_period: original sampling period of the DataFrames.
            If None and `anomaly_col` is False, the original sampling period will be inferred
            from the first two records of the first DataFrame.
    """
    if sampling_period != "na":
        sampling_timedelta = pd.Timedelta(sampling_period)
        original_sampling_timedelta = None
        feature_cols = [c for c in seq_dfs[0].columns if c != "anomaly"]
        if not anomaly_col:
            # turn any original sampling period to type `pd.Timedelta`
            original_sampling_timedelta = (
                pd.Timedelta(np.diff(seq_dfs[0].index[:2])[0])
                if original_sampling_period is None
                else pd.Timedelta(original_sampling_period)
            )
        logging.info(
            f"Resampling sequences applying records `{agg}` every {sampling_period}..."
        )
        for i, seq_df in enumerate(seq_dfs):
            if not anomaly_col:
                # reindex the period DataFrame to start from a round date before downsampling
                seq_df = seq_df.set_index(
                    pd.date_range(
                        "01-01-2000",
                        periods=len(seq_df),
                        freq=original_sampling_timedelta,
                    )
                )
            seq_dfs[i] = (
                seq_df[feature_cols]
                .resample(sampling_timedelta)
                .agg(agg)
                .ffill()
                .bfill()
            )
            if anomaly_col:
                # if no records during `sampling_timedelta`, we choose to simply
                # repeat the label of the last one here
                seq_dfs[i]["anomaly"] = (
                    seq_df["anomaly"]
                    .resample(sampling_timedelta)
                    .agg("max")
                    .ffill()
                    .bfill()
                )
        logging.info("Done.")


def load_files(
    input_path,
    file_names,
    file_format,
    *,
    drop_info_suffix=False,
    drop_labels_prefix=False,
    parse_keys=True,
):
    """Loads and returns the provided `file_names` files from `input_path`.

    Args:
        input_path (str): path to load the files from.
        file_names (list): list of file names to load (without file extensions).
        file_format (str): format of the files to load, must be either `pickle`, `json` or `numpy`.
        drop_info_suffix (bool): if True, drop any `_info` string from the output dict keys.
        drop_labels_prefix (bool): if True, drop any `y_` string from the output dict keys.
        parse_keys (bool): if True and loading `json` format, parse keys to type `int` or `float` if relevant.

    Returns:
        dict: the loaded files of the form `{file_name: file_content}`.
    """
    a_t = "supported formats only include `pickle`, `json` and `numpy`"
    assert file_format in ["pickle", "json", "numpy"], a_t
    if file_format == "pickle":
        ext = "pkl"
    else:
        ext = "json" if file_format == "json" else "npy"
    files_dict = dict()
    print(f"loading {file_format} files from {input_path}")
    for fn in file_names:
        print(f"loading `{fn}.{ext}`...", end=" ", flush=True)
        if file_format == "pickle":
            files_dict[fn] = pickle.load(
                open(os.path.join(input_path, f"{fn}.{ext}"), "rb")
            )
        elif file_format == "json":
            files_dict[fn] = json.load(open(os.path.join(input_path, f"{fn}.{ext}")))
            # parse keys to type float or int if specified and relevant
            if parse_keys:
                if all([is_string_type(k, float) for k in files_dict[fn]]):
                    if all([is_string_type(k, int) for k in files_dict[fn]]):
                        files_dict[fn] = {int(k): v for k, v in files_dict[fn].items()}
                    else:
                        files_dict[fn] = {
                            float(k): v for k, v in files_dict[fn].items()
                        }
        else:
            files_dict[fn] = np.load(
                os.path.join(input_path, f"{fn}.{ext}"), allow_pickle=True
            )
        print("done.")
    if not (drop_info_suffix or drop_labels_prefix):
        return files_dict
    if drop_info_suffix:
        files_dict = {k.replace("_info", ""): v for k, v in files_dict.items()}
    if drop_labels_prefix:
        return {k.replace("y_", ""): v for k, v in files_dict.items()}
    return files_dict


def save_files(output_path, files_dict, file_format):
    """Saves files from the provided `files_dict` to `output_path` in the relevant format.

    Args:
        output_path (str): path to save the files to.
        files_dict (dict): dictionary of the form `{file_name: file_content}` (file names without extensions).
        file_format (str): format of the files to save, must be either `pickle`, `json` or `numpy`.
    """
    a_t = "supported formats only include `pickle`, `json` and `numpy`"
    assert file_format in ["pickle", "json", "numpy"], a_t
    if file_format == "pickle":
        ext = "pkl"
    else:
        ext = "json" if file_format == "json" else "npy"
    print(f"saving {file_format} files to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    for fn in files_dict:
        print(f"saving `{fn}.{ext}`...", end=" ", flush=True)
        if file_format == "pickle":
            with open(os.path.join(output_path, f"{fn}.{ext}"), "wb") as pickle_file:
                pickle.dump(files_dict[fn], pickle_file)
        elif file_format == "json":
            with open(os.path.join(output_path, f"{fn}.{ext}"), "w") as json_file:
                json.dump(
                    files_dict[fn],
                    json_file,
                    separators=(",", ":"),
                    sort_keys=True,
                    indent=4,
                    cls=NumpyJSONEncoder,
                )
        else:
            np.save(
                os.path.join(output_path, f"{fn}.{ext}"),
                files_dict[fn],
                allow_pickle=True,
            )
        print("done.")


def load_mixed_formats(file_paths, file_names, file_formats):
    """Loads and returns `file_names` stored as `file_formats` files at `file_paths`.

    Args:
        file_paths (list): list of paths for each file name.
        file_names (list): list of file names to load (without file extensions).
        file_formats (list): list of file formats for each name, must be either `pickle`, `json` or `numpy`.

    Returns:
        dict: the loaded files, with as keys the file names.
    """
    assert (
        len(file_names) == len(file_paths) == len(file_formats)
    ), "the provided lists must be of same lengths"
    files = dict()
    for name, path, format_ in zip(file_names, file_paths, file_formats):
        files[name] = load_files(path, [name], format_)[name]
    return files


def load_datasets_data(
    input_path,
    info_path,
    dataset_names,
    data_sampling_period="1s",
    labels_sampling_period="1s",
    n_starting_ignored=0,
    n_ending_ignored=0,
):
    """Returns the periods records, labels and information for the provided dataset names.

    The provided `data_sampling_period` and `labels_sampling_period` should either be both valid
    arguments to `pd.Timedelta` or both "na". In the latter case, this means the concept of a
    sampling period is not applicable to the dataset used.

    Args:
        input_path (str): path from which to load the records and labels.
        info_path (str): path from which to load the periods information.
        dataset_names (list): list of dataset names.
        data_sampling_period (str): data sampling period, as a valid argument to `pd.Timedelta`, or
            "na" if not applicable.
        labels_sampling_period (str): labels sampling period, as a valid argument to `pd.Timedelta`,
            or "na" if not applicable.
        n_starting_ignored (int): number of beginning records and labels to ignore.
        n_ending_ignored (int): number of ending records and labels to ignore.

    Returns:
        dict: the datasets data, with keys of the form `{n}`, `y_{n}` and `{n}_info` (`n` the dataset name).
    """
    # load period records, labels and information for each specified dataset
    file_names = [fn for n in dataset_names for fn in [n, f"y_{n}", f"{n}_info"]]
    n_sets = len(dataset_names)
    file_paths, file_formats = n_sets * (2 * [input_path] + [info_path]), n_sets * (
        2 * ["numpy"] + ["pickle"]
    )
    data = load_mixed_formats(file_paths, file_names, file_formats)

    # number of labels for a given data record (accounting for potentially different sampling periods)
    if not (any([sp == "na" for sp in [data_sampling_period, labels_sampling_period]])):
        dsp, lsp = (
            pd.Timedelta(data_sampling_period).seconds,
            pd.Timedelta(labels_sampling_period).seconds,
        )
        assert (
            dsp % lsp == 0
        ), "the data sampling period must be a multiple of the labels sampling period"
        labels_prop_factor = int(dsp / lsp)
    else:
        labels_prop_factor = 1

    # starting and ending records and labels to ignore
    if all([ignored == 0 for ignored in [n_starting_ignored, n_ending_ignored]]):
        # no data trimming
        return data
    info_keys = [k for k in data if "info" in k]
    trimmed_data = {k: [] for k in data if k not in info_keys}
    for set_name in dataset_names:
        for i, p in enumerate(data[set_name]):
            trimmed_data[set_name].append(
                get_trimmed_array(p, n_starting_ignored, n_ending_ignored)
            )
            trimmed_data[f"y_{set_name}"].append(
                get_trimmed_labels(
                    data[f"y_{set_name}"][i],
                    len(p),
                    n_starting_ignored,
                    n_ending_ignored,
                    labels_prop_factor,
                )
            )
    # return trimmed periods back as ndarrays
    return dict(
        {k: np.array(v) for k, v in trimmed_data.items()},
        **{k: data[k] for k in info_keys},
    )


def get_trimmed_labels(
    labels, data_length, n_starting=0, n_ending=0, labels_prop_factor=1
):
    """Returns `labels` without their first `n_starting` and last `n_ending` records, accounting for
        the provided proportionality factor between labels and data.

    We assume records and/or labels might have been downsampled based on datetime using pandas' strategy:

    => Every record corresponds to `labels_prop_factor` labels, except for the last one that may correspond
    to *at most* `labels_prop_factor` labels, depending on whether the end datetime of the period was a
    multiple of the sampling period or not.

    Args:
        labels (ndarray): labels to trim of shape `(n_labels,)`.
        data_length (int): number of data records corresponding to the provided labels.
        n_starting (int): number of *data records* to remove from the beginning of labels, internally
            turned into the corresponding number of labels `n_starting_labels`.
        n_ending (int): number of *data records* to remove from the end of labels, internally
            turned into the corresponding number of labels `n_ending_labels`.
        labels_prop_factor (int): number of labels for a given record, as deduced from the ratio between
            data and labels sampling periods.

    Returns:
        ndarray: trimmed labels of shape `(n_labels - n_starting_labels - n_ending_labels,)`.
    """
    a_t = "there should remain some data after removing beginning and ending records"
    assert data_length > n_starting + n_ending, a_t
    upsampled_length = labels_prop_factor * data_length
    # number of labels corresponding to the last data record
    n_last_labels = labels_prop_factor - (upsampled_length - len(labels))
    # the `n_starting` records do not include the last one and hence all correspond to `labels_prop_factor` labels
    n_starting_labels = labels_prop_factor * n_starting
    # adapt number of ending labels according to the number of labels for the last record
    n_ending_labels = labels_prop_factor * n_ending
    if n_ending_labels != 0 and n_last_labels < labels_prop_factor:
        n_ending_labels = n_ending_labels - labels_prop_factor + n_last_labels
    return labels[
        n_starting_labels : (None if n_ending_labels == 0 else -n_ending_labels)
    ]


def extract_save_labels(
    seq_dfs: List[pd.DataFrame],
    labels_file_name: str,
    output_path: str,
    sampling_period: Optional[str] = None,
    original_sampling_period: Optional[str] = None,
) -> None:
    """Extracts and saves labels from the "anomaly" columns of the provided sequence DataFrames.

    If labels are resampled before being saved, their indices will also be reset to start
    from a round date, in order to keep them aligned with any subsequently resampled records.

    Note: the "anomaly" column is removed from the sequence DataFrames inplace.

    Args:
        seq_dfs: list of sequence pd.DataFrame.
        labels_file_name: name of the numpy labels file to save (without ".npy" extension).
        output_path: path to save the labels to, as a numpy array of shape `(n_seqs, seq_length)`.
            Where `seq_length` depends on the sequence.
        sampling_period: if specified, period to resample the labels to before saving them
            (as a valid argument to `pd.Timedelta`).
        original_sampling_period: original sampling period of the DataFrames.
            If None and `sampling_period` is specified, the original sampling period
            will be inferred from the first two records of the first DataFrame.
    """
    labels_list = []
    sampling_p, pre_sampling_p = None, None
    if sampling_period is not None:
        sampling_p = pd.Timedelta(sampling_period)
        if original_sampling_period is None:
            pre_sampling_p = pd.Timedelta(np.diff(seq_dfs[0].index[:2])[0])
        else:
            pre_sampling_p = pd.Timedelta(original_sampling_period)
    for i, seq_df in enumerate(seq_dfs):
        if sampling_period is None:
            labels_list.append(seq_df["anomaly"].values.astype(int))
        else:
            # resample labels (after resetting their indices to start from a round date)
            labels_series = seq_df[["anomaly"]].set_index(
                pd.date_range("01-01-2000", periods=len(seq_df), freq=pre_sampling_p)
            )["anomaly"]
            labels_list.append(
                labels_series.resample(sampling_p)
                .agg("max")
                .ffill()
                .bfill()
                .values.astype(int)
            )
        # remove the "anomaly" column from the sequence inplace
        seq_dfs[i].drop("anomaly", axis=1, inplace=True)

    # save extracted labels
    logging.info(f"Saving {labels_file_name} labels file...")
    np.save(
        os.path.join(output_path, f"{labels_file_name}.npy"),
        get_numpy_from_numpy_list(labels_list),
    )
    logging.info("Done.")


def get_numpy_from_numpy_list(numpy_list):
    """Returns the equivalent numpy array for the provided list of numpy arrays.

    Args:
        numpy_list (list): the list of numpy arrays to turn into a numpy array.

    Returns:
        ndarray: corresponding numpy array of shape `(list_length, ...)`. If the arrays
            in the list have different shapes, the final array is returned with dtype object.
    """
    # if the numpy list contains a single ndarray or all its ndarrays have the same shape
    if len(numpy_list) == 1 or len(set([a.shape for a in numpy_list])) == 1:
        # return with original data type
        return np.array(numpy_list)
    # else return with data type "object"
    return np.array(numpy_list, dtype=object)


def get_numpy_from_dfs(period_dfs):
    """Returns the equivalent numpy 3d-array for the provided list of period DataFrames.

    Args:
        period_dfs (list): the list of period DataFrames to turn into a numpy array.

    Returns:
        ndarray: corresponding numpy array of shape `(n_periods, period_size, n_features)`.
            Where `period_size` depends on the period.
    """
    return get_numpy_from_numpy_list(
        [period_df.to_numpy().astype(np.float32) for period_df in period_dfs]
    )


def get_dfs_from_numpy(periods, sampling_period):
    """Returns the equivalent list of period DataFrames for the provided numpy 3d-array.

    Args:
        periods (ndarray): numpy array of shape `(n_periods, period_size, n_features)` (where
            `period_size` depends on the period), to turn into a list of DataFrames.
        sampling_period (str): time resolution to use for the output DataFrames (starting from Jan. 1st, 2000).

    Returns:
        list: corresponding list of DataFrames with their time resolution set to `sampling_period`.
    """
    return [
        pd.DataFrame(
            p, pd.date_range("01-01-2000", periods=len(p), freq=sampling_period)
        )
        for p in periods
    ]


def get_aligned_shuffle(array, *arrays, **kw_arrays):
    """Returns `array`, `arrays` and `kw_arrays` randomly shuffled, preserving alignments between their elements.

    All provided arrays must have the same length as `array`.

    Args:
        array (ndarray): first array to shuffle.
        *arrays (ndarrays): optional additional arrays to shuffle accordingly.
        **kw_arrays (ndarrays): optional additional arrays with keywords, to shuffle accordingly
            and return as a dictionary.

    Returns:
        (ndarray, ndarray, ...)|(ndarray, ndarray, ...), dict: the shuffled array(s).
    """
    single_array = len(arrays) == 0 and len(kw_arrays) == 0
    if not single_array:
        a_t = "all arrays to shuffle must have the same length"
        assert (
            len(
                {
                    len(array),
                    *[len(a) for a in arrays],
                    *[len(a) for a in kw_arrays.values()],
                }
            )
            == 1
        ), a_t
    mask = np.random.permutation(len(array))
    if single_array:
        return array[mask]
    if len(kw_arrays) == 0:
        return [array[mask], *[a[mask] for a in arrays]]
    if len(arrays) == 0:
        return array[mask], {k: a[mask] for k, a in kw_arrays.items()}
    return [
        *[array[mask], *[a[mask] for a in arrays]],
        {k: a[mask] for k, a in kw_arrays.items()},
    ]


def get_aligned_shuffled_arrays_dict(arrays_dict):
    """Returns `arrays_dict` with its ndarray values shuffled, preserving alignments between their elements."""
    assert (
        len(set([len(v) for v in arrays_dict.values()])) == 1
    ), "all ndarray values must have the same length"
    mask = np.random.permutation(len(list(arrays_dict.values())[0]))
    return {k: v[mask] for k, v in arrays_dict.items()}


def get_batches(
    X: NDArray[np.float32], batch_size: int = 256
) -> List[NDArray[np.float32]]:
    return list(
        tf.data.Dataset.from_tensor_slices(X)
        .batch(batch_size, drop_remainder=False)
        .as_numpy_iterator()
    )


def get_sliding_windows(
    array,
    window_size,
    window_step,
    include_remainder=False,
    dtype=np.float64,
    ranges_only=False,
):
    """Returns sliding windows of `window_size` elements extracted every `window_step` from `array`.

    Args:
        array (ndarray): input ndarray whose first axis will be used for extracting windows.
        window_size (int): number of elements of the windows to extract.
        window_step (int): step size between two adjacent windows to extract.
        include_remainder (bool): whether to include any remaining window, with a different step size.
        dtype (type): optional data type to enforce for the window elements (default np.float64).
            Only relevant if `ranges_only` is False.
        ranges_only (bool): whether to only return sliding window (inclusive) start and (exclusive)
            end indices in `array`.

    Returns:
        ndarray: windows (or window ranges) array of shape `(n_windows, window_size, *array.shape[1:])`.
    """
    window_ranges, start_idx = [], 0
    while start_idx <= array.shape[0] - window_size:
        window_ranges.append((start_idx, start_idx + window_size))
        start_idx += window_step
    if include_remainder and start_idx - window_step + window_size != array.shape[0]:
        window_ranges.append((array.shape[0] - window_size, array.shape[0]))
    if ranges_only:
        return np.array(window_ranges)
    return np.array([array[s:e] for s, e in window_ranges], dtype=dtype)


def get_nansum(array, **kwargs):
    """Returns the sum of array elements over a given axis treating NaNs as zero.

    Same as `np.nansum`, expect NaN is returned instead of zero if all array elements are NaN.

    Args:
        array (ndarray): array whose sum of elements to compute.
        **kwargs: optional keyword arguments to pass to `np.nansum`.

    Returns:
        float: the sum of array elements (or NaN if all NaNs).
    """
    if np.isnan(array).all():
        return np.nan
    return np.nansum(array, **kwargs)


def get_matching_sampling(
    arrays,
    target_arrays,
    *,
    agg_func=np.max,
    sampling_period="1s",
    target_sampling_period="1s",
):
    """Returns `arrays` resampled so as to match the sampling period of `target_arrays`.

    All arrays within `arrays` and `target_arrays` are expected to have the same sampling period.
    `arrays` and `target_arrays` are expected to match, in the sense that they should contain the
    same number of elements, but just expressed using different sampling periods.

    Hence, if `arrays` contain more (resp., less) records than `target_arrays`, `arrays` will
    be downsampled (resp., upsampled). If relevant, `agg_func` will be used to downsample `arrays`,
    and `sampling_period`, `target_sampling_period` will be used to upsample `arrays`.

    The provided `sampling_period` and `target_sampling_period` should either be both valid
    arguments to `pd.Timedelta` or both "na". In the latter case, this means the concept of a
    sampling period is not applicable to the dataset used.

    Args:
        arrays (ndarray): arrays to resample along the first axis, of shape
            `(n_arrays, array_length, ...)`, with `array_length` depending on the array.
        target_arrays (ndarray): arrays whose lengths to match in the same format.
        agg_func (func): the numpy aggregation function to apply in case `arrays` need to be downsampled.
        sampling_period (str): sampling period of the original arrays, as a valid argument to `pd.Timedelta`
            (or "na" if not applicable).
        target_sampling_period (str): sampling period of the target arrays, as a valid argument to `pd.Timedelta`
            (or "na" if not applicable).

    Returns:
        ndarray: resampled `arrays`, in the same format but with the same lengths as `target_arrays`.
    """
    if any([sp == "na" for sp in [sampling_period, target_sampling_period]]):
        return arrays
    a_t = "the number of arrays should match the number of target arrays"
    assert len(arrays) == len(target_arrays), a_t
    if len(arrays[0]) <= len(target_arrays[0]):
        # upsample records of arrays
        f = get_upsampled
        kwargs = {
            "sampling_period": sampling_period,
            "target_sampling_period": target_sampling_period,
        }
    else:
        # downsample records of arrays using the provided aggregation function
        f, kwargs = get_downsampled, {"agg_func": agg_func}
    return get_numpy_from_numpy_list(
        [f(a, len(ta), **kwargs) for a, ta in zip(arrays, target_arrays)]
    )


def get_upsampled(array, target_length, sampling_period, target_sampling_period):
    """Returns `array` upsampled so as to have `target_length` elements.

    We assume `array` shorter than `target_length`, and therefore `sampling_period` larger than
    `target_sampling_period`.

    Each element of `array` corresponds to `prop_factor = sampling_period / target_sampling_period`
    elements in the array of `target_length`, except for the last element, that might correspond to
    at most `prop_factor` elements, as told by the target length.

    Args:
        array (ndarray): the array to upsample of shape `(array_length, ...)`.
        target_length (int): the number of elements to reach for `array`.
        sampling_period (str): sampling period of the original array, as a valid argument to `pd.Timedelta`.
        target_sampling_period (str): sampling period of the target array, as a valid argument to `pd.Timedelta`.

    Returns:
        array (ndarray): the upsampled array of shape `(target_length, ...)`.
    """
    sp, tsp = (
        pd.Timedelta(sampling_period).seconds,
        pd.Timedelta(target_sampling_period).seconds,
    )
    assert (
        sp % tsp == 0
    ), "the array sampling period must be a multiple of the target sampling period"
    upsampled = np.repeat(array, int(sp / tsp))
    n_removed = len(upsampled) - target_length
    return upsampled[: None if n_removed == 0 else -n_removed]


def get_downsampled_windows(
    X: NDArray[np.float32],
    downsampling_size: int = 1,
    downsampling_step: int = 1,
    downsampling_func: str = "mean",
):
    """Returns downsampled `X` of shape `(n_windows, window_size, n_features)`."""
    # sub_windows shape: (window_size, downsampling_size, n_features)
    if downsampling_func != "mean":
        raise NotImplementedError(
            f'Only "mean" window downsampling is currently supported. '
            f"Received {downsampling_func}."
        )
    X_downsampled = []
    for x in X:
        # x_windows: (n_sub_windows, downsampling_size, n_features)
        x_windows = get_sliding_windows(
            x,
            window_size=downsampling_size,
            window_step=downsampling_step,
            include_remainder=True,
            dtype=np.float32,
        )
        # x_downsampled: (n_sub_windows, n_features)
        x_downsampled = np.mean(x_windows, axis=1)
        X_downsampled.append(x_downsampled)
    # X_downsampled: (n_windows, n_sub_windows, n_features)
    X_downsampled = np.array(X_downsampled, dtype=np.float32)
    return X_downsampled


def get_downsampled(array, target_length, agg_func=np.max):
    """Returns `array` downsampled so as to have `target_length` elements.

    Args:
        array (ndarray): the array to downsample of shape `(array_length, ...)`.
        target_length (int): the number of elements to reach for `array`.
        agg_func (func): the numpy aggregation function to apply when downsampling `array`.

    Returns:
        array (ndarray): the downsampled array of shape `(target_length, ...)`.
    """
    array_length = len(array)
    window_size = round(array_length / target_length)
    jumping_windows = get_sliding_windows(
        array,
        window_size,
        window_size,
        include_remainder=window_size * target_length >= array_length,
        dtype=array.dtype,
    )
    return get_numpy_from_numpy_list([agg_func(w) for w in jumping_windows])


def get_trimmed_array(array, n_starting, n_ending):
    """Returns `array` without its first `n_starting` and last `n_ending` records."""
    return array[n_starting : (None if n_ending == 0 else -n_ending)]


def get_trimmed(arrays, n_starting=0, n_ending=0):
    """Returns `arrays` without their first `n_starting` and last `n_ending` records.

    Args:
        arrays (ndarray): arrays to trim of shape `(n_arrays, array_length, ...)`, with `array_length`
            depending on the array.
        n_starting (int): number of records to remove from the beginning of each array.
        n_ending (int): number of records to remove from the end of each array.

    Returns:
        ndarray: trimmed arrays of shape `(n_arrays, array_length - n_starting - n_ending, ...)`.
    """
    end_index = None if n_ending == 0 else -n_ending
    return get_numpy_from_numpy_list([a[n_starting:end_index] for a in arrays])


def get_concatenated(array_1, array_2):
    """Returns `array_1` and `array_2` concatenated along their first axis."""
    if len(array_2) == 0:
        return array_1
    if len(array_1) == 0:
        return array_2
    return np.concatenate([array_1, array_2])


# function returning True if its argument is a sequence (excluding strings), False otherwise
def is_sequence(v):
    return hasattr(v, "__len__") and not isinstance(v, str)


def get_aligned_list_values(dict_):
    """Returns `dict_` with its unit-length sequence values repeated `max_length` times.

    The number of repetitions is defined as the maximum length across sequence values (i.e., `max_length`).
    All dictionary values must be provided as sequences whose lengths are either one or `max_length`.

    Args:
        dict_ (dict): dictionary whose unit-length sequences to repeat `max_length` times.

    Returns:
        dict: the updated dictionary, with all sequences returned as lists of length `max_length`.
    """
    returned_dict, lengths = copy.deepcopy(dict_), []
    for v in dict_.values():
        if not is_sequence(v):
            raise ValueError("all dictionary values must be provided as sequences")
        lengths.append(len(v))
    lengths_set = set(lengths)
    if len(lengths_set) > 2 or (len(lengths_set) == 2 and 1 not in lengths_set):
        raise ValueError(
            "dictionary sequence lengths must be either one or the maximum provided"
        )
    max_length = max(lengths_set)
    for k, v in returned_dict.items():
        returned_dict[k] = (
            [v[0] for _ in range(max_length)] if len(v) == 1 else list(returned_dict[k])
        )
    return returned_dict


def get_list_value(value, length):
    """Returns `value` as a list of length `length`.

    If value is already a sequence, make sure it is of length `length` and return it as a list.
    If it is not a sequence, return it as a list repeating it `length` times.

    Args:
        value (object): value to return as a list.
        length (int): length of the list to return.

    Returns:
        list: the value as a list of length `length`.
    """
    if is_sequence(value):
        assert len(value) == length, f"the provided sequence must be of length {length}"
        return list(value)
    return [value for _ in range(length)]


def get_balanced_sample_ids(
    y: np.array,
    resampled_n_ano_per_normal: float = 1.0,
    balance_ano_types: bool = True,
    random_seed: int = 0,
    shuffle: bool = True,
) -> np.array:
    """Returns resampled IDs from `y`, with `resampled_n_ano_per_normal` anomalies per normal sample.

    Args:
        y: input multiclass labels of shape `(n_samples,)`.
        resampled_n_ano_per_normal: number of anomalies per normal sample after resampling.
        balance_ano_types: whether to balance anomaly types within the anomaly class if multiple are found.
        random_seed: random seed used for random oversampling and shuffling.
        shuffle: whether to shuffle IDs after resampling (if `False`, then IDs are returned such that classes
          in `y` appear sequentially).

    Returns:
        Resampled IDs (not shuffled by class if `shuffle` is `False`).
    """
    np.random.seed(random_seed)
    ano_mask = y > 0
    normal_ids = np.where(~ano_mask)[0]
    ano_ids = np.where(ano_mask)[0]
    n_normal = normal_ids.shape[0]
    n_resampled_ano = math.ceil(resampled_n_ano_per_normal * n_normal)
    if balance_ano_types:
        ano_classes = np.unique(y[ano_ids])
    else:
        ano_classes = np.array([1.0], dtype=np.float32)
    n_ano_classes = len(ano_classes)
    n_per_ano_class = n_resampled_ano // n_ano_classes
    resampled_ids = normal_ids
    for c in ano_classes:
        ano_class_mask = y == c if balance_ano_types else y > 0.0
        ano_class_ids = np.where(ano_class_mask)[0]
        resampled_class_ano_ids = np.random.choice(
            ano_class_ids,
            n_per_ano_class,
            replace=(n_per_ano_class > len(ano_class_ids)),
        )
        resampled_ids = np.concatenate([resampled_ids, resampled_class_ano_ids], axis=0)
    if shuffle:
        resampled_ids = np.random.permutation(resampled_ids)
    return resampled_ids


def get_sample_weights(
    y: np.array, ano_weight_per_normal: float = 1.0, balance_ano_types: bool = True
) -> np.array:
    """Returns the sample weights to "balance" classes in `y`, with a proportion `resampled_n_ano_per_normal`
     of the total normal weight assigned to anomalies.

    Args:
        y: input multiclass labels of shape `(n_samples,)`.
        ano_weight_per_normal: proportion of the total normal weight to assign to anomalies.
        balance_ano_types: whether to assign higher weights to less prevalent anomaly classes to
         make all classes count as much no matter their cardinality.

    Returns:
        The sample weights to "balance" classes in `y`.
    """
    ano_mask = y > 0
    normal_ids = np.where(~ano_mask)[0]
    ano_ids = np.where(ano_mask)[0]
    n_normal = normal_ids.shape[0]
    sample_weights = np.ones(y.shape[0], dtype=np.float32)
    if balance_ano_types:
        ano_classes = np.unique(y[ano_ids])
    else:
        ano_classes = np.array([1.0], dtype=np.float32)
    n_ano_classes = len(ano_classes)
    for c in ano_classes:
        ano_class_mask = y == c if balance_ano_types else y > 0.0
        n_ano_class = np.sum(ano_class_mask)
        ano_class_weight = ano_weight_per_normal * n_normal / n_ano_class
        ano_class_weight /= n_ano_classes
        sample_weights[ano_class_mask] = ano_class_weight
    return sample_weights
