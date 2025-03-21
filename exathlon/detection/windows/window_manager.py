"""Modeling split module. Constituting the train/val/test sets for normality modeling.
"""
import re
import math
import random
import logging
from typing import Optional, Union

import numpy as np
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
)
from imblearn.under_sampling import RandomUnderSampler

from utils.logging_ import get_verbose_print
from utils.guarding import check_value_in_choices
from data.helpers import save_files, get_sliding_windows, get_downsampled_windows
from detection.detectors.helpers.general import get_memory_used
from detection.metrics.helpers import extract_multiclass_ranges_ids
from detection.windows.helpers.spark import (
    get_balanced,
    get_balanced_keys_by_app,
    get_balanced_files_by_app_rate,
)
from detection.windows.helpers.asd import get_balanced_asd

# TODO: move this function elsewhere.
from detection.detectors.helpers.general import get_parsed_integer_list_str


AUGMENTATION_TO_CLASS = {
    "smote": SMOTE,
    "borderline_smote": BorderlineSMOTE,
    "svm_smote": SVMSMOTE,
    "adasyn": ADASYN,
}


class WindowManager:
    """Window manager class.

    TODO: rename to window_balancing and window_balancing_seed, and make only the choices
     depend on the dataset.

    Args:
        window_size: window size to use when constituting windows.
        window_step: window step to use when constituting windows.
        downsampling_size: window size to use when downsampling windows (1 for no downsampling).
        downsampling_step: window step to use when downsampling windows if `downsampling_size > 1`.
        downsampling_func: function to use when downsampling windows if `downsampling_size > 1`.
        n_periods: number of periods to set as input normal data (first largest,
          then selected at random).
        normal_data_prop: proportion of input normal data to consider when constituting
          the modeling datasets.
        normal_sampling_seed: random seed to use when subsampling normal data if relevant.
        window_min_ano_coverage: minimum proportion of achievable anomaly coverage
          defining anomalous windows.
        window_weak_ano_policy: policy regarding anomalous windows that are "not anomalous enough".
        class_balancing: class balancing strategy ("none", "naive" or "naive.ano).
        class_balancing_seed: random seed to use when balancing classes if relevant.
        dropped_anomaly_types: anomaly types to drop when constituting modeling windows,
          as an empty string for no dropped type, an integer for a single type, or a string of
          space-separated integers for multiple types.
        anomaly_augmentation: anomaly augmentation strategy (either "none", "smote",
          "borderline_smote", "svm_smote" or "adasyn"). Currently, options other than "none"
          are only supported for a window size of 1.
        ano_augment_n_per_normal: if `anomaly_augmentation` is not "none", augmentation will
          be performed so that we end up with this number of anomalies for each normal point.
        ano_augment_seed: if `anomaly_augmentation` is not "none", random seed to use when augmenting
          anomaly samples.
        output_path: output path to save datasets and information to.
        dataset_name: dataset name used for dataset-specific window management.
        spark_balancing: balancing strategy for spark data, only relevant if `dataset_name == "spark"`.
        spark_balancing_seed: random seed to use for spark balancing.
        asd_balancing: balancing strategy for ASD data, only relevant if `dataset_name == "asd"`.
        asd_balancing_seed: random seed to use for ASD balancing.
        train_set_name: training set name to use when saving training windows.
        val_set_name: validation set name to use when saving validation windows.
        test_set_name: test set name to use when saving test windows.
    """

    def __init__(
        self,
        window_size: int = 1,
        window_step: int = 1,
        downsampling_size: int = 1,
        downsampling_step: int = 1,
        downsampling_func: str = "mean",
        n_periods: int = -1,
        normal_data_prop: float = 1.0,
        normal_sampling_seed: int = 0,
        window_min_ano_coverage: float = 0.2,
        window_weak_ano_policy: str = "drop",
        class_balancing: str = "none",
        class_balancing_seed: int = 0,
        dropped_anomaly_types: Union[int, str] = "",
        anomaly_augmentation: str = "none",
        ano_augment_n_per_normal: int = 500,
        ano_augment_seed: int = 0,
        output_path: str = ".",
        dataset_name: str = None,
        spark_balancing: str = "none",
        spark_balancing_seed: int = 0,
        asd_balancing: str = "none",
        asd_balancing_seed: int = 0,
        train_set_name: str = "train",
        val_set_name: str = "val",
        test_set_name: str = "test",
    ):
        check_value_in_choices(downsampling_func, "downsampling_func", ["mean"])
        check_value_in_choices(
            class_balancing, "class_balancing", ["none", "naive", "naive_ano"]
        )
        check_value_in_choices(
            anomaly_augmentation,
            "anomaly_augmentation",
            ["none", "smote", "borderline_smote", "svm_smote", "adasyn"],
        )
        self.window_size = window_size
        self.window_step = window_step
        self.downsampling_size = downsampling_size
        self.downsampling_step = downsampling_step
        self.downsampling_func = downsampling_func
        self.n_periods = n_periods
        self.normal_data_prop = normal_data_prop
        self.normal_sampling_seed = normal_sampling_seed
        self.window_min_ano_coverage = window_min_ano_coverage
        self.window_weak_ano_policy = window_weak_ano_policy
        self.class_balancing = class_balancing
        self.class_balancing_seed = class_balancing_seed
        # data-specific anomaly types to drop when constituting anomalous samples
        if isinstance(dropped_anomaly_types, int):
            dropped_anomaly_types = str(dropped_anomaly_types)
        self.dropped_anomaly_types = get_parsed_integer_list_str(dropped_anomaly_types)
        self.anomaly_augmentation = anomaly_augmentation
        self.ano_augment_n_per_normal = ano_augment_n_per_normal
        self.ano_augment_seed = ano_augment_seed
        self.output_path = output_path
        self.dataset_name = dataset_name
        # `period_info_parsing_func`: dataset-specific `list -> dict` period information parsing
        #   function, to add information about the containing periods of windows.
        self.spark_balancing = None
        self.spark_balancing_seed = None
        self.asd_balancing = None
        self.asd_balancing_seed = None
        if self.dataset_name == "spark":
            check_value_in_choices(
                spark_balancing,
                "spark_balancing",
                [
                    "none",
                    "app",
                    "rate",
                    "type-rate",
                    "settings-rate",
                    "app-type-rate",
                    "app-settings-rate",
                    "app_file",
                    "app_rate",
                    "app_rate_file",
                ],
            )
            self.period_info_parsing_func = lambda info: {
                "file_name": info[0],
                "app_id": int(info[0].split("_")[0]),
                "trace_type": int(info[0].split("_")[1]),
                "input_rate": int(info[0].split("_")[2]),
                "settings": "-".join(info[0].split("_")[3:6]),
            }
            self.spark_balancing = spark_balancing
            self.spark_balancing_seed = spark_balancing_seed
        elif self.dataset_name == "synthetic_dg":
            self.period_info_parsing_func = lambda info: {
                "file_name": info[0],
                "mean": info[1],
                "offset": info[2],
                "rank": info[3],
            }
        elif self.dataset_name == "asd":
            check_value_in_choices(
                asd_balancing, "asd_balancing", ["none", "file_name"]
            )
            self.period_info_parsing_func = lambda info: {
                "file_name": info[0],
                "rank": info[1],
            }
            self.asd_balancing = asd_balancing
            self.asd_balancing_seed = asd_balancing_seed
        else:
            self.period_info_parsing_func = lambda x: dict()
        self.train_set_name = train_set_name
        self.val_set_name = val_set_name
        self.test_set_name = test_set_name
        # "not-applicable" (N/A) anomaly information keys to default as zero instead of -1
        # (for "type 0 = normal" and "0% of achievable anomaly coverage", respectively)
        self.zero_defaulting_keys = ["type", "coverage"]
        # types to use for anomaly range information values
        self.ano_info_types_dict = {
            "type": int,
            "start": int,
            "end": int,
            "coverage": float,
            "instance_id": int,
        }

    def save_window_datasets(
        self,
        train_periods: np.array,
        train_periods_labels: np.array,
        train_periods_info: list,
        val_periods: Optional[np.array] = None,
        val_periods_labels: Optional[np.array] = None,
        val_periods_info: Optional[list] = None,
        test_periods: Optional[np.array] = None,
        test_periods_labels: Optional[np.array] = None,
        test_periods_info: Optional[list] = None,
    ) -> dict:
        """Saves and returns the final shuffled train/val/test window datasets from the provided sequences.

        Returns `data` dictionary, with at keys for `set_name` in ["train", "val", "test"], (when relevant):

        - "X_`set_name`": set windows of shape `(n_windows, window_size, n_features)`.
        - "y_`set_name`": window labels (0 if normal, else label of their "main" anomaly range).
        - "`set_name`_info": dictionary with at keys ndarrays of shape `(n_windows)`:
            - "sample_id": a unique identifier for the window.
            - "period_id": a unique identifier for the period the window was extracted from.
            - "n_anomalies": the number of anomaly ranges contained in the window.
            - "main_ano_id": the index of the "main" anomaly range of the window (i.e., the range for which
                the window achieves the largest achievable coverage given its size, or its latest range in
                case of equality).
            - For each anomaly range `i` in the window (starting from zero, with keys prefixed with "ano`i`_"):
                - "type": the integer type of the range.
                - "start": the (included) start index of the range in the window.
                - "end": the (excluded) end index of the range in the window.
                - "coverage": the proportion of achievable anomaly coverage given the window size.
                    ("Anomaly coverage" refers to the proportion of the anomaly range that lies in the window.)
                    ("Achievable anomaly coverage" is the coverage that would be achieved if the range spanned all
                    the window, or 1, if the window is larger than the anomaly range.)
                - "instance_id": a unique identifier for the anomaly instance the range belongs to.
            - "window_size": the original window size before any downsampling.
            - "downsampling_size": the downsampling window size (1 for no downsampling).
            - "downsampling_step": the downsampling window step if "downsampling_size" is greater than 1.
            - "downsampling_func": the downsampling window function if "downsampling_size" is greater than 1.

        Args:
            train_periods: periods of shape `(n_periods, period_length, n_features)`,
              where `period_length` depends on the period.
            train_periods_labels: multiclass labels for each period of shape `(n_periods, period_length)`.
            train_periods_info: period information lists (one per period).
            val_periods: optional validation periods, in the same format as `periods`.
            val_periods_labels: optional multiclass labels for each validation period.
            val_periods_info: optional validation period information lists (one per period).
            test_periods: optional test periods, in the same format as `periods`.
            test_periods_labels: optional multiclass labels for each test period.
            test_periods_info: optional test period information lists (one per period).

        Returns:
            The final shuffled train/val/test window datasets, as described above.
        """
        # fix data selection random seeds for reproducibility across calls
        random.seed(self.normal_sampling_seed)

        # only consider the provided number of training sequences if specified
        sampled_train_periods = train_periods
        sampled_train_periods_labels = train_periods_labels
        sampled_train_periods_info = train_periods_info
        if self.n_periods != -1:
            # the first selected sequence is always the largest one
            largest_idx = np.argmax([p.shape[0] for p in train_periods])
            train_period_ids = [largest_idx]
            # the remaining sequences, if any, are randomly sampled
            if self.n_periods > 1:
                train_period_ids += random.sample(
                    [i for i in range(len(train_periods)) if i != largest_idx],
                    self.n_periods - 1,
                )
            sampled_train_periods = train_periods[train_period_ids]
            sampled_train_periods_labels = sampled_train_periods_labels[
                train_period_ids
            ]
            sampled_train_periods_info = [
                v for i, v in enumerate(train_periods_info) if i in train_period_ids
            ]
            # save the selected periods information
            logging.info(f"Saving {self.n_periods} selected periods information...")
            saved_dict = {
                "train_period_ids": [int(id_) for id_ in train_period_ids],
                "prop": sum([p.shape[0] for p in sampled_train_periods])
                / sum([p.shape[0] for p in train_periods]),
            }
            save_files(self.output_path, {"selected_periods_info": saved_dict}, "json")
            logging.info("Done.")

        # get window data, labels and information
        X_train, y_train, train_windows_info = self.get_window_data(
            sampled_train_periods,
            sampled_train_periods_labels,
            sampled_train_periods_info,
        )
        data = {
            f"X_{self.train_set_name}": X_train,
            f"y_{self.train_set_name}": y_train,
            f"{self.train_set_name}_info": train_windows_info,
        }
        non_empty_set_names = [self.train_set_name]
        added_set_names = []
        added_data = []
        if val_periods is not None:
            added_set_names.append(self.val_set_name)
            added_data.append([val_periods, val_periods_labels, val_periods_info])
        if test_periods is not None:
            added_set_names.append(self.test_set_name)
            added_data.append([test_periods, test_periods_labels, test_periods_info])
        non_empty_set_names += added_set_names
        for set_name, (set_periods, set_periods_labels, set_periods_info) in zip(
            added_set_names, added_data
        ):
            set_X, set_y, set_windows_info = self.get_window_data(
                set_periods, set_periods_labels, set_periods_info
            )
            for k, v in zip(
                [f"X_{set_name}", f"y_{set_name}", f"{set_name}_info"],
                [set_X, set_y, set_windows_info],
            ):
                data[k] = v
        shuffle_datasets(data, non_empty_set_names)

        # downsample windows if specified
        for n in non_empty_set_names:
            if self.downsampling_size > 1:
                logging.info(f'data["X_{n}"] before downsampling:')
                logging.info(f'Shape: {data[f"X_{n}"].shape}')
                mem, mem_unit = get_memory_used(data[f"X_{n}"])
                logging.info(f"{round(mem, 2)} {mem_unit}")
                data[f"X_{n}"] = get_downsampled_windows(
                    data[f"X_{n}"],
                    self.downsampling_size,
                    self.downsampling_step,
                    self.downsampling_func,
                )
                logging.info(f'data["X_{n}"] after downsampling:')
                logging.info(f'Shape: {data[f"X_{n}"].shape}')
                mem, mem_unit = get_memory_used(data[f"X_{n}"])
                logging.info(f"{round(mem, 2)} {mem_unit}")
            keys = [
                "window_size",
                "downsampling_size",
                "downsampling_step",
                "downsampling_func",
            ]
            values = [
                self.window_size,
                self.downsampling_size,
                self.downsampling_step,
                self.downsampling_func,
            ]
            for k, v in zip(keys, values):
                data[f"{n}_info"][k] = np.repeat(v, data[f"X_{n}"].shape[0])

        # spark-specific window balancing
        if self.dataset_name == "spark" and self.spark_balancing != "none":
            logging.info("SPARK BALANCING")
            # balance windows according to application id
            for sn in non_empty_set_names:
                logging.info(sn.upper())
                logging.info(f'Windows before balancing: {data[f"X_{sn}"].shape[0]}')
                balanced_set_data = data
                if self.spark_balancing in [
                    "app",
                    "app_rate",
                    "app_file",
                    "app_rate_file",
                ]:
                    if self.spark_balancing == "app_rate_file":
                        # balance traces within each (application, input rate) pair
                        balanced_set_data = get_balanced_files_by_app_rate(
                            balanced_set_data, sn, random_seed=self.spark_balancing_seed
                        )
                    if self.spark_balancing in ["app_rate", "app_rate_file"]:
                        # balance input rates within each application
                        balanced_set_data = get_balanced_keys_by_app(
                            balanced_set_data,
                            sn,
                            "input_rate",
                            random_seed=self.spark_balancing_seed,
                        )
                    elif self.spark_balancing == "app_file":
                        # balance traces within each application
                        balanced_set_data = get_balanced_keys_by_app(
                            balanced_set_data,
                            sn,
                            "file_name",
                            random_seed=self.spark_balancing_seed,
                        )
                    # balance applications
                    balanced_set_data = get_balanced(
                        balanced_set_data,
                        sn,
                        key="app",
                        random_seed=self.spark_balancing_seed,
                    )
                elif self.spark_balancing in [
                    "rate",
                    "type-rate",
                    "settings-rate",
                    "app-type-rate",
                    "app-settings-rate",
                ]:
                    balanced_set_data = get_balanced(
                        balanced_set_data,
                        sn,
                        key=self.spark_balancing,
                        random_seed=self.spark_balancing_seed,
                    )
                else:
                    raise ValueError
                for k in ["X", "y"]:
                    data[f"{k}_{sn}"] = balanced_set_data[f"{k}_{sn}"]
                for info_k in data[f"{sn}_info"]:
                    data[f"{sn}_info"][info_k] = balanced_set_data[f"{sn}_info"][info_k]
                del balanced_set_data
                # logging and debugging
                logging.info(f'Windows after balancing: {data[f"X_{sn}"].shape[0]}')
                for app_id in np.unique(data[f"{sn}_info"]["app_id"]):
                    app_window_ids = np.where(data[f"{sn}_info"]["app_id"] == app_id)[0]
                    logging.info(f"Application {app_id}: {app_window_ids.shape[0]}")
                    for app_rate in np.unique(
                        data[f"{sn}_info"]["input_rate"][app_window_ids]
                    ):
                        app_rate_window_ids = np.intersect1d(
                            app_window_ids,
                            np.where(data[f"{sn}_info"]["input_rate"] == app_rate)[0],
                        )
                        logging.info(
                            f"  Rate={app_rate:,}: {app_rate_window_ids.shape[0]}"
                        )
                        for app_rate_file in np.unique(
                            data[f"{sn}_info"]["file_name"][app_rate_window_ids]
                        ):
                            n_app_rate_file_windows = sum(
                                data[f"{sn}_info"]["file_name"][app_rate_window_ids]
                                == app_rate_file
                            )
                            logging.info(
                                f"    File={app_rate_file}: {n_app_rate_file_windows}"
                            )
        # ASD-specific window balancing
        if self.dataset_name == "asd" and self.asd_balancing != "none":
            logging.info("ASD BALANCING")
            # balance windows according to file name
            for sn in non_empty_set_names:
                logging.info(sn.upper())
                logging.info(f'Windows before balancing: {data[f"X_{sn}"].shape[0]}')
                for file_name in np.unique(data[f"{sn}_info"]["file_name"]):
                    file_window_ids = np.where(
                        data[f"{sn}_info"]["file_name"] == file_name
                    )[0]
                    logging.info(f"File {file_name}: {file_window_ids.shape[0]}")
                balanced_set_data = data
                balanced_set_data = get_balanced_asd(
                    balanced_set_data,
                    sn,
                    key=self.asd_balancing,
                    random_seed=self.asd_balancing_seed,
                )
                for k in ["X", "y"]:
                    data[f"{k}_{sn}"] = balanced_set_data[f"{k}_{sn}"]
                for info_k in data[f"{sn}_info"]:
                    data[f"{sn}_info"][info_k] = balanced_set_data[f"{sn}_info"][info_k]
                del balanced_set_data
                # logging and debugging
                logging.info(f'Windows after balancing: {data[f"X_{sn}"].shape[0]}')
                for file_name in np.unique(data[f"{sn}_info"]["file_name"]):
                    file_window_ids = np.where(
                        data[f"{sn}_info"]["file_name"] == file_name
                    )[0]
                    logging.info(f"File {file_name}: {file_window_ids.shape[0]}")

        # consider only the specified normal data proportion of each dataset (except test) if relevant
        if self.normal_data_prop < 1:
            for sn in [n for n in non_empty_set_names if n != self.test_set_name]:
                normal_ids = np.where(data[f"y_{sn}"] == 0)[0]
                n_removed = int((1 - self.normal_data_prop) * len(normal_ids))
                removed_ids = np.random.choice(normal_ids, n_removed, replace=False)
                for k in [f"X_{sn}", f"y_{sn}"]:
                    data[k] = np.delete(data[k], removed_ids, axis=0)
                for info_key, info_values in data[f"{sn}_info"].items():
                    data[f"{sn}_info"][info_key] = np.delete(
                        info_values, removed_ids, axis=0
                    )

        # if specified and relevant, augment anomaly classes using the provided strategy
        if self.anomaly_augmentation != "none":
            n_train_samples, window_size, n_features = data[
                f"X_{self.train_set_name}"
            ].shape
            n_train_normal = sum(data[f"y_{self.train_set_name}"] == 0)
            if n_train_normal == n_train_samples:
                raise ValueError(
                    "Cannot augment anomalies without any being in training data."
                )
            if window_size > 1:
                raise ValueError(
                    "Current anomaly augmentation methods only support a window size of 1."
                )
            ano_classes = [
                c for c in np.unique(data[f"y_{self.train_set_name}"]) if c != 0
            ]
            n_per_ano_class = math.ceil(
                (n_train_normal / self.ano_augment_n_per_normal) / len(ano_classes)
            )
            oversampler = AUGMENTATION_TO_CLASS[self.anomaly_augmentation](
                sampling_strategy={c: n_per_ano_class for c in ano_classes},
                random_state=self.ano_augment_seed,
            )
            (
                X_train_augmented,
                data[f"y_{self.train_set_name}"],
            ) = oversampler.fit_resample(
                data[f"X_{self.train_set_name}"].reshape(
                    (n_train_samples, window_size * n_features)
                ),
                data[f"y_{self.train_set_name}"],
            )
            augmented_n_train_samples = X_train_augmented.shape[0]
            data[f"X_{self.train_set_name}"] = X_train_augmented.reshape(
                (augmented_n_train_samples, window_size, n_features)
            )
            # reset window information as some anomalies could be new
            for info_k in data[f"{self.train_set_name}_info"]:
                data[f"{self.train_set_name}_info"][info_k] = np.array(
                    augmented_n_train_samples * [[]]
                )
            # shuffle augmented training windows
            shuffle_datasets(data, [self.train_set_name])

        # if specified, balance classes in the final datasets
        if self.class_balancing != "none":
            sets_to_balance = []
            for sn in non_empty_set_names:
                # only balance datasets that contain more than one class
                if sum(data[f"y_{sn}"] > 0) > 0:
                    sets_to_balance.append(sn)
            for sn in sets_to_balance:
                n_samples, window_size, n_features = data[f"X_{sn}"].shape
                ano_classes = [c for c in np.unique(data[f"y_{sn}"]) if c > 0]
                # samplers are only supported for 2-dimensional arrays
                X_reshaped = data[f"X_{sn}"].reshape(
                    (n_samples, window_size * n_features)
                )
                n_per_class = None
                if self.class_balancing == "naive.ano":
                    # over-sampling of anomaly classes to match the anomaly class
                    # with the largest cardinality (only relevant if multiple types)
                    n_per_class = max([sum(data[f"y_{sn}"] == c) for c in ano_classes])
                elif self.class_balancing == "naive":
                    # over-sampling of anomaly classes and under-sampling of the normal
                    # class, so that all classes have `n / n_classes` samples
                    n_per_class = data[f"X_{sn}"].shape[0] // (1 + len(ano_classes))
                # get over-sampled anomalous examples
                ros = RandomOverSampler(
                    sampling_strategy={c: n_per_class for c in ano_classes},
                    random_state=self.class_balancing_seed,
                )
                X_ano_resampled, y_ano_resampled = ros.fit_resample(
                    X_reshaped, data[f"y_{sn}"]
                )
                ano_mask = y_ano_resampled > 0
                X_ano_resampled, y_ano_resampled = (
                    X_ano_resampled[ano_mask],
                    y_ano_resampled[ano_mask],
                )
                sampled_ano_indices = ros.sample_indices_[ano_mask]
                X_ano_resampled = X_ano_resampled.reshape(
                    (X_ano_resampled.shape[0], window_size, n_features)
                )
                X_normal_resampled = None
                y_normal_resampled = None
                sampled_normal_indices = None
                if self.class_balancing == "naive.ano":
                    # leave normal samples as they are
                    normal_mask = data[f"y_{sn}"] == 0
                    n_normal = sum(normal_mask)
                    X_normal_resampled = data[f"X_{sn}"][normal_mask]
                    y_normal_resampled = np.zeros(n_normal)
                    sampled_normal_indices = np.where(normal_mask)[0]
                elif self.class_balancing == "naive":
                    # get under-sampled normal examples
                    rus = RandomUnderSampler(
                        sampling_strategy={0: n_per_class},
                        random_state=self.class_balancing_seed,
                    )
                    X_normal_resampled, y_normal_resampled = rus.fit_resample(
                        X_reshaped, data[f"y_{sn}"]
                    )
                    normal_mask = y_normal_resampled == 0
                    X_normal_resampled, y_normal_resampled = (
                        X_normal_resampled[normal_mask],
                        y_normal_resampled[normal_mask],
                    )
                    sampled_normal_indices = rus.sample_indices_[normal_mask]
                    # derive resampled dataset
                    X_normal_resampled = X_normal_resampled.reshape(
                        (X_normal_resampled.shape[0], window_size, n_features)
                    )
                data[f"X_{sn}"] = np.concatenate(
                    [X_normal_resampled, X_ano_resampled], axis=0
                )
                data[f"y_{sn}"] = np.concatenate([y_normal_resampled, y_ano_resampled])
                sampled_indices = np.concatenate(
                    [sampled_normal_indices, sampled_ano_indices], axis=0
                )
                for info_k in data[f"{sn}_info"]:
                    data[f"{sn}_info"][info_k] = data[f"{sn}_info"][info_k][
                        sampled_indices
                    ]
            # shuffle back balanced datasets
            shuffle_datasets(data, sets_to_balance)

        for n in non_empty_set_names:
            info_key = f"{n}_info"
            data_keys = [k for k in data if n in k and k != info_key]
            save_files(self.output_path, {info_key: data[info_key]}, "pickle")
            save_files(self.output_path, {k: data[k] for k in data_keys}, "numpy")
        return data

    def get_window_data(
        self,
        periods: np.array,
        periods_labels: np.array,
        periods_info: list,
        sample_ids_offset: int = 0,
        period_ids_offset: int = 0,
        instance_ids_offset: int = 0,
        return_all: bool = False,
        verbose: bool = True,
    ) -> (np.array, np.array, dict):
        """Returns window data and information extracted from `periods` as a dictionary.

        The keys of the returned dictionary are the same as described in `self.get_modeling_split()`,
        except there are no dataset names (samples are returned in a single "dataset"), and the returned
        dictionary has only one level of depth: information keys are returned with an "info_" prefix.

        Note: identifiers at keys "info_window_id", "info_period_id" and "info_ano`i`_instance_id" will
        only be unique within the provided periods. To make them unique at a more global level, id
        offsets can be provided through `window_ids_offset`, `period_ids_offset` and `instance_ids_offset`.

        Note: windows are simply extracted without shuffling, and sample ids reflect the chronological
        extraction order of windows (hence, sorting by window id amounts to sorting by window start time).

        Note: only the provided periods that contain at least one window of records will be considered,
        and period ids will relate to these periods only (as if the shorter ones had not been provided).
        Example for a window size of 2:
        periods = [[23, 2, 84], [54], [18, 12]]
        => period_ids = [0, 1] (for the first and last periods).

        Args:
            periods: `(n_periods, period_length, n_features)`; `period_length` depends on period.
            periods_labels: multiclass labels for each period of shape `(n_periods, period_length)`.
            periods_info: period information lists (one per period).
            sample_ids_offset: offset to add to every window id.
            period_ids_offset: offset to add to every period id.
            instance_ids_offset: offset to add to every anomaly instance id.
            return_all: whether to return every windows, including those labeled as -1.
            verbose (bool): whether to print progress texts to the console.

        Returns:
            The window data and information, with keys as described above and values as
              ndarrays of  shapes `(n_windows,)`.
        """
        # get printing behavior from verbose
        v_print = get_verbose_print(verbose)
        v_print(
            "extracting samples and information from the periods...",
            end=" ",
            flush=True,
        )
        samples, samples_labels, general_info_keys = (
            [],
            [],
            ["sample_id", "period_id", "n_anomalies", "main_ano_id"],
        )
        samples_info_dict = {
            k: []
            for k in general_info_keys
            + list(self.period_info_parsing_func(periods_info[0]).keys())
        }
        # cumulative number of samples and anomaly ranges per period
        periods_sample_cumsum, periods_range_cumsum = [], []
        # positive (anomaly) ranges dicts within periods and samples
        periods_pos_range_dicts, samples_pos_ranges_dicts = [], []
        # index offsets to apply per period and type when deriving unique anomaly instance ids
        periods_type_offset_dicts = []
        # only consider "long enough" periods (containing at least one window of records)
        l_period_ids = [
            i for i, p in enumerate(periods) if p.shape[0] >= self.window_size
        ]
        l_periods, l_periods_labels = (
            periods[l_period_ids],
            periods_labels[l_period_ids],
        )
        l_periods_info = [
            info for i, info in enumerate(periods_info) if i in l_period_ids
        ]
        for period_idx, (period, period_labels, period_info) in enumerate(
            zip(l_periods, l_periods_labels, l_periods_info)
        ):
            # anomaly ranges in the period, grouped by type
            periods_pos_range_dicts.append(extract_multiclass_ranges_ids(period_labels))
            # number of anomaly instances of each type in the period
            period_n_pos_ranges_dict = {
                k: len(v) for k, v in periods_pos_range_dicts[-1].items()
            }
            # derive type offsets for the period
            period_type_offsets_dict, offset = dict(), 0
            for type_, n_ranges in period_n_pos_ranges_dict.items():
                period_type_offsets_dict[type_] = offset
                offset += n_ranges
            periods_type_offset_dicts.append(period_type_offsets_dict)
            # add cumulative number of anomaly ranges of the current period
            prev_n_ranges = (
                0 if len(periods_range_cumsum) == 0 else periods_range_cumsum[-1]
            )
            periods_range_cumsum.append(
                prev_n_ranges + sum([n for n in period_n_pos_ranges_dict.values()])
            )
            # extract sample ranges with the specified size and step
            period_samples_ranges = get_sliding_windows(
                period,
                self.window_size,
                self.window_step,
                include_remainder=True,
                ranges_only=True,
            )
            # add cumulative number of samples of the current period
            prev_n_samples = (
                0 if len(periods_sample_cumsum) == 0 else periods_sample_cumsum[-1]
            )
            periods_sample_cumsum.append(prev_n_samples + len(period_samples_ranges))
            # add samples data and information
            for sample_start_idx, sample_end_idx in period_samples_ranges:
                samples.append(period[sample_start_idx:sample_end_idx])
                # sample information (unique identifier and information about the period it comes from)
                for k, v in zip(
                    ["sample_id", "period_id"],
                    [
                        sample_ids_offset + len(samples) - 1,
                        period_ids_offset + period_idx,
                    ],
                ):
                    samples_info_dict[k].append(v)
                for info_name, info_value in self.period_info_parsing_func(
                    period_info
                ).items():
                    samples_info_dict[info_name].append(info_value)
                # sample anomaly information
                sample_labels = period_labels[sample_start_idx:sample_end_idx]
                # anomaly ranges in the sample, grouped by type
                samples_pos_ranges_dicts.append(
                    extract_multiclass_ranges_ids(sample_labels)
                )
                # number of anomaly ranges in the sample
                samples_info_dict["n_anomalies"].append(
                    sum([len(v) for v in samples_pos_ranges_dicts[-1].values()])
                )
        v_print("done.")
        v_print(
            "completing potential sample anomaly information...", end=" ", flush=True
        )
        # set keys according to the maximum number of anomalies encountered in the samples
        max_anomalies = np.max(samples_info_dict["n_anomalies"])
        ano_info_keys = ["type", "instance_id", "coverage", "start", "end"]
        for ano_idx in range(max_anomalies):
            for k in [f"ano{ano_idx}_{ano_info_key}" for ano_info_key in ano_info_keys]:
                samples_info_dict[k] = self.get_default_ano_info_values(len(samples), k)
        # the method-local "indices" may be different from the global "ids", accounting for offsets
        for sample_idx in range(len(samples)):
            # period the sample was extracted from
            period_idx = samples_info_dict["period_id"][sample_idx] - period_ids_offset
            # number of samples in the previous periods, and sample start index in the period
            n_samples_prev_periods = (
                0 if period_idx == 0 else periods_sample_cumsum[period_idx - 1]
            )
            sample_start_idx = self.window_step * (sample_idx - n_samples_prev_periods)
            # number of anomaly ranges in the previous periods, and type offsets to apply for the period
            n_ranges_prev_periods = (
                0 if period_idx == 0 else periods_range_cumsum[period_idx - 1]
            )
            period_type_offsets_dict = periods_type_offset_dicts[period_idx]
            # potential case of samples extracted with a step larger than one (`include_remainder=True`)
            period_len = len(l_periods[period_idx])
            if sample_start_idx + self.window_size > period_len:
                sample_start_idx = period_len - self.window_size
            # anomaly ranges in the period and in the sample, grouped by type
            period_pos_ranges_dict = periods_pos_range_dicts[period_idx]
            sample_pos_ranges_dict = samples_pos_ranges_dicts[sample_idx]
            sample_ano_id = 0
            for sample_ano_type, sample_ano_ranges in sample_pos_ranges_dict.items():
                # fill anomaly information for each range in the sample
                for sample_ano_start, sample_ano_end in sample_ano_ranges:
                    for k, v in zip(
                        ["type", "start", "end"],
                        [sample_ano_type, sample_ano_start, sample_ano_end],
                    ):
                        samples_info_dict[f"ano{sample_ano_id}_{k}"][sample_idx] = v
                    # anomaly instance id and amount of its achievable coverage achieved by the sample
                    ano_instance_id = -1
                    total_ano_length, sample_ano_length = (
                        0,
                        sample_ano_end - sample_ano_start,
                    )
                    for i, (period_ano_start, period_ano_end) in enumerate(
                        period_pos_ranges_dict[sample_ano_type]
                    ):
                        # start is inclusive, end is exclusive
                        if (
                            period_ano_start
                            <= sample_start_idx + sample_ano_start
                            < period_ano_end
                        ):
                            # the instance id is unique overall (across periods, anomaly types, and type range ids)
                            ano_instance_id = (
                                n_ranges_prev_periods
                                + period_type_offsets_dict[sample_ano_type]
                                + i
                            )
                            total_ano_length = period_ano_end - period_ano_start
                            break
                    samples_info_dict[f"ano{sample_ano_id}_instance_id"][sample_idx] = (
                        instance_ids_offset + ano_instance_id
                    )
                    # achievable anomaly coverage given the sample size
                    achievable_coverage = min(self.window_size / total_ano_length, 1)
                    # set proportion of achievable coverage achieved by the sample
                    samples_info_dict[f"ano{sample_ano_id}_coverage"][sample_idx] = (
                        sample_ano_length / total_ano_length
                    ) / achievable_coverage
                    sample_ano_id += 1
            # derive "main" sample label and relevant anomaly range id from its anomaly information
            sample_label, main_ano_id = self.get_sample_label(
                samples_info_dict, sample_idx
            )
            samples_labels.append(sample_label)
            samples_info_dict["main_ano_id"].append(main_ano_id)
        v_print("done.")
        # turn all sample-wise elements to ndarrays, optionally removing elements for which labels are -1
        samples_labels = np.array(samples_labels, dtype=np.int32)
        samples_mask = (
            np.repeat(True, len(samples_labels))
            if return_all
            else (samples_labels > -1)
        )
        samples, samples_labels = (
            np.array(samples, dtype=np.float32)[samples_mask],
            samples_labels[samples_mask],
        )
        samples_info_dict = {
            k: np.array(v)[samples_mask] for k, v in samples_info_dict.items()
        }
        return samples, samples_labels, samples_info_dict

    def get_default_ano_info_values(self, n_samples, ano_info_key):
        """Returns an ndarray of `n_samples` default anomaly information values for `ano_info_key`.

        Args:
            n_samples (int): number of default anomaly information values to return.
            ano_info_key (str): anomaly information key prefixed by "ano`i`_", defining
                the default value and data type.

        Returns:
            ndarray: the ndarray of sample-wise default anomaly information values, of shape `(n_samples,)`.
        """
        pruned_info_key = re.sub(r"ano\d+_", "", ano_info_key)
        default_items = np.empty(
            n_samples, dtype=self.ano_info_types_dict[pruned_info_key]
        )
        default_items[:] = 0 if pruned_info_key in self.zero_defaulting_keys else -1
        return default_items

    def get_sample_label(self, samples_info_dict, sample_idx):
        """Returns the integer type and index of the "main" anomaly range of the sample at
            index `sample_idx` in all values of `samples_info_dict`.

        The "main" anomaly range of a sample is defined as the range for which this sample achieves
        the largest achievable coverage given its size, or as its latest range in case of equality.

        A sample that does not contain any anomaly range is returned with type 0 (for "normal").
        A sample that only contains ranges of an anomaly type to ignore is returned with type -1,
        indicating that the sample should be dropped.

        If the maximum coverage achieved by an anomalous sample is below `self.min_ano_coverage`,
        it will be deemed "weakly anomalous", and either returned:
        - To be dropped, with a label of -1, if `self.weak_ano_policy` is "drop".
        - As normal, with a label of 0, if `self.weak_ano_policy` is "keep".

        Args:
            samples_info_dict (dict): samples information, as described in `self.get_modeling_split()`.
            sample_idx (int): index of the sample of interest in the (ndarray) values of `samples_info_dict`.

        Returns:
            int, int: the integer type and index of the sample's main anomaly range.
        """
        n_anomalies = samples_info_dict["n_anomalies"][sample_idx]
        if n_anomalies == 0:
            # type 0 for "normal", index -1 for "not-applicable"
            return 0, -1
        # get the index of the main anomaly range of the sample
        ano_idx, max_coverage = -1, -np.inf
        for i in range(n_anomalies):
            ano_coverage = samples_info_dict[f"ano{i}_coverage"][sample_idx]
            if (
                samples_info_dict[f"ano{i}_type"][sample_idx]
                not in self.dropped_anomaly_types
            ):
                if ano_coverage > max_coverage:
                    ano_idx, max_coverage = i, ano_coverage
                # in case of coverage proportion equality, the main anomaly is set as the latest one
                elif ano_coverage == max_coverage:
                    old_start_idx = samples_info_dict[f"ano{ano_idx}_start"][sample_idx]
                    new_start_idx = samples_info_dict[f"ano{i}_start"][sample_idx]
                    if new_start_idx > old_start_idx:
                        ano_idx, max_coverage = i, ano_coverage
        if ano_idx == -1:
            # the sample only contains anomaly types to ignore
            return -1, -1
        # return label according to the minimum achievable coverage and weakly anomalous samples policy
        if max_coverage >= self.window_min_ano_coverage:
            return int(samples_info_dict[f"ano{ano_idx}_type"][sample_idx]), ano_idx
        return (0 if self.window_weak_ano_policy == "keep" else -1), -1


def shuffle_datasets(datasets, set_names):
    """Shuffles the provided datasets (keys described in `WindowManager.get_modeling_split()`) inplace."""
    for n in set_names:
        mask = np.random.permutation(datasets[f"X_{n}"].shape[0])
        for set_k in [k for k in datasets if n in k]:
            if set_k != f"{n}_info":
                datasets[set_k] = datasets[set_k][mask]
            else:
                for info_k in datasets[set_k]:
                    datasets[set_k][info_k] = datasets[set_k][info_k][mask]
