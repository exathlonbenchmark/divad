"""ASD-specific data management module.
"""
import os
import pickle
import logging
from typing import Union, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils.guarding import check_is_percentage
from data.managers.base_manager import BaseManager
from detection.detectors.helpers.general import get_parsed_integer_list_str
from detection.metrics.helpers import extract_binary_ranges_ids


def load_app_data(
    input_path: str, app_id: Union[int, str]
) -> (NDArray[np.float32], NDArray[np.float32]):
    X = []
    y = []
    for sn in ["train", "test"]:
        X.append(
            pickle.load(open(os.path.join(input_path, f"omi-{app_id}_{sn}.pkl"), "rb"))
        )
        if sn == "train":
            y.append(np.zeros(X[0].shape[0]))
        else:
            y.append(
                pickle.load(
                    open(os.path.join(input_path, f"omi-{app_id}_{sn}_label.pkl"), "rb")
                )
            )
    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0).astype(np.float32)
    return X, y


def get_parsed_app_ids(app_ids: Union[int, str] = "") -> List[int]:
    if isinstance(app_ids, int):
        app_ids = str(app_ids)
    app_ids = get_parsed_integer_list_str(app_ids)
    return app_ids


class AsdManager(BaseManager):
    """ASD-specific data management class.

    Args:
        app_ids: applications to consider in the pipeline.
        test_app_ids: applications to use as a test set (among the ones considered).
        val_prop: proportion of the training data to use as validation.
        **base_kwargs: keyword arguments of `BaseManager`.
    """

    def __init__(
        self,
        app_ids: Union[int, str] = "",
        test_app_ids: Union[int, str] = "1",
        val_prop: float = 0.15,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        app_ids = get_parsed_app_ids(app_ids)
        test_app_ids = get_parsed_app_ids(test_app_ids)
        if len(app_ids) == 0:
            app_ids = list(range(1, 13))
        if len(test_app_ids) == 0:
            raise ValueError("At least one application must be used as test data.")
        for a in test_app_ids:
            if a not in app_ids:
                raise ValueError(
                    f"All test applications must be in `app_ids`. Received {a} not in {app_ids}."
                )
        check_is_percentage(val_prop, "val_prop")
        self.app_ids = app_ids
        self.test_app_ids = test_app_ids
        self.val_prop = val_prop

    def _load_raw_sequences(self, input_path: str) -> (List[pd.DataFrame], List[list]):
        """ASD implementation.

        - Loaded sequences are the application traces.
        - Sequences information are initialized in the form `[file_name]` (without the file extension).
        """
        seq_dfs = []
        seqs_info = []
        for app_id in self.app_ids:
            X, y = load_app_data(input_path, app_id)
            app_df = pd.DataFrame(
                X, columns=[f"Ft {i}" for i in range(1, X.shape[1] + 1)]
            )
            app_df["anomaly"] = y
            # set dummy index with a 5-minute spacing
            start = pd.to_datetime("2024-01-01 09:00:00")
            index = pd.date_range(start, periods=X.shape[0], freq="5min")
            app_df = app_df.set_index(index)
            seq_dfs.append(app_df)
            seqs_info.append([f"omi-{app_id}"])
        return seq_dfs, seqs_info

    def _get_preprocessed_sequences(
        self, seq_dfs: List[pd.DataFrame], seqs_info: List[list]
    ) -> List[pd.DataFrame]:
        """Spark implementation.

        Prunes sequences based a previous analysis.
        """
        if any([df.isnull().any().any() for df in seq_dfs]):
            raise ValueError("There should be no NaN values in traces.")
        return seq_dfs

    def _get_sequence_datasets(
        self,
        seq_dfs: List[pd.DataFrame],
        seqs_info: List[list],
        train_set_name: str = "train",
        val_set_name: str = "val",
        test_set_name: str = "test",
    ) -> (dict, dict):
        """ASD implementation.

        Sequences information are returned in the form `[file_name, period_rank]`, where
        `period_rank` refers to the chronological rank of the period in its original sequence.
        """
        seq_dfs = np.array(seq_dfs, dtype=object)
        seqs_info = np.array(seqs_info, dtype=object)
        set_to_seqs = dict()
        set_to_info = dict()

        # training and test sequences and information
        test_mask = np.array(
            [int(info[0][4:]) in self.test_app_ids for info in seqs_info]
        )
        train_seqs = seq_dfs[~test_mask]
        train_info = seqs_info[~test_mask]
        test_seqs = seq_dfs[test_mask]
        test_info = seqs_info[test_mask]
        set_to_seqs[test_set_name] = list(test_seqs)
        set_to_info[test_set_name] = [[*info, 0] for info in test_info]

        # logging of test anomaly statistics
        logging.info("Test Sequences:")
        test_ano_lengths = []
        for seq, info in zip(set_to_seqs[test_set_name], set_to_info[test_set_name]):
            ano_ranges = extract_binary_ranges_ids(seq.anomaly.values)
            ano_lengths = [e - s for s, e in ano_ranges]
            n_ano_ranges = len(ano_lengths)
            n_ano_points = np.sum(ano_lengths)
            mean_ano_length = np.mean(ano_lengths)
            min_ano_length = np.min(ano_lengths)
            max_ano_length = np.max(ano_lengths)
            length_message = f"lengths: mean={mean_ano_length:.2f} (min={min_ano_length}, max={max_ano_length})"
            logging.info(
                f"Sequence {info[0]} anomalies: {n_ano_ranges} ranges - {n_ano_points} points - {length_message}."
            )
            test_ano_lengths += ano_lengths
        test_ano_lengths_median = np.median(test_ano_lengths)
        test_ano_lengths_mean = np.mean(test_ano_lengths)
        test_ano_lengths_std = np.std(test_ano_lengths)
        test_ano_lengths_min = np.min(test_ano_lengths)
        test_ano_lengths_max = np.max(test_ano_lengths)
        length_message = (
            f"median={test_ano_lengths_median} - mean={test_ano_lengths_mean:.2f} +/- {test_ano_lengths_std:.2f} "
            f"(min={test_ano_lengths_min}, max={test_ano_lengths_max})"
        )
        logging.info(f"All lengths: {length_message}.")

        # train/validation split
        logging.info("Train/validation Sequences:")
        if self.val_prop == 0:
            set_to_seqs[train_set_name] = list(train_seqs)
            set_to_info[train_set_name] = [[*info, 0] for info in train_info]
        else:
            set_to_seqs[train_set_name] = []
            set_to_info[train_set_name] = []
            set_to_seqs[val_set_name] = []
            set_to_info[val_set_name] = []
            percents_ano_ranges = []
            percents_ano_points = []
            for seq, info in zip(train_seqs, train_info):
                train_size = seq.shape[0] - int(self.val_prop * seq.shape[0])
                train_seq = seq.iloc[:train_size]
                set_to_seqs[train_set_name].append(train_seq)
                set_to_info[train_set_name].append([*info, 0])
                val_seq = seq.iloc[train_size:]
                set_to_seqs[val_set_name].append(val_seq)
                set_to_info[val_set_name].append([*info, 1])
                # logging of training and validation anomaly statistics
                train_ano_ranges = extract_binary_ranges_ids(train_seq.anomaly.values)
                val_ano_ranges = extract_binary_ranges_ids(val_seq.anomaly.values)
                n_train_ano_ranges = len(train_ano_ranges)
                n_val_ano_ranges = len(val_ano_ranges)
                tot_ano_ranges = n_train_ano_ranges + n_val_ano_ranges
                n_train_ano_points = sum([e - s for s, e in train_ano_ranges])
                n_val_ano_points = sum([e - s for s, e in val_ano_ranges])
                tot_ano_points = n_train_ano_points + n_val_ano_points
                percent_train_ano_ranges = 100 * n_train_ano_ranges / tot_ano_ranges
                percent_train_ano_points = 100 * n_train_ano_points / tot_ano_points
                percents_ano_ranges.append(percent_train_ano_ranges)
                percents_ano_points.append(percent_train_ano_points)
                range_message = (
                    f"{n_train_ano_ranges} ranges ({percent_train_ano_ranges:.2f}%)"
                )
                point_message = (
                    f"{n_train_ano_points} points ({percent_train_ano_points:.2f}%)"
                )
                logging.info(
                    f"Sequence {info[0]} training anomalies: {range_message} - {point_message}."
                )
            range_message = f"{np.mean(percents_ano_ranges):.2f}% +/- {np.std(percents_ano_ranges):.2f}% ranges"
            point_message = f"{np.mean(percents_ano_points):.2f}% +/- {np.std(percents_ano_points):.2f}% points"
            logging.info(
                f"Average training anomalies: {range_message} - {point_message}."
            )
        return set_to_seqs, set_to_info
