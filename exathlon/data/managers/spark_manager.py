"""Spark-specific data management module.
"""

import os
import logging
from itertools import compress
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.guarding import (
    check_is_percentage,
    check_value_in_choices,
    check_all_values_in_choices,
)
from utils.spark.metadata import (
    APP_IDS,
    TRACE_TO_BATCH_INTERVAL,
    TRACE_TO_N_RUNNING_EXECUTORS,
    TRACE_TO_MAX_EXEC_MEMORY,
    TRACE_TO_RENAMED,
    ANOMALY_TYPES,
    TRACE_TYPES,
    REMOVAL_OPTIONS,
    PRUNING_OPTIONS,
    USED_FEATURES,
)
from data.helpers import save_files
from data.managers.base_manager import BaseManager
from detection.metrics.helpers import (
    extract_binary_ranges_ids,
    extract_multiclass_ranges_ids,
)
from detection.detectors.helpers.general import get_parsed_integer_list_str


def get_renamed_trace(seq_name: str) -> str:
    return TRACE_TO_RENAMED[seq_name] if seq_name in TRACE_TO_RENAMED else seq_name


class SparkManager(BaseManager):
    """Spark-specific data management class.

    Args:
        setup: experimental setup (either "unsupervised", "weakly" or "generalization").
        app_ids: application ids, as an empty string for all applications, an integer for a single
          application, or a string of space-separated integers for multiple applications.
        trace_types: restricted trace types to consider as a string of space-separate trace types
          (empty for no restriction).
        label_as_unknown: anomalies to label as "unknown" (either "none" or dot-separated values in
         ("os_only", "integer anomaly type") where "os_only" refers to anomalies that had no effect on Spark
         metrics): neither positive nor negative predictions are penalized.
        include_extended_effect: whether to include the "extended effect interval" when labeling anomalies.
        trace_removal_idx: trace removal index in `utils.spark.metadata.REMOVAL_OPTIONS`.
        data_pruning_idx: data pruning index in `utils.spark.metadata.PRUNING_OPTIONS`.
        val_prop: proportion of data to set as validation sequences.
        test_prop: proportion of each disturbed trace to send to test if `setup` is "generalization"
          (time split).
        train_val_split: type of train/validation split to perform if `val_prop` is greater than
          zero (either "time" or "random"), only relevant for "unsupervised" and "weakly" setups.
        random_min_window_size: minimum subsequence length to ensure within a trace when extracting
          a random validation subsequence from it.
        random_split_seed: seed to use when extracting random validation subsequences from traces.
        generalization_min_window_size: minimum subsequence length to ensure within a trace, and
          of normal data before a test anomaly, in the "generalization" setup.
        **base_kwargs: keyword arguments of `BaseManager`.
    """

    def __init__(
        self,
        setup: str = "unsupervised",
        app_ids: Union[int, str] = "",
        trace_types: str = "",
        label_as_unknown: str = "os_only",
        include_extended_effect: bool = True,
        trace_removal_idx: int = 0,
        data_pruning_idx: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.4,
        train_val_split: str = "time",
        random_min_window_size: int = 120,
        random_split_seed: int = 0,
        generalization_min_window_size: int = 120,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        check_value_in_choices(
            setup, "setup", ["unsupervised", "weakly", "generalization"]
        )
        if len(trace_types) == 0:
            # important to copy as can be modified later
            trace_types = TRACE_TYPES[:]
        else:
            trace_types = trace_types.split(" ")
            # remove any duplicates
            trace_types = list(dict.fromkeys(trace_types))
        if isinstance(app_ids, int):
            app_ids = str(app_ids)
        app_ids = get_parsed_integer_list_str(app_ids)
        if len(app_ids) == 0:
            app_ids = APP_IDS
        check_all_values_in_choices(app_ids, "app_ids", APP_IDS)
        check_all_values_in_choices(trace_types, "trace_types", TRACE_TYPES)
        label_as_unknown = label_as_unknown.split(".")
        if len(label_as_unknown) > 1 and "none" in label_as_unknown:
            raise ValueError(
                'If "none" is provided in `label_as_unknown`, it should be alone.'
            )
        if "os_only" in label_as_unknown and "4" in label_as_unknown:
            raise ValueError(
                '"os_only" in `label_as_unknown` only makes sense if not '
                "labeling all CPU contention as unknown."
            )
        for i, label in enumerate(label_as_unknown):
            check_value_in_choices(
                label,
                f"label_as_unknown[{i}]",
                ["none", "os_only"] + list(map(lambda x: str(x), range(7))),
            )
        check_is_percentage(val_prop, "val_prop")
        check_is_percentage(test_prop, "test_prop")
        check_value_in_choices(train_val_split, "train_val_split", ["time", "random"])
        self.setup = setup
        self.app_ids = app_ids
        if setup == "generalization":
            # the "generalization" setup does not use undisturbed traces
            try:
                trace_types.remove("undisturbed")
            except ValueError:
                pass
        self.trace_types = trace_types
        self.label_as_unknown = label_as_unknown
        self.include_extended_effect = include_extended_effect
        self.removed_traces = REMOVAL_OPTIONS[trace_removal_idx]
        self.seq_name_to_pruned_ranges = PRUNING_OPTIONS[data_pruning_idx]
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.train_val_split = train_val_split
        if setup in ["unsupervised", "weakly"]:
            self.min_window_size = random_min_window_size
        elif setup == "generalization":
            self.min_window_size = generalization_min_window_size
        else:
            self.min_window_size = None
        self.random_split_seed = random_split_seed

    def _load_raw_sequences(self, input_path: str) -> (List[pd.DataFrame], List[list]):
        """Spark implementation.

        - Loaded sequences are the application(s) traces.
        - Sequences information are initialized in the form `[seq_name, trace_type]`.
        """
        seq_dfs = []
        seqs_info = []
        logging.info("Loading ground-truth table...")
        gt_table = pd.read_csv(os.path.join(input_path, "ground_truth.csv"))
        # convert timestamps to datetime format
        for c in ["root_cause_start", "root_cause_end", "extended_effect_end"]:
            gt_table[c] = pd.to_datetime(gt_table[c], unit="s")
        logging.info("Done.")

        # select relevant application keys
        app_keys = [f"app{app_id}" for app_id in self.app_ids]

        # load traces of the selected application(s) and type(s)
        for app_key in app_keys:
            logging.info(f'Loading traces of application {app_key.replace("app", "")}.')
            app_path = os.path.join(input_path, app_key)
            file_names = [
                fn for fn in os.listdir(app_path) if fn[:-4] not in self.removed_traces
            ]
            for trace_type in self.trace_types:
                type_file_names = [
                    fn
                    for fn in file_names
                    if int(fn.split("_")[1]) == TRACE_TYPES.index(trace_type)
                ]
                if len(type_file_names) > 0:
                    logging.info(f'Loading {trace_type.replace("_", " ")} traces.')
                    for fn in type_file_names:
                        logging.info(f"Loading {fn}...")
                        seq_dfs.append(
                            load_trace(
                                os.path.join(app_path, fn), usecols=USED_FEATURES
                            )
                        )
                        logging.info("Done.")
                        seqs_info.append([fn[:-4], trace_type])
        if len(seq_dfs) == 0:
            raise ValueError("No traces for the provided application(s) and type(s).")
        # add "anomaly" column
        self._add_anomaly_column(seq_dfs, seqs_info, gt_table)
        return seq_dfs, seqs_info

    def _get_preprocessed_sequences(
        self, seq_dfs: List[pd.DataFrame], seqs_info: List[list]
    ) -> List[pd.DataFrame]:
        """Spark implementation.

        Prunes sequences based a previous analysis.
        """
        if any([df.isnull().any().any() for df in seq_dfs]):
            raise ValueError("There should be no NaN values left in traces.")
        self._prune_sequences(seq_dfs, seqs_info)
        return seq_dfs

    def _get_sequence_datasets(
        self,
        seq_dfs: List[pd.DataFrame],
        seqs_info: List[list],
        train_set_name: str = "train",
        val_set_name: str = "val",
        test_set_name: str = "test",
    ) -> (dict, dict):
        """Spark implementation.

        Each item in `seqs_info` is assumed to be of the form `[file_name, trace_type]`. It will however
          be returned in the form `[file_name, trace_type, seq_rank]`. Where `seq_rank` is the
          chronological rank of the sequence in its trace file.
        """
        np.random.seed(self.random_split_seed)
        set_to_seqs = dict()
        set_to_info = dict()

        # undisturbed and disturbed sequence ids
        undisturbed_mask = [info[1] == "undisturbed" for info in seqs_info]
        disturbed_mask = [not i for i in undisturbed_mask]
        undisturbed_seqs = list(compress(seq_dfs, undisturbed_mask))
        undisturbed_info = list(compress(seqs_info, undisturbed_mask))
        disturbed_seqs = list(compress(seq_dfs, disturbed_mask))
        disturbed_info = list(compress(seqs_info, disturbed_mask))

        if self.setup == "unsupervised":
            # send all disturbed traces to the test set
            set_to_seqs[test_set_name] = disturbed_seqs
            set_to_info[test_set_name] = [[*info, 0] for info in disturbed_info]

            # split undisturbed traces into training and validation based on the strategy
            if self.val_prop == 0:
                set_to_seqs[train_set_name] = undisturbed_seqs
                set_to_info[train_set_name] = [[*info, 0] for info in undisturbed_info]
            else:
                set_to_seqs[train_set_name] = []
                set_to_info[train_set_name] = []
                set_to_seqs[val_set_name] = []
                set_to_info[val_set_name] = []
                for seq, info in zip(undisturbed_seqs, undisturbed_info):
                    (
                        train_seqs,
                        train_seqs_info,
                        val_seqs,
                        val_seqs_info,
                    ) = self._get_undisturbed_seq_split(seq, info)
                    set_to_seqs[train_set_name] += train_seqs
                    set_to_info[train_set_name] += train_seqs_info
                    set_to_seqs[val_set_name] += val_seqs
                    set_to_info[val_set_name] += val_seqs_info
        elif self.setup == "weakly":
            # manual selection of training and test sequences
            test_undisturbed_seq_names = ["4_0_100000_32"]
            train_disturbed_seq_names = [
                "1_4_1000000_80",
                "2_5_1000000_87",
                "3_2_500000_70",
                "3_2_1000000_71",
                "4_1_100000_61",
                "5_5_1000000_91",
                "6_1_500000_65",
                "9_3_500000_74",
                "10_4_1000000_79",
            ]
            # Without applications 7 and 8, nor "unknown" and (the 3 remaining) "os_only" anomalies:
            # - total available anomalies: 29 T1, 7 T2, 10 T3, 17 T4, 8 T5, 9 T6
            # - remaining anomalies in test: 15 T1, 5 T2, 6 T3, 7 T4, 5 T5, 5 T6
            # - anomalies removed from test: 14 T1, 2 T2, 4 T3, 10 T4, 3 T5, 4 T6
            if self.train_val_split == "random":
                # anomaly range indices in chronological order in the trace (no matter the type)
                # train anomalies: 8 T1, 2 T2, 2 T3, 6 T4, 2 T5, 2 T6
                # validation anomalies: 6/14 T1, 0/2 T2, 2/4 T3, 4/10 T4, 1/3 T5, 2/4 T6
                disturbed_to_val_range_ids = {
                    "1_4_1000000_80": [2, 3],  # 2/5 T4
                    "2_5_1000000_87": [1, 2],  # 1/2 T5, 1/2 T6
                    "4_1_100000_61": [3, 4, 5, 6],  # 4/9 T1
                    "5_5_1000000_91": [1],  # 0/1 T5, 1/2 T6
                    "6_1_500000_65": [2, 3],  # 2/5 T1
                    "9_3_500000_74": [1, 2],  # 2/4 T3
                    "10_4_1000000_79": [2, 3],  # 2/5 T4
                }
            else:
                # train anomalies: 8 T1, 2 T2, 2 T3, 6 T4, 2 T5, 3 T6
                # validation anomalies: 6/14 T1, 0/2 T2, 2/4 T3, 4/10 T4, 1/3 T5, 1/4 T6
                # we prefer to remove 1 T6 anomalies from val to keep normal data from trace 91 in train
                disturbed_to_val_range_ids = {
                    "1_4_1000000_80": [3, 4, 5],  # 4 is "os_only": does not count
                    "2_5_1000000_87": [3],
                    "4_1_100000_61": [5, 6, 7, 8],
                    "5_5_1000000_91": [2],
                    "6_1_500000_65": [3, 4],
                    "9_3_500000_74": [2, 3],
                    "10_4_1000000_79": [3, 4, 5],  # 4 is "os_only": does not count
                }
            assert all(
                [
                    k in train_disturbed_seq_names
                    for k in disturbed_to_val_range_ids.keys()
                ]
            )
            set_to_seqs[train_set_name] = []
            set_to_info[train_set_name] = []
            if self.val_prop > 0:
                set_to_seqs[val_set_name] = []
                set_to_info[val_set_name] = []
            set_to_seqs[test_set_name] = []
            set_to_info[test_set_name] = []
            # undisturbed traces
            for seq, info in zip(undisturbed_seqs, undisturbed_info):
                seq_name = info[0]
                if seq_name in test_undisturbed_seq_names:
                    # send the whole sequence to the test set
                    set_to_seqs[test_set_name].append(seq)
                    set_to_info[test_set_name].append([*info, 0])
                else:
                    # perform automated train/validation split based on the specified strategy
                    if self.val_prop == 0:
                        set_to_seqs[train_set_name].append(seq)
                        set_to_info[train_set_name].append([*info, 0])
                    else:
                        (
                            train_seqs,
                            train_seqs_info,
                            val_seqs,
                            val_seqs_info,
                        ) = self._get_undisturbed_seq_split(seq, info)
                        set_to_seqs[train_set_name] += train_seqs
                        set_to_info[train_set_name] += train_seqs_info
                        set_to_seqs[val_set_name] += val_seqs
                        set_to_info[val_set_name] += val_seqs_info
            # disturbed traces
            for seq, info in zip(disturbed_seqs, disturbed_info):
                seq_name = info[0]
                if seq_name not in train_disturbed_seq_names:
                    # send the whole sequence to the test set
                    set_to_seqs[test_set_name].append(seq)
                    set_to_info[test_set_name].append([*info, 0])
                elif self.val_prop == 0:
                    # send the whole sequence to the train set
                    set_to_seqs[train_set_name].append(seq)
                    set_to_info[train_set_name].append([*info, 0])
                else:
                    # manual train/validation split based on the specified strategy
                    if seq_name in disturbed_to_val_range_ids.keys():
                        val_range_ids = disturbed_to_val_range_ids[seq_name]
                        # consider ranges of every type sorted by start index
                        ano_type_to_ranges = extract_multiclass_ranges_ids(
                            seq.anomaly.values
                        )
                        ano_ranges = sorted(
                            [
                                r
                                for ranges in ano_type_to_ranges.values()
                                for r in ranges
                            ],
                            key=lambda r: r[0],
                        )
                        val_rank = 1
                        val_first = False
                        val_last = False
                        if 0 in val_range_ids:
                            val_start = 0
                            val_rank = 0
                            val_first = True
                        else:
                            # the first anomaly of the sequence should not be in validation
                            first_val_range_start = ano_ranges[val_range_ids[0]][0]
                            last_train_range_end = ano_ranges[val_range_ids[0] - 1][1]
                            # start validation sequence between the last train anomaly and first val anomaly
                            val_start = int(
                                (last_train_range_end + first_val_range_start) / 2
                            )
                        if len(ano_ranges) - 1 in val_range_ids:
                            if val_first:
                                raise ValueError(
                                    f"Encountered validation spanning the whole range for seq_name={seq_name}."
                                )
                            val_end = seq.shape[0]
                            val_last = True
                        else:
                            # the last anomaly of the sequence should not be in validation
                            last_val_range_end = ano_ranges[val_range_ids[-1]][1]
                            first_train_range_start = ano_ranges[val_range_ids[-1] + 1][
                                0
                            ]
                            # end validation sequence between the last val anomaly and first train anomaly
                            val_end = int(
                                (last_val_range_end + first_train_range_start) / 2
                            )
                        set_to_seqs[val_set_name].append(seq.iloc[val_start:val_end])
                        set_to_info[val_set_name].append([*info, val_rank])
                        if val_first:
                            set_to_seqs[train_set_name].append(seq.iloc[val_end:])
                            set_to_info[train_set_name].append([*info, 1])
                        elif val_last:
                            set_to_seqs[train_set_name].append(seq.iloc[:val_start])
                            set_to_info[train_set_name].append([*info, 0])
                        else:
                            # the validation subsequence is in the middle of the sequence
                            set_to_seqs[train_set_name].append(seq.iloc[:val_start])
                            set_to_info[train_set_name].append([*info, 0])
                            set_to_seqs[train_set_name].append(seq.iloc[val_end:])
                            set_to_info[train_set_name].append([*info, 2])
                    else:
                        # special cases: T2 traces (having a single anomaly range)
                        assert seq_name in ["3_2_500000_70", "3_2_1000000_71"]
                        if self.train_val_split == "time":
                            # send the whole sequence to the training set
                            set_to_seqs[train_set_name].append(seq)
                            set_to_info[train_set_name].append([*info, 0])
                        else:
                            # send a random `val_prop` proportion of normal data to the validation set
                            try:
                                ano_start = np.where(seq.anomaly.values == 2.0)[0][0]
                            except IndexError:  # labeled as unknown
                                assert "2" in self.label_as_unknown
                                ano_start = np.where(seq.anomaly.values == 7.0)[0][0]
                            val_start, val_end = self._get_random_val_interval(
                                ano_start, seq_name, val_prop=self.val_prop
                            )
                            if val_start == 0:
                                set_to_seqs[train_set_name].append(seq.iloc[val_end:])
                                set_to_info[train_set_name].append([*info, 1])
                                val_rank = 0
                            elif val_end == seq.shape[0]:
                                set_to_seqs[train_set_name].append(seq.iloc[:val_start])
                                set_to_info[train_set_name].append([*info, 0])
                                val_rank = 1
                            else:
                                set_to_seqs[train_set_name].append(seq.iloc[:val_start])
                                set_to_info[train_set_name].append([*info, 0])
                                set_to_seqs[train_set_name].append(seq.iloc[val_end:])
                                set_to_info[train_set_name].append([*info, 2])
                                val_rank = 1
                            set_to_seqs[val_set_name].append(
                                seq.iloc[val_start:val_end]
                            )
                            set_to_info[val_set_name].append([*info, val_rank])
        elif self.setup == "generalization":
            # perform a time split of disturbed traces, removing train and validation anomalies
            set_to_seqs[train_set_name] = []
            set_to_info[train_set_name] = []
            if self.val_prop > 0:
                set_to_seqs[val_set_name] = []
                set_to_info[val_set_name] = []
            set_to_seqs[test_set_name] = []
            set_to_info[test_set_name] = []
            for seq, info in zip(disturbed_seqs, disturbed_info):
                # get ranges of every type sorted by start index
                ano_type_to_ranges = extract_multiclass_ranges_ids(seq.anomaly.values)
                ano_ranges = sorted(
                    [r for ranges in ano_type_to_ranges.values() for r in ranges],
                    key=lambda r: r[0],
                )
                # perform train/test split
                test_start = int((1 - self.test_prop) * seq.shape[0])
                # pull test back to include overlapping anomalies (would not be used in train)
                # (also ensure that at least `self.min_window_size` of normal data precedes the
                # first anomaly in test)
                for s, e in ano_ranges:
                    if s <= test_start < e or 0 < s - test_start < self.min_window_size:
                        test_start = s - self.min_window_size
                        break
                if test_start < self.min_window_size:
                    raise ValueError(
                        f"Encountered train_size={test_start} for seq_name={info[0]} "
                        f"(below minimum window size of {self.min_window_size})."
                    )
                elif seq.shape[0] - test_start < self.min_window_size:
                    raise ValueError(
                        f"Encountered test_size={seq.shape[0] - test_start} for seq_name={info[0]} "
                        f"(below minimum window size of {self.min_window_size})."
                    )
                train_seq = seq.iloc[:test_start]
                test_seq = seq.iloc[test_start:]
                # remove anomalies from the training sequence (only keeping large enough subsequences)
                # anomalies starting in the training sequence
                train_ano_ranges = [[s, e] for s, e in ano_ranges if s < test_start]
                train_starts = [0] + [e for s, e in train_ano_ranges]
                train_ends = [s for s, e in train_ano_ranges] + [test_start]
                # if the anomaly ends after the training sequence, then e - s < 0: not included
                train_seqs = [
                    train_seq.iloc[s:e]
                    for s, e in zip(train_starts, train_ends)
                    if e - s >= self.min_window_size
                ]
                train_seqs_info = [[*info, i] for i in range(len(train_seqs))]
                if self.val_prop > 0:
                    # perform train/validation time split
                    relevant_train_size = sum([seq.shape[0] for seq in train_seqs])
                    val_size = int(self.val_prop * relevant_train_size)
                    if val_size < self.min_window_size:
                        raise ValueError(
                            f"Validation prop resulted in val_size={val_size} for seq_name={info[0]} "
                            f"(below minimum window size of {self.min_window_size})."
                        )
                    val_seqs = []
                    val_seqs_info = []
                    remaining_val_size = val_size
                    removed_train_ids = []
                    for i in reversed(range(len(train_seqs))):
                        train_seq = train_seqs[i]
                        train_info = train_seqs_info[i]
                        if (
                            train_seq.shape[0] - remaining_val_size
                            >= self.min_window_size
                        ):
                            # the training subsequence is large enough to take a subset only as validation
                            val_seqs.append(train_seq.iloc[remaining_val_size:])
                            train_seqs[i] = train_seqs[i].iloc[:remaining_val_size]
                            val_seqs_info.append([*info, train_info[-1] + 1])
                            # we could take all the validation data needed
                            remaining_val_size = 0
                        else:
                            # it is not large enough: need to take the whole training subsequence
                            val_seqs.append(train_seq)
                            val_seqs_info.append(train_info)
                            removed_train_ids.append(i)
                            # we might have only taken a subset of the validation data needed
                            remaining_val_size -= train_seq.shape[0]
                        if remaining_val_size < self.min_window_size:
                            break
                    val_seqs = list(reversed(val_seqs))
                    val_seqs_info = list(reversed(val_seqs_info))
                    train_seqs = [
                        seq
                        for i, seq in enumerate(train_seqs)
                        if i not in removed_train_ids
                    ]
                    train_seqs_info = [
                        info
                        for i, info in enumerate(train_seqs_info)
                        if i not in removed_train_ids
                    ]
                    set_to_seqs[val_set_name] += val_seqs
                    set_to_info[val_set_name] += val_seqs_info
                    last_rank = val_seqs_info[-1][-1]
                else:
                    last_rank = train_seqs_info[-1][-1]
                set_to_seqs[train_set_name] += train_seqs
                set_to_info[train_set_name] += train_seqs_info
                set_to_seqs[test_set_name].append(test_seq)
                set_to_info[test_set_name].append([*info, last_rank + 1])
        else:
            raise NotImplementedError
        # rename traces to account for corrected input rates and settings information
        for set_name, set_info in set_to_info.items():
            new_set_info = []
            for seq_name, trace_type, rank in set_info:
                batch_interval = str(TRACE_TO_BATCH_INTERVAL[seq_name])
                n_running_execs = str(TRACE_TO_N_RUNNING_EXECUTORS[seq_name])
                max_exec_memory = str(TRACE_TO_MAX_EXEC_MEMORY[seq_name])
                new_seq_name = get_renamed_trace(seq_name)
                seq_id = new_seq_name.split("_")[-1]
                seq_original_elements = new_seq_name.split("_")[:-1]
                new_seq_name = "_".join(
                    seq_original_elements
                    + [batch_interval, n_running_execs, max_exec_memory, seq_id]
                )
                new_info = [new_seq_name, trace_type, rank]
                new_set_info.append(new_info)
            set_to_info[set_name] = new_set_info
        return set_to_seqs, set_to_info

    def _add_anomaly_column(
        self, seq_dfs: List[pd.DataFrame], seqs_info: List[list], gt_table: pd.DataFrame
    ) -> None:
        """Adds an "anomaly" column inplace to the provided sequence DataFrames.

        Note: each item of `seqs_info` is assumed to be of the form `[file_name, trace_type]`.

        "anomaly" will be set to 0 if the record is outside any anomaly range, otherwise it will be
        set to another value depending on the range type (as defined by utils.spark.metadata.ANOMALY_TYPES).
        => The label for a given range type corresponds to its index in ANOMALY_TYPES + 1.
        """
        logging.info('Adding an "anomaly" column to the Spark traces...')
        for seq_df, seq_info in zip(seq_dfs, seqs_info):
            seq_df["anomaly"] = 0
            file_name, trace_type = seq_info
            if trace_type != "undisturbed":
                for a_t in gt_table[gt_table["trace_name"] == file_name].itertuples():
                    anomaly_type = self._get_anomaly_type(a_t)
                    a_start = a_t.root_cause_start
                    if self.include_extended_effect and not pd.isnull(
                        a_t.extended_effect_end
                    ):
                        a_end = a_t.extended_effect_end
                    else:
                        a_end = a_t.root_cause_end
                    # set the label of an anomaly type as its index in the types list +1
                    seq_df.loc[
                        (seq_df.index >= a_start) & (seq_df.index <= a_end),
                        "anomaly",
                    ] = (
                        ANOMALY_TYPES.index(anomaly_type) + 1
                    )
        logging.info("Done.")

    def _get_anomaly_type(self, a_t) -> str:
        """Returns the anomaly type for the row, or "unknown" if it should be labeled as unknown."""
        anomaly_type = a_t.anomaly_type
        anomaly_type_int = ANOMALY_TYPES.index(anomaly_type) + 1
        if str(anomaly_type_int) in self.label_as_unknown or (
            "os_only" in self.label_as_unknown
            and a_t.anomaly_details == "no_application_impact"
        ):
            anomaly_type = "unknown"
        return anomaly_type

    def _prune_sequences(
        self, seq_dfs: List[pd.DataFrame], seqs_info: List[list]
    ) -> None:
        """Prunes the sequence DataFrames inplace according to `self.seq_name_to_pruned_ranges`.

        Args:
            seq_dfs: the list of input sequences, assumed with an "anomaly" column.
            seqs_info: corresponding sequence information.
        """
        for i, seq_info in enumerate(seqs_info):
            seq_name = seq_info[0]
            if seq_name in self.seq_name_to_pruned_ranges:
                pruned_ranges = self.seq_name_to_pruned_ranges[seq_name]
                seq_length = len(seq_dfs[i])
                masks = []
                for s, e in pruned_ranges:
                    mask = np.ones(seq_length, dtype=bool)
                    mask[slice(s, e)] = False
                    masks.append(mask)
                combined_mask = np.logical_and.reduce(masks)
                seq_dfs[i] = seq_dfs[i][combined_mask]

    def _get_random_val_interval(
        self, seq_length: int, seq_name: str, val_prop: Optional[float] = None
    ) -> (int, int):
        """Returns a (valid) random validation interval `(val_start, val_end)` between 0 and `seq_length`.

        "Valid" means such that the resulting training and validation subsequences all have
          at least `self.min_window_size` data points.

        If `val_prop` is not provided, `self.val_prop` will be used.
        """
        if val_prop is None:
            val_prop = self.val_prop
        val_size = int(val_prop * seq_length)
        if val_size < self.min_window_size:
            raise ValueError(
                f"Validation prop resulted in val_size={val_size} for seq_name={seq_name} "
                f"(below minimum window size of {self.min_window_size})."
            )
        if seq_length - val_size < self.min_window_size:
            raise ValueError(
                f"Validation prop resulted in train_size={seq_length - val_size} for "
                f"seq_name={seq_name} (below minimum window size of {self.min_window_size})."
            )
        # either at the exact start, end, or leaving at least `self.min_window_size` for
        # the training period at the start or end
        val_start_choices = [0]
        val_start_choices += list(
            range(
                self.min_window_size, seq_length - self.min_window_size - val_size + 1
            )
        )
        val_start_choices += [seq_length - val_size]
        val_start = np.random.choice(val_start_choices)
        val_end = val_start + val_size
        return val_start, val_end

    def _get_undisturbed_seq_split(
        self, seq: pd.DataFrame, info: list
    ) -> (List[pd.DataFrame], List[list], List[pd.DataFrame], List[list]):
        """Returns `(train_seqs, train_seqs_info, val_seqs, val_seqs_info)` for the undisturbed `(seq, info)`.

        The split is returned based on `self.val_prop` and `self.train_val_split`:

        - "time": the last `self.val_prop` proportion of records is used for validation.
        - "random": a random subsequence of `self.val_prop` proportion of records is used for validation.
        """
        np.random.seed(self.random_split_seed)
        train_seqs = []
        train_seqs_info = []
        val_seqs = []
        val_seqs_info = []
        if self.train_val_split == "time":
            train_seq, val_seq = train_test_split(
                seq, test_size=self.val_prop, shuffle=False
            )
            train_seqs.append(train_seq)
            val_seqs.append(val_seq)
            train_seqs_info.append([*info, 0])
            val_seqs_info.append([*info, 1])
        else:
            val_start, val_end = self._get_random_val_interval(seq.shape[0], info[0])
            if val_start == 0:
                train_seqs.append(seq.iloc[val_end:])
                train_seqs_info.append([*info, 1])
                val_rank = 0
            elif val_end == seq.shape[0]:
                train_seqs.append(seq.iloc[:val_start])
                train_seqs_info.append([*info, 0])
                val_rank = 1
            else:
                train_seqs.append(seq.iloc[:val_start])
                train_seqs_info.append([*info, 0])
                train_seqs.append(seq.iloc[val_end:])
                train_seqs_info.append([*info, 2])
                val_rank = 1
            val_seqs.append(seq.iloc[val_start:val_end])
            val_seqs_info.append([*info, val_rank])
        return train_seqs, train_seqs_info, val_seqs, val_seqs_info


def load_trace(trace_path, usecols=None):
    """Loads a Spark trace as a pd.DataFrame from its full input path.

    Args:
        trace_path (str): full path of the trace to load (with file extension).
        usecols (list): list of columns to use.

    Features of "inactive" (i.e., "non-running") executors are set to -1. Other -1 values
    are replaced with NaNs, and filled with the previous valid value (or next valid value
    if not applicable).

    Returns:
        pd.DataFrame: the trace indexed by time, with columns processed to be consistent between traces.
    """
    # load trace DataFrame with time as its converted datetime index
    provided_usecols = usecols
    if usecols is not None:
        # add it this way to make a copy (`*_runTime_count` needed by `is_fully_missing_inactive`)
        usecols = usecols + ["t", *[f"{e}_executor_runTime_count" for e in range(1, 5)]]
    try:
        trace_df = pd.read_csv(trace_path, usecols=usecols)
    except ValueError:  # driver "StreamingMetrics" column names need to be preprocessed
        trace_df = pd.read_csv(trace_path)
    trace_df.index = pd.to_datetime(trace_df["t"], unit="s")
    trace_df = trace_df.drop("t", axis=1)

    # remove the previous file prefix from streaming metrics for their name to be consistent across traces
    trace_df.columns = [
        (
            c.replace(f'{"_".join(c.split("_")[1:10])}_', "")
            if "StreamingMetrics" in c
            else c
        )
        for c in trace_df.columns
    ]

    # sort columns (as they might be in a different order depending on the file)
    trace_df = trace_df.reindex(sorted(trace_df.columns), axis=1)

    # replace -1's meaning "punctually unset" with NaN, leaving -1's for "inactive executor"
    trace_df = trace_df.replace(-1, np.nan)
    trace_name = os.path.basename(trace_path)[:-4]
    for exec_id in range(1, 5):
        exec_cols = [c for c in trace_df.columns if c[:2] == f"{exec_id}_"]
        fully_missing_exec_df = (
            trace_df[exec_cols].isnull().all(axis=1)
        )  # all features are missing
        fully_missing_exec_ranges = extract_binary_ranges_ids(
            fully_missing_exec_df.astype(np.int8).values
        )
        for r in fully_missing_exec_ranges:
            if is_fully_missing_inactive(r, trace_df, trace_name, exec_id):
                trace_df.loc[trace_df.index[r[0] : r[1]], exec_cols] = -1.0

    if usecols is not None:
        # remove all features that were not provided if relevant
        trace_df = trace_df.drop(
            [c for c in trace_df.columns if c not in provided_usecols], axis=1
        )

    # fill NaN values
    trace_df = trace_df.ffill().bfill()

    # three "usage-related" features can be their opposite due to corresponding "max" being unset
    trace_df[(trace_df < 0) & (trace_df != -1)] = -trace_df

    # resample data to its supposed resolution to handle duplicate and missing timestamps
    trace_df = trace_df.resample("1s").max().ffill().bfill()
    return trace_df


def is_fully_missing_inactive(missing_range, seq, seq_name, exec_id):
    """Tells whether `missing_range` corresponds to an "inactive executor period".

    Args:
        missing_range (tuple): `(start, end)` indices of the missing range (exclusive end).
        seq (pd.DataFrame): trace DataFrame.
        seq_name (str): name of the trace, needed to check the trace type.
        exec_id (int): executor id to which `missing_range` corresponds.

    Returns:
        bool: `True` if `missing_range` is an "inactive executor period", `False` otherwise.
    """
    # set an "trace-ending" missing range as inactive depending on its duration
    min_inactive_duration = 600  # 10 minutes
    # within T2 anomalies, ending missing ranges are more likely inactive executors
    min_t2_inactive_duration = 10
    s, e = missing_range
    seq_length = seq.shape[0]
    range_length = e - s
    if range_length == seq_length:  # spanning the whole range
        return True
    if s == 0 or e == seq_length:  # either starting or ending the trace
        if range_length >= min_inactive_duration:  # inactive if sufficiently long
            return True
        # relax length constraint to be inactive for ending ranges in T2 anomalies
        if (
            e == seq_length
            and int(seq_name.split("_")[1]) == 2
            and range_length >= min_t2_inactive_duration
        ):
            return True
        return False
    # between two non-missing ranges: was inactive iff counters were reset
    counter_series = seq[f"{exec_id}_executor_runTime_count"]
    return counter_series.iloc[e] < counter_series.iloc[s - 1]  # start is included


def get_seq_to_val_ranges(seqs, seq_names, val_prop, random_seed, output_path=None):
    """Returns a mapping from the sequence (file) name to its validation ranges.

    Args:
        seqs (list[pd.DataFrame]): list of sequences to derive validation ranges from.
        seq_names (list[str]): list of sequence (file) names.
        val_prop (float): proportion of anomaly ranges to include in validation ranges
            (the same proportion will be used for each anomaly type, except "unknown").
        random_seed (int): random seed to use for validation range sampling.
        output_path (str): if provided, will attempt to load required statistics,
            and save selected validation ranges to this path.

    Returns:
        dict: sequence (file) name to validation ranges mapping.
    """
    np.random.seed(random_seed)
    # compute/load relevant statistics for the sequences
    stats_file_name = "disturbed_to_stats"
    seq_to_stats = dict()
    for seq_name, seq in zip(seq_names, seqs):
        seq_ano_type_to_ranges = extract_multiclass_ranges_ids(seq["anomaly"].values)
        seq_ano_type_to_n_ranges = {
            t: len(rs) for t, rs in seq_ano_type_to_ranges.items()
        }
        seq_to_stats[seq_name] = {
            "length": seq.shape[0],
            "ano_type_to_ranges": seq_ano_type_to_ranges,
            "ano_type_to_n_ranges": seq_ano_type_to_n_ranges,
        }
    save_files(output_path, {stats_file_name: seq_to_stats}, "pickle")

    # derive target number of validation ranges for each anomaly type
    type_to_total_n_ranges = dict()
    for stats in seq_to_stats.values():
        for ano_type, n_ranges in stats["ano_type_to_n_ranges"].items():
            if ano_type not in type_to_total_n_ranges:
                type_to_total_n_ranges[ano_type] = 0
            type_to_total_n_ranges[ano_type] += n_ranges
    type_to_target_n_val = {
        t: int(val_prop * n) for t, n in type_to_total_n_ranges.items()
    }

    # sort by ascending anomaly type
    type_to_target_n_val = dict(sorted(type_to_target_n_val.items()))

    # select validation anomaly ranges (not considering "unknown" (T7) anomalies)
    type_to_seq_to_val_ano_ranges = dict()
    # sequences containing both T5 and T6 anomalies
    seqs_with_t5_t6 = []
    if all([t in type_to_seq_to_val_ano_ranges for t in [5, 6]]):
        seqs_with_t5_t6 = [
            sn
            for sn, stats in seq_to_stats.items()
            if all([sn in type_to_seq_to_val_ano_ranges[t].keys() for t in [5, 6]])
        ]
    # selected containing both T5 and T6 anomalies that were selected for T5
    selected_t5_t6_seqs = []
    for ano_type, target_n_val in {
        k: v for k, v in type_to_target_n_val.items() if k != 7
    }.items():
        type_to_seq_to_val_ano_ranges[ano_type] = dict()
        seqs_with_type = [
            sn
            for sn, stats in seq_to_stats.items()
            if ano_type in stats["ano_type_to_ranges"]
        ]
        selected_n_val = 0
        while selected_n_val < target_n_val:
            remaining_n_val = target_n_val - selected_n_val
            # pick a random sequence and select its first `remaining_n_val` ranges
            if ano_type == 5:
                # first try to sample from sequences containing T5 and T6 anomalies if any
                if len(seqs_with_t5_t6) > 0:
                    seq_name = np.random.choice(seqs_with_t5_t6)
                    selected_t5_t6_seqs.append(seq_name)
                    seqs_with_t5_t6.remove(seq_name)
                    seqs_with_type.remove(seq_name)
                elif len(seqs_with_type) > 0:
                    seq_name = np.random.choice(seqs_with_type)
                    seqs_with_type.remove(seq_name)
                else:
                    # no more sequences to select
                    break
            elif ano_type == 6:
                # first try to sample from sequences already selected for T5 if any
                if len(selected_t5_t6_seqs) > 0:
                    seq_name = np.random.choice(selected_t5_t6_seqs)
                    selected_t5_t6_seqs.remove(seq_name)
                    seqs_with_type.remove(seq_name)
                elif len(seqs_with_type) > 0:
                    seq_name = np.random.choice(seqs_with_type)
                    seqs_with_type.remove(seq_name)
                else:
                    # no more sequences to sample from
                    break
            else:
                if len(seqs_with_type) > 0:
                    seq_name = np.random.choice(seqs_with_type)
                    seqs_with_type.remove(seq_name)
                else:
                    # no more sequences to sample from
                    break
            type_to_seq_to_val_ano_ranges[ano_type][seq_name] = seq_to_stats[seq_name][
                "ano_type_to_ranges"
            ][ano_type][:remaining_n_val]
            selected_n_val = sum(
                [len(rs) for rs in type_to_seq_to_val_ano_ranges[ano_type].values()]
            )

    # derive complete validation ranges (including normal data)
    type_to_seq_to_val_ranges = dict()
    for ano_type, seq_to_val_ano_ranges in type_to_seq_to_val_ano_ranges.items():
        if ano_type not in type_to_seq_to_val_ranges:
            type_to_seq_to_val_ranges[ano_type] = dict()
        for seq_name, val_ano_ranges in seq_to_val_ano_ranges.items():
            seq_length = seq_to_stats[seq_name]["length"]
            n_seq_ranges = seq_to_stats[seq_name]["ano_type_to_n_ranges"][ano_type]
            # either consider the whole sequence or up to the last anomaly as validation
            val_end = (
                seq_length
                if len(val_ano_ranges) == n_seq_ranges
                else val_ano_ranges[-1][-1]
            )
            if ano_type not in [5, 6]:
                type_to_seq_to_val_ranges[ano_type][seq_name] = [(0, val_end)]
            else:
                other_type = 5 if ano_type == 6 else 6
                seq_to_other_type_val_ranges = type_to_seq_to_val_ano_ranges[other_type]
                if seq_name not in seq_to_other_type_val_ranges:
                    # the sequence does not contain the other type
                    type_to_seq_to_val_ranges[ano_type][seq_name] = [(0, val_end)]
                else:
                    # sequence contains the other type: have to remove corresponding ranges
                    type_ranges = val_ano_ranges
                    other_type_ranges = seq_to_other_type_val_ranges[seq_name]
                    # ranges up to `val_end`, removing the other type ranges
                    retained_ranges = get_complement_ranges(other_type_ranges, val_end)
                    # remove ranges that do not include any of the current type ranges
                    type_range_sets = [
                        set(range(*type_range)) for type_range in type_ranges
                    ]
                    for retained_range in retained_ranges:
                        retained_range_set = set(range(*retained_range))
                        if not any(
                            [s.issubset(retained_range_set) for s in type_range_sets]
                        ):
                            retained_ranges.remove(retained_range)
                    type_to_seq_to_val_ranges[ano_type][seq_name] = retained_ranges

    # drop anomaly type information, merging validation ranges of T5 and T6 if relevant
    seq_to_val_ranges = dict()
    for ano_type, s_to_val_ranges in type_to_seq_to_val_ranges.items():
        for seq_name, val_ranges in s_to_val_ranges.items():
            if ano_type != 6:
                # sequences are unique for a given anomaly type (except for T5 and T6)
                seq_to_val_ranges[seq_name] = val_ranges
            else:
                t5_to_seq_to_val_ranges = type_to_seq_to_val_ranges[5]
                if seq_name not in t5_to_seq_to_val_ranges:
                    # the sequence with T6 validation ranges does not contain T5 validation ranges
                    seq_to_val_ranges[seq_name] = val_ranges
                else:
                    # it contains T5 validation ranges: merge them if possible
                    t5_ranges = seq_to_val_ranges[seq_name]
                    merged_ranges = get_merged_ranges(
                        t5_ranges + val_ranges, overlapping=True, contiguous=True
                    )
                    seq_to_val_ranges[seq_name] = merged_ranges

    # convert `np.int64` range indices (not JSON-serializable) to `int`
    seq_to_val_ranges = {
        seq_name: [tuple(map(int, r)) for r in val_ranges]
        for seq_name, val_ranges in seq_to_val_ranges.items()
    }
    return seq_to_val_ranges


def get_merged_ranges(ranges, overlapping=True, contiguous=True):
    """Returns merged `ranges`.

    At least one of `overlapping` or `contiguous` must be True.

    Args:
        ranges (list[tuple]): input `(start, end)` ranges to merge.
        overlapping (bool): whether to merge overlapping ranges (i.e., sharing points).
        contiguous (bool): whether to merge contiguous ranges (i.e., starting when the other ends).

    Returns:
        list[tuple]: `ranges` with merged overlapping ranges, contiguous ranges, or both.
    """
    if len(ranges) <= 1:
        # none of the provided ranges can be merged
        return ranges

    if not any([overlapping, contiguous]):
        raise ValueError(
            "At least one of `overlapping` and `contiguous` should be True."
        )

    def should_merge(r1, r2):
        """Ranges are assumed sorted by start index: `r2` cannot start strictly before `r1`."""
        if r2[0] < r1[0]:
            raise ValueError("`r2` cannot start strictly before `r1`.")
        if any([r[0] == r[1] for r in [r1, r2]]):
            raise ValueError("`r1` and `r2` should not be empty.")
        return (overlapping and r1[0] <= r2[0] < r1[1]) or (
            contiguous and r2[0] == r1[1]
        )

    ranges.sort(key=lambda r: r[0])
    merged_ranges = [ranges[0]]
    for cur_range in ranges[1:]:
        prev_range = merged_ranges.pop()
        if should_merge(prev_range, cur_range):
            merged_range = (prev_range[0], max(prev_range[1], cur_range[1]))
            merged_ranges.append(merged_range)
        else:
            merged_ranges += [prev_range, cur_range]
    return merged_ranges


def get_complement_ranges(ranges, seq_length):
    """Returns the complement of `ranges` in a sequence of length `seq_length`."""
    if seq_length <= 0:
        raise ValueError("`seq_length` must be strictly positive.")
    # filter out ranges outside the sequence length
    ranges = [(min(s, seq_length), min(e, seq_length)) for s, e in ranges]
    # filter out empty ranges and check validity of others
    for s, e in ranges:
        if s == e:
            ranges.remove((s, e))
        elif s > e:
            raise ValueError(f"Found range with start after end: (s={s}, e={e}).")
    if len(ranges) == 0:
        return [(0, seq_length)]
    complement_ranges = []
    for i, (cur_start, _) in enumerate(ranges):
        prev_end = 0 if i == 0 else ranges[i - 1][1]
        if not cur_start == prev_end == 0:
            complement_ranges.append((prev_end, cur_start))
    last_end = ranges[-1][1]
    if last_end != seq_length:
        complement_ranges.append((last_end, seq_length))
    return get_merged_ranges(complement_ranges, overlapping=False, contiguous=True)
