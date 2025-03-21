import os
import pickle
import logging
from abc import abstractmethod
from typing import List

import pandas as pd


class BaseManager:
    """Base DataManager class.

    Args:
        output_path: path to save any information to.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path

    def load_raw_sequences(self, input_path: str) -> (List[pd.DataFrame], List[list]):
        seq_dfs, seqs_info = self._load_raw_sequences(input_path)
        if not ["anomaly" in seq for seq in seq_dfs]:
            raise ValueError(
                'All sequences should contain an "anomaly" columns after loading'
            )
        seq_dfs = self.get_preprocessed_sequences(seq_dfs, seqs_info)
        return seq_dfs, seqs_info

    def get_preprocessed_sequences(
        self, seq_dfs: List[pd.DataFrame], seqs_info: List[list]
    ) -> List[pd.DataFrame]:
        seq_dfs = self._get_preprocessed_sequences(seq_dfs, seqs_info)
        if any([seq.isnull().any().any() for seq in seq_dfs]):
            raise ValueError("There should be no NaN values left after preprocessing.")
        return seq_dfs

    def save_sequence_datasets(
        self,
        seq_dfs: List[pd.DataFrame],
        seqs_info: List[list],
        train_set_name: str = "train",
        val_set_name: str = "val",
        test_set_name: str = "test",
    ) -> None:
        """Saves the training, validation and test sequences (validation sequences are optional).

        Args:
            seq_dfs: preprocessed sequences.
            seqs_info: corresponding sequence-wise information.
            train_set_name: training set name to use to identify and save the related items.
            val_set_name: validation set name to use to identify and save the related items.
            test_set_name: test set name to use to identify and save the related items.
        """
        set_to_seqs, set_to_info = self.get_sequence_datasets(
            seq_dfs,
            seqs_info,
            train_set_name=train_set_name,
            val_set_name=val_set_name,
            test_set_name=test_set_name,
        )
        logging.info(f"Saving datasets to {self.output_path}.")
        os.makedirs(self.output_path, exist_ok=True)
        for set_name in set_to_seqs:
            logging.info(f"Saving {set_name} sequences and information...")
            for file_name, item in zip(
                [set_name, f"{set_name}_info"],
                [set_to_seqs[set_name], set_to_info[set_name]],
            ):
                with open(
                    os.path.join(self.output_path, f"{file_name}.pkl"), "wb"
                ) as pickle_file:
                    pickle.dump(item, pickle_file, protocol=4)
            logging.info("Done.")

    def get_sequence_datasets(
        self,
        seq_dfs: List[pd.DataFrame],
        seqs_info: List[list],
        train_set_name: str = "train",
        val_set_name: str = "val",
        test_set_name: str = "test",
    ) -> (dict, dict):
        set_to_seqs, set_to_info = self._get_sequence_datasets(
            seq_dfs,
            seqs_info,
            train_set_name=train_set_name,
            val_set_name=val_set_name,
            test_set_name=test_set_name,
        )
        dataset_names = [train_set_name, val_set_name, test_set_name]
        for k in list(set_to_seqs.keys()) + list(set_to_info.keys()):
            if k not in dataset_names:
                raise ValueError(
                    f"All dataset names should be in {dataset_names} (received {k})."
                )
        return set_to_seqs, set_to_info

    @abstractmethod
    def _load_raw_sequences(self, input_path: str) -> (List[pd.DataFrame], List[list]):
        """Loads and returns the raw sequences DataFrames and corresponding sequence-wise information.

        The sequence DataFrames should have the timestamps as their row index, and include an
        "anomaly" column, being zero for normal and greater than zero for a given anomaly class.

        For a given sequence, "information" refers to any formatted list of information items
        that may be useful to further processing.

        Args:
            input_path: root path of input data.

        Returns:
            The loaded raw sequences DataFrames and corresponding sequence-wise information.
        """

    @abstractmethod
    def _get_preprocessed_sequences(
        self, seq_dfs: List[pd.DataFrame], seqs_info: List[list]
    ) -> List[pd.DataFrame]:
        """Returns the final preprocessed sequences to use as input to splitting and feature extraction.

        In particular, no NaN values should remain after this step.

        Args:
            seq_dfs: original sequences.
            seqs_info: corresponding sequence-wise information.

        Returns:
            The preprocessed sequences.
        """

    @abstractmethod
    def _get_sequence_datasets(
        self,
        seq_dfs: List[pd.DataFrame],
        seqs_info: List[list],
        train_set_name: str = "train",
        val_set_name: str = "val",
        test_set_name: str = "test",
    ) -> (dict, dict):
        """Returns the training, validation and test sequences (validation sequences are optional).

        Args:
            seq_dfs: preprocessed sequences.
            seqs_info: corresponding sequence-wise information.
            train_set_name: training set name to use to identify and save the related items.
            val_set_name: validation set name to use to identify and save the related items.
            test_set_name: test set name to use to identify and save the related items.

        Returns:
            Sequences and information for each dataset, with as keys the dataset names.
        """
