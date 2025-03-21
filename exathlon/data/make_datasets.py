"""Data partitioning module (preprocessing and train/val/test sequences constitution).
"""
import logging
import importlib
from omegaconf import DictConfig

from utils.data import TRAIN_SET_NAME, VAL_SET_NAME, TEST_SET_NAME


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    make_datasets_path = step_to_out_path["make_datasets"]
    # set data manager depending on the input data
    manager_module = importlib.import_module(
        f"data.managers.{cfg.dataset.name}_manager"
    )
    formatted_dataset_name = cfg.dataset.name.replace("_", " ").title().replace(" ", "")
    data_manager = getattr(manager_module, f"{formatted_dataset_name}Manager")(
        output_path=make_datasets_path, **cfg.make_datasets.data_manager
    )

    # load and preprocess sequences
    seq_dfs, seqs_info = data_manager.load_raw_sequences(cfg.dataset.path)

    # create and save training, validation and test sequences
    data_manager.save_sequence_datasets(
        seq_dfs,
        seqs_info,
        train_set_name=TRAIN_SET_NAME,
        val_set_name=VAL_SET_NAME,
        test_set_name=TEST_SET_NAME,
    )
