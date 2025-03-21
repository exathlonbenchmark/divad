"""Window datasets constitution module.

Turns sequence datasets to window datasets, with windows of shape `(window_size, n_features)`.
"""
import logging
from omegaconf import DictConfig

from utils.data import TRAIN_SET_NAME, VAL_SET_NAME
from detection.windows.window_manager import WindowManager
from data.helpers import load_mixed_formats


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    make_datasets_path = step_to_out_path["make_datasets"]
    build_features_path = step_to_out_path["build_features"]
    window_datasets_path = step_to_out_path["make_window_datasets"]

    # data passed to the window manager
    data = dict()
    files = dict()
    file_names = []
    set_names = []
    data_keys = []
    for set_name in [TRAIN_SET_NAME, VAL_SET_NAME]:
        set_file_names = [
            f"{p}{set_name}{s}" for p, s in [("", ""), ("y_", ""), ("", "_info")]
        ]
        try:
            set_files = load_mixed_formats(
                [build_features_path, build_features_path, make_datasets_path],
                set_file_names,
                ["numpy", "numpy", "pickle"],
            )
            for k, v in set_files.items():
                files[k] = v
            file_names += set_file_names
            set_names.append(set_name)
            data_keys += [f"{set_name}_periods{s}" for s in ["", "_labels", "_info"]]
        except FileNotFoundError:
            pass
    for k, file_name in zip(data_keys, file_names):
        data[k] = files[file_name]

    # make modeling window datasets using the window manager
    window_manager = WindowManager(
        **cfg.make_window_datasets.window_manager, output_path=window_datasets_path
    )
    _ = window_manager.save_window_datasets(**data)
