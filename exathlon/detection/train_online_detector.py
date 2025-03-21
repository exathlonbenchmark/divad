"""Online detector training module.

"Training" can amount to setting hyperparameters of non-parametric methods, and saving
the resulting object.
"""
import os
import pickle
import logging
import importlib
from omegaconf import DictConfig
from timeit import default_timer as timer

import numpy as np

from utils.data import VAL_SET_NAME, get_dataset_names
from data.helpers import load_files, save_files


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    build_features_path = step_to_out_path["build_features"]
    online_detector_path = step_to_out_path["train_online_detector"]
    dataset_names = get_dataset_names(build_features_path)
    detector_name = cfg.detector
    detector_module = importlib.import_module(f"detection.detectors.{detector_name}")
    detector_class = getattr(detector_module, detector_name.title().replace("_", ""))
    prev_relevant_steps = detector_class.get_previous_relevant_steps(
        "train_online_detector"
    )
    if len(prev_relevant_steps) == 0:
        detector = detector_class(online_detector_path=online_detector_path)
    else:
        last_detector_step = prev_relevant_steps[-1]
        last_detector_path = step_to_out_path[last_detector_step]
        detector = pickle.load(
            open(os.path.join(last_detector_path, "detector.pkl"), "rb")
        )
        detector.online_detector_path = online_detector_path
    threshold_selection = cfg.train_online_detector.pop("threshold_selection")
    detector.set_online_detector_params(**cfg.train_online_detector)
    if "train_online_detector" in detector.fitting_steps:
        if threshold_selection == "supervised":
            # supervised threshold selection on the validation set
            build_features_path = step_to_out_path["build_features"]
            if VAL_SET_NAME not in dataset_names:
                raise ValueError(
                    "Validation sequences must exist to use supervised threshold selection."
                )
            sequence_data = load_files(
                build_features_path, [VAL_SET_NAME, f"y_{VAL_SET_NAME}"], "numpy"
            )
            # the pipeline-level validation sequences is the online detector's training set
            sequence_data = {
                k.replace("val", "train"): v for k, v in sequence_data.items()
            }
        else:
            sequence_data = load_files(
                step_to_out_path["build_features"],
                [f"{p}{sn}" for sn in dataset_names[:-1] for p in ["", "y_"]],
                "numpy",
            )
        detector.fit_online_detector(**sequence_data)
        del sequence_data

    # save online detector predictions for all relevant sequence datasets
    predict_metadata = dict()
    os.makedirs(online_detector_path, exist_ok=True)
    for set_name in dataset_names:
        set_sequences = load_files(build_features_path, [set_name], "numpy")[set_name]
        start = timer()
        set_preds = detector.predict_online_detector(set_sequences)
        end = timer()
        predict_metadata[f"{set_name}_num_seconds"] = end - start
        # TODO: add sequence ID information
        np.savez_compressed(
            os.path.join(online_detector_path, f"{set_name}_preds.npz"), set_preds
        )
    save_files(online_detector_path, {"predict_metadata": predict_metadata}, "json")
    save_files(online_detector_path, {"detector": detector}, "pickle")
