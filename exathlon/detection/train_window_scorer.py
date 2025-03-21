"""Window scorer training module.

"Training" can amount to setting hyperparameters of non-parametric methods, and saving
the resulting object.
"""
import os
import pickle
import logging
import importlib
from omegaconf import DictConfig

from utils.data import DATASET_NAMES
from data.helpers import save_files
from detection.windows.helpers.general import load_window_datasets


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    window_scorer_path = step_to_out_path["train_window_scorer"]
    detector_name = cfg.detector
    detector_module = importlib.import_module(f"detection.detectors.{detector_name}")
    detector_class = getattr(detector_module, detector_name.title().replace("_", ""))
    prev_relevant_steps = detector_class.get_previous_relevant_steps(
        "train_window_scorer"
    )
    if len(prev_relevant_steps) == 1:  # after `make_window_datasets`
        detector = detector_class(window_scorer_path=window_scorer_path)
    else:
        last_detector_step = prev_relevant_steps[-1]
        last_detector_path = step_to_out_path[last_detector_step]
        detector = pickle.load(
            open(os.path.join(last_detector_path, "detector.pkl"), "rb")
        )
        detector.window_scorer_path = window_scorer_path
    detector.set_window_scorer_params(**cfg.train_window_scorer)
    if "train_window_scorer" in detector.fitting_steps:
        window_datasets_path = step_to_out_path["make_window_datasets"]
        window_datasets = load_window_datasets(
            window_datasets_path,
            set_names=DATASET_NAMES[:2],
            return_labels=True,
            return_info=True,
            concat_datasets=False,
            normal_only=False,
            verbose=False,
        )
        # detector.tune_window_scorer(**window_datasets)
        detector.fit_window_scorer(**window_datasets)
    save_files(window_scorer_path, {"detector": detector}, "pickle")
