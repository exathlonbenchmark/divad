"""Window model training module.

"Training" can amount to setting hyperparameters of non-parametric methods, and saving
the resulting object.
"""
import importlib
import logging
from omegaconf import DictConfig

from utils.data import DATASET_NAMES
from data.helpers import save_files
from detection.windows.helpers.general import load_window_datasets


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    window_datasets_path = step_to_out_path["make_window_datasets"]
    window_model_path = step_to_out_path["train_window_model"]
    window_datasets = load_window_datasets(
        window_datasets_path,
        set_names=DATASET_NAMES[:2],
        return_labels=True,
        return_info=True,
        concat_datasets=False,
        normal_only=False,
        verbose=False,
    )
    detector_name = cfg.detector
    detector_module = importlib.import_module(f"detection.detectors.{detector_name}")
    detector = getattr(detector_module, detector_name.title().replace("_", ""))(
        window_model_path=window_model_path
    )
    detector.set_window_model_params(**cfg.train_window_model)
    detector.fit_window_model(**window_datasets)
    # detector.tune_window_model(**window_datasets)
    save_files(window_model_path, {"detector": detector}, "pickle")
