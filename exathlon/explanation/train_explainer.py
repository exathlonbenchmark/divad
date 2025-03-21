"""Explainer training module.

TODO: should be able to specify "KPIs", ignored from anomaly explanations.

"Training" can amount to setting hyperparameters of non-parametric methods, and saving
the resulting object.
"""
import os
import pickle
import logging
import importlib
from omegaconf import DictConfig

from utils.data import get_dataset_names
from utils.explanation import get_explainer_type, get_last_detector_step
from data.helpers import load_files, save_files


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    build_features_path = step_to_out_path["build_features"]
    explainer_path = step_to_out_path["train_explainer"]
    dataset_names = get_dataset_names(build_features_path)
    explainer_name = cfg.explainer
    explainer_type = get_explainer_type(explainer_name)
    if explainer_type == "data":
        # detectors are not relevant to initializing standalone data explainers
        detector = None
    else:
        detector_name = cfg.detector
        detector_module = importlib.import_module(
            f"detection.detectors.{detector_name}"
        )
        detector_class = getattr(
            detector_module, detector_name.title().replace("_", "")
        )
        last_detector_step = get_last_detector_step("train_explainer", explainer_name)
        prev_relevant_steps = detector_class.get_previous_relevant_steps(
            last_detector_step
        )
        if len(prev_relevant_steps) == 0:
            detector = detector_class(explainer_path=explainer_path)
        else:
            last_detector_step = prev_relevant_steps[-1]
            last_detector_path = step_to_out_path[last_detector_step]
            detector = pickle.load(
                open(os.path.join(last_detector_path, "detector.pkl"), "rb")
            )
            detector.explainer_path = explainer_path
    if explainer_type == "detector":
        # fit explainable detector
        detector.set_explainer_params(**cfg.train_explainer)
        if "train_explainer" in detector.fitting_steps:
            sequence_datasets = load_files(
                step_to_out_path["build_features"],
                [f"{p}{sn}" for sn in dataset_names[:-1] for p in ["", "y_"]],
                "numpy",
            )
            detector.fit_explainer(**sequence_datasets)
            save_files(explainer_path, {"detector": detector}, "pickle")
    else:
        # fit standalone explainer
        explainer_module = importlib.import_module(
            f"explanation.explainers.{explainer_type}_explainers.{explainer_name}"
        )
        explainer_class = getattr(
            explainer_module, explainer_name.title().replace("_", "")
        )
        detector_kwargs = dict()
        if explainer_type == "model":
            detector_kwargs["predict_window_scorer"] = detector.predict_window_scorer
            detector_kwargs["window_size"] = detector.window_size_
        explainer = explainer_class(
            **detector_kwargs, output_path=explainer_path, **cfg.train_explainer
        )
        if explainer.fitting_step:
            sequence_datasets = load_files(
                step_to_out_path["build_features"],
                [f"{p}{sn}" for sn in dataset_names[:-1] for p in ["", "y_"]],
                "numpy",
            )
            explainer.fit(**sequence_datasets)
        save_files(explainer_path, {"explainer": explainer}, "pickle")
