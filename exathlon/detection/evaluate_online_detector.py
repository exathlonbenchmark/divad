"""Online detector evaluation module.
"""
import os
import pickle
import logging

import pandas as pd
from omegaconf import DictConfig

from utils.guarding import check_value_in_choices
from utils.data import VAL_SET_NAME, TEST_SET_NAME
from data.helpers import load_mixed_formats, get_matching_sampling
from detection.metrics.evaluators.point import PointEvaluator
from detection.metrics.evaluators.range import RangeEvaluator
from detection.metrics.evaluation import (
    get_detection_metric_to_name,
    get_detection_metrics_row,
)


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    from run import get_leaderboard_steps  # import here to avoid cross-import issue

    logging.info(cfg)
    logging.info(step_to_out_path)
    make_datasets_path = step_to_out_path["make_datasets"]
    build_features_path = step_to_out_path["build_features"]
    online_detector_path = step_to_out_path["train_online_detector"]
    online_detector_evaluation_path = step_to_out_path["evaluate_online_detector"]
    detector = pickle.load(
        open(os.path.join(online_detector_path, "detector.pkl"), "rb")
    )
    evaluator_type = cfg.evaluate_online_detector.evaluator.pop("evaluation_type")
    evaluator_class = PointEvaluator if evaluator_type == "point" else RangeEvaluator
    evaluator = evaluator_class(**cfg.evaluate_online_detector.evaluator)
    check_value_in_choices(
        cfg.evaluate_online_detector.test_data,
        "cfg.evaluate_online_detector.test_data",
        [VAL_SET_NAME, TEST_SET_NAME],
    )
    if cfg.evaluate_online_detector.test_data == VAL_SET_NAME and not os.path.exists(
        os.path.join(build_features_path, f"{VAL_SET_NAME}.npy")
    ):
        raise ValueError(
            "Cannot evaluate on validation data if it was not created in the pipeline."
        )
    test_set = cfg.evaluate_online_detector.test_data
    files = load_mixed_formats(
        [make_datasets_path, build_features_path, build_features_path],
        [f"{test_set}_info", test_set, f"y_{test_set}"],
        ["pickle", "numpy", "numpy"],
    )
    sequences_preds = detector.predict_online_detector(files.pop(test_set))
    sequences_labels = files.pop(f"y_{test_set}")
    sequences_info = files.pop(f"{test_set}_info")
    if (
        cfg.dataset.sampling_period != "na"
        and cfg.build_features.data_sampling_period != cfg.dataset.sampling_period
    ):
        # upsample anomaly predictions to match the sampling period of labels
        # TODO: test this when relevant.
        sequences_preds = get_matching_sampling(
            sequences_preds,
            sequences_labels,
            sampling_period=cfg.build_features.data_sampling_period,
            target_sampling_period=cfg.dataset.sampling_period,
        )
    metrics_row = get_detection_metrics_row(
        evaluator=evaluator,
        sequences_preds=sequences_preds,
        sequences_labels=sequences_labels,
        metric_to_name=get_detection_metric_to_name(evaluator.beta),
        sequences_info=sequences_info,
        output_path=online_detector_evaluation_path,
    )
    step_to_id = {s: None for s in get_leaderboard_steps("evaluate_online_detector")}
    relevant_step_to_id = {
        s: os.path.basename(p).split("__")[-1] for s, p in step_to_out_path.items()
    }
    for step, step_id in relevant_step_to_id.items():
        step_to_id[step] = step_id
    evaluation_id = step_to_id.pop("evaluate_online_detector")
    relevant_step_to_id.pop("evaluate_online_detector")
    os.makedirs(online_detector_evaluation_path, exist_ok=True)
    leaderboard_path = os.path.abspath(
        os.path.join(online_detector_evaluation_path, os.pardir, f"{evaluation_id}.csv")
    )
    leaderboard_row = pd.concat([pd.Series(step_to_id), metrics_row])
    try:
        leaderboard = pd.read_csv(leaderboard_path, index_col=False)
        irrelevant_steps = [s for s in step_to_id if s not in relevant_step_to_id]
        same_relevant_steps = (
            leaderboard[relevant_step_to_id.keys()] == relevant_step_to_id.values()
        ).all(axis=1)
        empty_irrelevant_steps = leaderboard[irrelevant_steps].isnull().all(axis=1)
        same_config_ids = leaderboard.loc[
            same_relevant_steps & empty_irrelevant_steps
        ].index
        if len(same_config_ids) > 0:
            logging.warning(
                f"Replacing identical configuration found in leaderboard: "
                f"{leaderboard.iloc[same_config_ids[0]].to_dict()}"
            )
            leaderboard = leaderboard.drop(same_config_ids, axis=0)
            leaderboard = leaderboard.reset_index(drop=True)
        leaderboard = leaderboard.append(leaderboard_row, ignore_index=True)
        leaderboard.to_csv(leaderboard_path, index=False)
    except FileNotFoundError:
        leaderboard = pd.DataFrame(columns=leaderboard_row.index)
        leaderboard.loc[0] = leaderboard_row.values
        leaderboard.to_csv(leaderboard_path, index=False)
