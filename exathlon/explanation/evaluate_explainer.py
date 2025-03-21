"""Explainer training module.

TODO: should be able to specify "KPIs", ignored from anomaly explanations.

"Training" can amount to setting hyperparameters of non-parametric methods, and saving
the resulting object.
"""
import os
import pickle
import logging
from omegaconf import DictConfig

import pandas as pd

from utils.guarding import check_value_in_choices
from utils.data import VAL_SET_NAME, TEST_SET_NAME
from utils.explanation import get_explainer_type
from data.helpers import load_files, load_mixed_formats
from explanation.metrics.evaluators.data_explainer_evaluator import (
    DataExplainerEvaluator,
)
from explanation.metrics.evaluators.model_explainer_evaluator import (
    ModelExplainerEvaluator,
)
from explanation.metrics.evaluation import get_explanation_metrics_row


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    from run import get_leaderboard_steps  # import here to avoid cross-import issue

    logging.info(cfg)
    logging.info(step_to_out_path)
    make_datasets_path = step_to_out_path["make_datasets"]
    build_features_path = step_to_out_path["build_features"]
    explainer_path = step_to_out_path["train_explainer"]
    explainer_evaluation_path = step_to_out_path["evaluate_explainer"]
    explainer_type = get_explainer_type(cfg.explainer)
    # TODO: update.
    feature_names = load_files(build_features_path, ["features_info"], "json")[
        "features_info"
    ]["input_feature_names"]
    # explanation derivation function (always relevant)
    if explainer_type == "detector":
        detector = pickle.load(open(os.path.join(explainer_path, "detector.pkl"), "rb"))
        predict_explainer_func = detector.predict_explainer
        window_size = detector.window_size_
    else:
        explainer = pickle.load(
            open(os.path.join(explainer_path, "explainer.pkl"), "rb")
        )
        predict_explainer_func = explainer.predict
        window_size = explainer.window_size if explainer_type == "model" else None
    # sequence scoring function: only relevant for model explainers and explainer detectors
    evaluator_cfg = cfg.evaluate_explainer.evaluator
    if explainer_type in ["detector", "model"]:
        online_scorer_path = step_to_out_path["train_online_scorer"]
        detector = pickle.load(
            open(os.path.join(online_scorer_path, "detector.pkl"), "rb")
        )
        predict_online_scorer_func = detector.predict_online_scorer
        # TODO: check that explainer is of the right type here, and for data as well.
        evaluator = ModelExplainerEvaluator(
            predict_explainer_func=predict_explainer_func,
            window_size=window_size,
            **evaluator_cfg,
            **cfg.evaluate_explainer.model_explainer_evaluator,
        )
    else:
        predict_online_scorer_func = None
        evaluator = DataExplainerEvaluator(
            predict_explainer_func=predict_explainer_func,
            **evaluator_cfg,
            **cfg.evaluate_explainer.data_explainer_evaluator,
        )
    # sequence binary prediction function: only relevant evaluating model-deemed anomalies
    explained_anomalies = cfg.evaluate_explainer.explained_anomalies
    check_value_in_choices(
        explained_anomalies, "explained_anomalies", ["ground_truth", "model_preds"]
    )
    if explained_anomalies == "model_preds":
        online_detector_path = step_to_out_path["train_online_detector"]
        detector = pickle.load(
            open(os.path.join(online_detector_path, "detector.pkl"), "rb")
        )
        predict_online_detector_func = detector.predict_online_detector
    else:
        predict_online_detector_func = None
    # load relevant explanation data
    check_value_in_choices(
        cfg.evaluate_explainer.test_data,
        "cfg.evaluate_explainer.test_data",
        [VAL_SET_NAME, TEST_SET_NAME],
    )
    if cfg.evaluate_explainer.test_data == VAL_SET_NAME and not os.path.exists(
        os.path.join(build_features_path, f"{VAL_SET_NAME}.npy")
    ):
        raise ValueError(
            "Cannot evaluate on validation data if it was not created in the pipeline."
        )
    test_set = cfg.evaluate_explainer.test_data
    files = load_mixed_formats(
        [make_datasets_path, build_features_path],
        [f"{test_set}_info", test_set],
        ["pickle", "numpy"],
    )
    sequences = files.pop(test_set)
    sequences_info = files.pop(f"{test_set}_info")
    if predict_online_detector_func is None:
        sequences_labels = load_files(build_features_path, [f"y_{test_set}"], "numpy")[
            f"y_{test_set}"
        ]
    else:
        sequences_labels = predict_online_detector_func(sequences)
    if predict_online_scorer_func is None:
        sequences_scores = None
    else:
        sequences_scores = predict_online_scorer_func(sequences)
    metrics_row = get_explanation_metrics_row(
        evaluator=evaluator,
        feature_names=feature_names,
        sequences=sequences,
        sequences_labels=sequences_labels,
        sequences_scores=sequences_scores,
        sequences_info=sequences_info,
        output_path=explainer_evaluation_path,
    )
    if explainer_type == "data" and explained_anomalies == "ground_truth":
        leaderboard_steps = ["train_explainer"]
    else:
        leaderboard_steps = get_leaderboard_steps("evaluate_explainer")
    step_to_id = {s: None for s in leaderboard_steps}
    # TODO: check `step_to_out_path`, not sure adapted for explainers.
    relevant_step_to_id = {
        s: os.path.basename(p).split("__")[-1] for s, p in step_to_out_path.items()
    }
    for step, step_id in relevant_step_to_id.items():
        step_to_id[step] = step_id
    evaluation_id = step_to_id.pop("evaluate_explainer")
    relevant_step_to_id.pop("evaluate_explainer")
    os.makedirs(explainer_evaluation_path, exist_ok=True)
    leaderboard_path = os.path.join(
        explainer_evaluation_path, os.pardir, f"{evaluation_id}.csv"
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
