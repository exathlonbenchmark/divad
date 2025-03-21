from typing import List

import numpy as np
import pandas as pd

from data.helpers import save_files

from explanation.metrics.evaluators.base_explainer_evaluator import (
    BaseExplainerEvaluator,
)


def get_column_names(metric_to_name: dict, anomaly_labels: list) -> list:
    """Returns the column names to use for an evaluation DataFrame.

    Args:
        metric_to_name: dictionary mapping metrics to compute to their name.
        anomaly_labels: anomaly labels present in the test set.

    Returns:
        The column names to use for the evaluation.
    """
    if len(anomaly_labels) == 1:
        # single anomaly "type": assumed no type information
        metric_names = [
            "ed1_size",
            "ed1_perturbed_size",
            "ed1_instability",
            "ed1_f1_score",
            "ed1_precision",
            "ed1_recall",
        ]
        column_names = [v for k, v in metric_to_name.items() if k in metric_names]
    else:
        column_names = []
        # anomaly "classes" ("mixed", "balanced" + one per type)
        class_names = ["balanced", "mixed"] + [f"t{i}" for i in anomaly_labels]
        # metrics that are not considered "mixed", metrics that are only considered "mixed"
        no_mixed = [k for k in metric_to_name if "ed2" in k]
        only_mixed = []
        for cn in class_names:
            removed_metrics = no_mixed if cn == "mixed" else only_mixed
            class_metric_names = [
                v for k, v in metric_to_name.items() if k not in removed_metrics
            ]
            column_names += [f"{cn.upper()}_{m}" for m in class_metric_names]
    return column_names


def get_explanation_metrics_row(
    evaluator: BaseExplainerEvaluator,
    feature_names: List[str],
    sequences: np.array,
    sequences_labels: np.array,
    sequences_scores: np.array,
    sequences_info: List[list],
    output_path: str = None,
) -> pd.Series:
    """Returns the metrics row to add to a scoring evaluation DataFrame.

    Args:
        evaluator: anomaly detection evaluation object, defining the metrics of interest.
        feature_names: feature names to use when saving explanations.
        sequences: data point sequences, each of possibly different `seq_length`.
        sequences_labels: corresponding sequence multiclass label of the same shape (0 for normal).
        sequences_scores: anomaly scores of shape `(seq_length,)` for each sequence,
          where higher scores denote more outlyingness.
        sequences_info: corresponding information items for each sequence.
        output_path: path to save the best relevant threshold to, if provided.

    Returns:
        Explanation metrics row.
    """
    anomaly_labels = list(
        np.delete(np.unique(np.concatenate(sequences_labels, axis=0)), 0)
    )
    single_ano_type = len(anomaly_labels) == 1
    metric_to_name = {m: m.upper() for m in evaluator.metric_names}
    column_names = get_column_names(metric_to_name, anomaly_labels)
    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row.append(pd.Series(), ignore_index=True)

    metric_to_type_to_value, seq_to_ano_to_explanation = evaluator.compute_metrics(
        sequences, sequences_labels, sequences_info
    )
    # save the instances explanations as a JSON files (with feature indices and feature names)
    # TODO: fix this, do not know why `NumpyJSONEncoder` works for np.float32 but not np.int64
    # save_files(output_path, {"explanations": seq_to_ano_to_explanation}, "json")
    for seq_id, ano_to_explanation in seq_to_ano_to_explanation.items():
        for ano_id, explanation in ano_to_explanation.items():
            ft_to_importance = explanation["feature_to_importance"]
            ft_name_to_importance = {
                feature_names[ft]: importance
                for ft, importance in ft_to_importance.items()
            }
            seq_to_ano_to_explanation[seq_id][ano_id][
                "feature_to_importance"
            ] = ft_name_to_importance
            if "feature_to_intervals" in explanation:
                ft_to_intervals = explanation["feature_to_intervals"]
                ft_name_to_intervals = {
                    feature_names[ft]: intervals
                    for ft, intervals in ft_to_intervals.items()
                }
                seq_to_ano_to_explanation[seq_id][ano_id][
                    "feature_to_intervals"
                ] = ft_name_to_intervals
    # TODO: rename to "explanations_with_names" if saving the other one as well
    save_files(output_path, {"explanations": seq_to_ano_to_explanation}, "json")

    # case of a single anomaly type (only use "mixed" values of type-wise metrics)
    if single_ano_type:
        if any(["ed2" in k for k in metric_to_type_to_value.keys()]):
            raise ValueError(
                "ED2 metrics should not be computed in case of a single anomaly type."
            )
        for m_name, m_type_to_value in metric_to_type_to_value.items():
            metrics_row.at[0, f"{metric_to_name[m_name]}"] = m_type_to_value["mixed"]
    # case of multiple anomaly types (loop through "balanced", "mixed" and all relevant anomaly types)
    else:
        evaluated_types = ["balanced", "mixed"] + [f"t{i}" for i in anomaly_labels]
        for t in evaluated_types:
            if t == "balanced":
                eval_key = "avg"
            elif t == "mixed":
                eval_key = "mixed"
            else:
                eval_key = int(t[1:])
            for m_name, m_type_to_value in metric_to_type_to_value.items():
                if not (t == "mixed" and "ed2" in m_name):
                    metrics_row.at[
                        0, f"{t.upper()}_{metric_to_name[m_name]}"
                    ] = m_type_to_value[eval_key]
    return metrics_row.iloc[0]
