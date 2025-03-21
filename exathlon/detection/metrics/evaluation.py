"""Anomaly detection evaluation module.
"""
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from detection.metrics.helpers import get_peak_fprt, get_auc
from detection.metrics.evaluators.base import BaseEvaluator
from utils.guarding import check_value_in_choices
from data.helpers import save_files


def get_scoring_metric_to_name(f_score_beta: float = 1.0) -> dict:
    return {
        "auprc": "AUPRC",
        "peak_f_score": f"PEAK_F{f_score_beta}_SCORE",
        # precision and recall at the peak f-score
        "precision_at_peak": "PRECISION@PEAK",
        "recall_at_peak": "RECALL@PEAK",
    }


def get_detection_metric_to_name(f_score_beta: float = 1.0) -> dict:
    return {
        "f_score": f"F{f_score_beta}_SCORE",
        "precision": "PRECISION",
        "recall": "RECALL",
    }


def get_column_names(
    evaluation_step: str,
    metric_to_name: dict,
    anomaly_labels: list,
) -> list:
    """Returns the column names to use for an evaluation DataFrame.

    Args:
        evaluation_step: evaluation step (must be either "scoring", "detection").
        metric_to_name: dictionary mapping metrics to compute to their name.
        anomaly_labels: anomaly labels present in the test set.

    Returns:
        list: the column names to use for the evaluation.
    """
    check_value_in_choices(evaluation_step, "evaluation_step", ["scoring", "detection"])
    if len(anomaly_labels) == 1:
        # case of a single anomaly type
        return list(metric_to_name.values())
    column_names = []
    # anomaly "classes" ("mixed", "balanced" + one per type)
    class_names = ["balanced", "mixed"] + [f"t{i}" for i in anomaly_labels]
    # metrics that are not considered "mixed", metrics that are only considered "mixed"
    no_mixed = []
    only_mixed = []
    if evaluation_step == "detection":
        only_mixed += ["precision"]
    for cn in class_names:
        removed_metrics = no_mixed if cn == "mixed" else only_mixed
        class_metric_names = [
            v for k, v in metric_to_name.items() if k not in removed_metrics
        ]
        column_names += [f"{cn.upper()}_{m}" for m in class_metric_names]
    return column_names


def get_scoring_metrics_row(
    evaluator: BaseEvaluator,
    sequences_scores: np.array,
    sequences_labels: np.array,
    metric_to_name: dict,
    sequences_info: Optional[List[Tuple]] = None,
    output_path: Optional[str] = None,
) -> pd.Series:
    """Returns the metrics row to add to a scoring evaluation DataFrame.

    Args:
        evaluator: anomaly detection evaluation object, defining the metrics of interest.
        sequences_scores: anomaly scores of shape `(seq_length,)` for each sequence,
          where higher scores denote more outlyingness.
        sequences_labels: corresponding sequence multiclass label of the same shape (0 for normal).
        metric_to_name: metric-to-name mapping.
        sequences_info: corresponding information items for each sequence.
        output_path: path to save the best relevant threshold to, if provided.

    Returns:
        Anomaly score metrics row.
    """
    anomaly_labels = list(
        np.delete(np.unique(np.concatenate(sequences_labels, axis=0)), 0)
    )
    column_names = get_column_names(
        "scoring",
        metric_to_name,
        anomaly_labels,
    )
    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row = pd.concat([metrics_row, pd.DataFrame()], ignore_index=True)

    # compute the PR curve(s) using the Precision and Recall metrics defined by the evaluator
    f, p, r, pr_ts = evaluator.precision_recall_curves(
        sequences_labels, sequences_scores, return_f_scores=True
    )
    peak_f, p_at_peak, r_at_peak, ts_at_peak = dict(), dict(), dict(), dict()
    # case of a single anomaly type (only use "mixed" values of type-wise metrics)
    if len(anomaly_labels) == 1:
        relevant_ts_key = "mixed"
        metrics_row.at[0, f'{metric_to_name["auprc"]}'] = get_auc(r["mixed"], p)
        (
            peak_f["mixed"],
            p_at_peak["mixed"],
            r_at_peak["mixed"],
            ts_at_peak["mixed"],
        ) = get_peak_fprt(f["mixed"], p, r["mixed"], pr_ts)
        metrics_row.at[0, f'{metric_to_name["peak_f_score"]}'] = peak_f["mixed"]
        for k, v in zip(
            ["precision", "recall"], [p_at_peak["mixed"], r_at_peak["mixed"]]
        ):
            metrics_row.at[0, f'{metric_to_name[f"{k}_at_peak"]}'] = v
    # case of multiple anomaly types (loop through "balanced", "mixed" and all relevant anomaly types)
    else:
        relevant_ts_key = "balanced"
        evaluated_types = ["balanced", "mixed"] + [f"t{i}" for i in anomaly_labels]
        for t in evaluated_types:
            if t == "balanced":
                eval_key = "avg"  # f['avg'] is actually f(r['avg'], p)
            elif t == "mixed":
                eval_key = "mixed"
            else:
                eval_key = int(t[1:])
            if eval_key in r:  # type not ignored by evaluator
                auc = get_auc(r[eval_key], p)
                peak_f[t], p_at_peak[t], r_at_peak[t], ts_at_peak[t] = get_peak_fprt(
                    f[eval_key], p, r[eval_key], pr_ts
                )
            else:
                auc = np.nan
                peak_f[t] = np.nan
                p_at_peak[t] = np.nan
                r_at_peak[t] = np.nan
                ts_at_peak[t] = np.nan
            metrics_row.at[0, f'{t.upper()}_{metric_to_name["auprc"]}'] = auc
            metrics_row.at[0, f'{t.upper()}_{metric_to_name["peak_f_score"]}'] = peak_f[
                t
            ]
            for k, v in zip(["precision", "recall"], [p_at_peak[t], r_at_peak[t]]):
                metrics_row.at[0, f'{t.upper()}_{metric_to_name[f"{k}_at_peak"]}'] = v
            if eval_key in r and t not in ["balanced", "mixed"]:
                # replace label keys by type keys for recall and f-score (used in PR and F-score plots)
                r[t], f[t] = r.pop(eval_key), f.pop(eval_key)
        # we do not need average metrics across types anymore
        for d in f, r:
            d.pop("avg")

    if output_path is not None:
        best_relevant_ts = ts_at_peak[relevant_ts_key]
        save_files(
            output_path,
            {f"best_{relevant_ts_key}_threshold": {"value": best_relevant_ts}},
            "json",
        )
    return metrics_row.iloc[0]


def get_best_score_threshold(
    scoring_evaluator: BaseEvaluator,
    sequences_scores: np.array,
    sequences_labels: np.array,
    f_score_type: str = "mixed",
) -> float:
    """Returns the score threshold corresponding to the best F-score for the provided sequences.

    Args:
        scoring_evaluator: anomaly scoring evaluation object, defining the metrics of interest.
        sequences_scores: anomaly scores of shape `(seq_length,)` for each sequence,
          where higher scores denote more outlyingness.
        sequences_labels: corresponding sequence multiclass label of the same shape (0 for normal).
        f_score_type: type of F-score to consider ("mixed" or "balanced"; "balanced" only being
          relevant for multiple anomaly types.)

    Returns:
        The best anomaly score threshold in terms of `f_score_type` F`scoring_evaluator.beta`-score.
    """
    check_value_in_choices(f_score_type, "f_score_type", ["mixed", "balanced"])
    anomaly_labels = list(
        np.delete(np.unique(np.concatenate(sequences_labels, axis=0)), 0)
    )
    if len(anomaly_labels) == 1 and f_score_type == "balanced":
        raise ValueError('Cannot report "balanced" F-score for a single anomaly type.')
    # compute the PR curve(s) using the Precision and Recall metrics defined by the evaluator
    f, p, r, pr_ts = scoring_evaluator.precision_recall_curves(
        sequences_labels, sequences_scores, return_f_scores=True
    )
    # f["avg"] corresponds to f(r["avg"], p)
    k = "mixed" if f_score_type == "mixed" else "avg"
    _, _, _, peak_threshold = get_peak_fprt(f[k], p, r[k], pr_ts)
    return peak_threshold


def get_detection_metrics_row(
    evaluator: BaseEvaluator,
    sequences_preds: np.array,
    sequences_labels: np.array,
    metric_to_name: dict,
    sequences_info: Optional[List[Tuple]] = None,
    output_path: Optional[str] = None,
) -> pd.Series:
    """Returns the metrics row to add to a detection evaluation DataFrame.

    Args:
        evaluator: anomaly detection evaluation object, defining the metrics of interest.
        sequences_preds: (binary) anomaly predictions of shape `(seq_length,)` for each sequence,
          0 for normal, 1 for anomaly.
        sequences_labels: corresponding sequence multiclass label of the same shape (0 for normal).
        metric_to_name: metric-to-name mapping.
        sequences_info: corresponding information items for each sequence.
        output_path: path to save evaluation items to, if provided.

    Returns:
        Anomaly detection metrics row.
    """
    anomaly_labels = list(
        np.delete(np.unique(np.concatenate(sequences_labels, axis=0)), 0)
    )
    column_names = get_column_names(
        "detection",
        metric_to_name,
        anomaly_labels,
    )
    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row.append(pd.Series(), ignore_index=True)

    # compute the metrics defined by the evaluator
    f, p, r = evaluator.compute_metrics(sequences_labels, sequences_preds)

    # case of a single anomaly type (only use "mixed" values of type-wise metrics)
    if len(anomaly_labels) == 1:
        for k, v in zip(
            ["f_score", "precision", "recall"], [f["mixed"], p, r["mixed"]]
        ):
            metrics_row.at[0, metric_to_name[k]] = v
    # case of multiple anomaly types (loop through "balanced", "mixed" and all relevant anomaly types)
    else:
        evaluated_types = ["balanced", "mixed"] + [f"t{i}" for i in anomaly_labels]
        for t in evaluated_types:
            if t == "balanced":
                eval_key = "avg"  # f['avg'] is actually f(r['avg'], p)
            elif t == "mixed":
                eval_key = "mixed"
            else:
                eval_key = int(t[1:])
            if eval_key in r:  # type not ignored by evaluator
                keys = ["f_score"]
                values = [f[eval_key]]
                if t == "mixed":
                    keys.append("precision")
                    values.append(p)
                keys.append("recall")
                values.append(r[eval_key])
            else:
                # "mixed" is never ignored
                keys = ["f_score", "recall"]
                values = [np.nan, np.nan]
            for k, v in zip(keys, values):
                metrics_row.at[0, f"{t.upper()}_{metric_to_name[k]}"] = v
    return metrics_row.iloc[0]
