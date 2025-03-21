import numpy as np

from detection.metrics.helpers import get_f_beta_score
from sklearn.metrics import (
    recall_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from detection.metrics.evaluators.base import BaseEvaluator


class PointEvaluator(BaseEvaluator):
    """Point-based evaluation.

    The Precision and Recall metrics are defined for point-based anomaly detection.
    """

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def _precision_recall_curves(
        self, periods_labels, periods_scores, return_f_scores=False
    ):
        """Point-based (faster) implementation."""
        flat_scores = np.concatenate(periods_scores, axis=0)
        flat_labels = np.concatenate(periods_labels, axis=0)
        ano_labels = [l for l in np.unique(flat_labels) if l > 0]
        recalls_dict = dict()
        f_scores_dict = dict()
        precisions = None
        thresholds = None
        for k in ["mixed"] + ano_labels:
            labels_mask = flat_labels > 0 if k == "mixed" else flat_labels == k
            flat_masked_labels = labels_mask.astype(np.int8)
            if k == "mixed":
                precisions, recalls_dict["mixed"], thresholds = precision_recall_curve(
                    flat_masked_labels, flat_scores
                )
                f_scores_dict["mixed"] = np.nan_to_num(
                    get_f_beta_score(precisions, recalls_dict["mixed"], self.beta)
                )
            else:
                _, recalls_dict[k], type_thresholds = precision_recall_curve(
                    flat_masked_labels, flat_scores
                )
                # some low thresholds may be thrown away if they lead to recalls of one
                # (https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/ranking.py#L530).
                n_discarded_thresholds = precisions.shape[0] - recalls_dict[k].shape[0]
                if not np.all(type_thresholds == thresholds[n_discarded_thresholds:]):
                    raise ValueError(
                        "expected sklearn behavior not verified: unreliable evaluation."
                    )
                recalls_dict[k] = np.concatenate(
                    [np.ones(n_discarded_thresholds), recalls_dict[k]]
                )
                f_scores_dict[k] = np.nan_to_num(
                    get_f_beta_score(precisions, recalls_dict[k], self.beta)
                )
        if len(ano_labels) > 0:
            recalls_dict["avg"] = sum([recalls_dict[k] for k in ano_labels]) / len(
                ano_labels
            )
            f_scores_dict["avg"] = get_f_beta_score(
                precisions, recalls_dict["avg"], self.beta
            )
        return (
            f_scores_dict,
            precisions,
            recalls_dict,
            np.concatenate([thresholds, np.full(1, np.inf)]),
        )

    def _compute_metrics(self, periods_labels, periods_preds):
        """Returns point-based metrics for the provided `periods_labels` and `periods_preds`."""
        flattened_preds = np.concatenate(periods_preds, axis=0)
        flattened_labels = np.concatenate(periods_labels, axis=0)
        flattened_binary = np.array(flattened_labels > 0, dtype=int)

        # mixed Precision, Recall and F-score (considering all anomaly types as one)
        recalls_dict = dict()
        f_scores_dict = dict()
        # define the Precision/Recall as 1 if no positive predictions/labels
        (
            precision,
            recalls_dict["mixed"],
            f_scores_dict["mixed"],
            _,
        ) = precision_recall_fscore_support(
            flattened_binary,
            flattened_preds,
            beta=self.beta,
            average="binary",
            zero_division=1,
        )
        # a single anomaly type
        if (flattened_labels == flattened_binary).all():
            for k in [1, "avg"]:
                recalls_dict[k], f_scores_dict[k] = (
                    recalls_dict["mixed"],
                    f_scores_dict["mixed"],
                )
        # multiple anomaly types
        else:
            # type-wise Recall and corresponding F-scores
            unique_labels = np.unique(flattened_labels)
            pos_classes = unique_labels[unique_labels != 0]
            for pc in pos_classes:
                # Recall and corresponding F-score setting the class label to 1 and all others to 0
                recalls_dict[pc] = recall_score(
                    np.array(flattened_labels == pc, dtype=int),
                    flattened_preds,
                    zero_division=1,
                )
                f_scores_dict[pc] = get_f_beta_score(
                    precision, recalls_dict[pc], self.beta
                )
            # average Recall across anomaly types and corresponding F-scores
            label_recalls = {
                k: v for k, v in recalls_dict.items() if k != "mixed"
            }.values()
            recalls_dict["avg"] = (
                sum(label_recalls) / len(label_recalls)
                if len(label_recalls) > 0
                else np.nan
            )
            f_scores_dict["avg"] = get_f_beta_score(
                precision, recalls_dict["avg"], self.beta
            )
        return f_scores_dict, precision, recalls_dict
