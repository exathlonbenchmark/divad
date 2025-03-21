import abc
from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray

from detection.metrics.helpers import extract_binary_ranges_ids
from detection.detectors.helpers.general import get_parsed_integer_list_str


class BaseEvaluator:
    """Anomaly detection evaluation base class.

    Computes the Precision, Recall and F_{beta}-score for the predicted anomalies on a dataset.

    While Precision is always computed considering all anomaly types as one ("mixed") and
    returned as a single value, Recall and F-score are returned as dictionaries whose keys are:
    - "mixed", considering all non-zero labels as a single positive anomaly class.
    - every distinct non-zero label encountered in the data, considering the corresponding class only.
    - "avg", corresponding to the average scores across each positive class (excluding the "mixed" key).

    Args:
        f_score_beta: assign `beta` times more importance to Recall than Precision.
        ignored_anomaly_labels: anomaly labels to ignore in the evaluation, as an empty string for no ignored
         labels, an integer for a single ignored label, or a string of space-separated integers for multiple
         ignored labels.
        ignored_delayed_window: window size to ignore in the evaluation after every anomaly range (to account
         for rightful delays in anomaly predictions).
    """

    def __init__(
        self,
        f_score_beta: float = 1.0,
        ignored_anomaly_labels: Union[str, int] = "",
        ignored_delayed_window: int = 0,
    ):
        # turn parameter to an actual list
        if isinstance(ignored_anomaly_labels, int):
            ignored_anomaly_labels = str(ignored_anomaly_labels)
        ignored_anomaly_labels = get_parsed_integer_list_str(ignored_anomaly_labels)
        self.beta = f_score_beta
        self.ignored_anomaly_labels = ignored_anomaly_labels
        self.ignored_delayed_window = ignored_delayed_window

    def _get_filtered_periods(self, periods_labels, periods_preds):
        filtered_p_preds = []
        filtered_p_labels = []
        w = self.ignored_delayed_window
        for preds, labels in zip(periods_preds, periods_labels):
            filtered_labels_mask = ~np.isin(labels, self.ignored_anomaly_labels)
            for s, e in extract_binary_ranges_ids(
                filtered_labels_mask.astype(np.int32)
            ):
                f_p_preds = preds[s:e]
                f_p_labels = labels[s:e]
                f_p_ano_ranges = extract_binary_ranges_ids(
                    (f_p_labels > 0).astype(np.int32)
                )
                if len(f_p_ano_ranges) == 0:
                    # add the whole filtered period
                    filtered_p_preds.append(f_p_preds)
                    filtered_p_labels.append(f_p_labels)
                else:
                    # further split period, removing `self.ignored_delayed_window` points
                    # after every anomaly
                    filtered_windows_mask = np.ones(e - s)
                    for ano_s, ano_e in f_p_ano_ranges:
                        filtered_windows_mask[ano_e : min(ano_e + w, e - s)] = 0.0
                        # we want to include all remaining anomalies no matter what: correct
                        # if removed by previous iteration
                        filtered_windows_mask[ano_s:ano_e] = 1.0
                    for f_s, f_e in extract_binary_ranges_ids(filtered_windows_mask):
                        filtered_p_preds.append(preds[s + f_s : s + f_e])
                        filtered_p_labels.append(labels[s + f_s : s + f_e])
        return filtered_p_labels, filtered_p_preds

    def precision_recall_curves(
        self,
        periods_labels: NDArray,
        periods_scores: NDArray,
        return_f_scores: bool = False,
    ) -> (Optional[dict], NDArray, dict, NDArray):
        """Returns the evaluator's precisions and recalls.

        A Precision score is returned for each threshold. Recalls and F-scores follow the same format,
        except the lists are grouped inside dictionaries with keys described in the class documentation.

        Thresholds are set to `self.n_thresholds` equally-spaced values in the sorted list of considered outlier
        scores (always including the min and max values).

        Args:
            periods_labels (ndarray): periods record-wise anomaly labels of shape `(n_periods, period_length)`
                Where `period_length` depends on the period.
            periods_scores (ndarray): periods record-wise outlier scores of the same shape.
            return_f_scores (bool): whether to also return F-beta scores derived from the precisions and recalls.

        Returns:
            The corresponding (F-scores,) Precisions, Recalls and evaluated thresholds.
        """
        filtered_p_labels, filtered_p_scores = self._get_filtered_periods(
            periods_labels, periods_scores
        )
        return self._precision_recall_curves(
            filtered_p_labels, filtered_p_scores, return_f_scores
        )

    @abc.abstractmethod
    def _precision_recall_curves(
        self,
        periods_labels: List[NDArray[np.float32]],
        periods_scores: List[NDArray[np.float32]],
        return_f_scores: bool = False,
    ):
        """Returns the evaluator's precisions and recalls.

        A Precision score is returned for each threshold. Recalls and F-scores follow the same format,
        except the lists are grouped inside dictionaries with keys described in the class documentation.

        Thresholds are set to `self.n_thresholds` equally-spaced values in the sorted list of considered outlier
        scores (always including the min and max values).

        Args:
            periods_labels (ndarray): periods record-wise anomaly labels.
            periods_scores (ndarray): periods record-wise outlier scores.
            return_f_scores (bool): whether to also return F-beta scores derived from the precisions and recalls.

        Returns:
            The corresponding (F-scores,) Precisions, Recalls and evaluated thresholds.
        """

    def compute_metrics(
        self, periods_labels: NDArray, periods_preds: NDArray
    ) -> (dict, float, dict):
        """Returns the F-score, Precision and Recall for the provided `periods_labels` and `periods_preds`.

        Recalls and F-scores are returned as dictionaries like described in the class documentation.

        Args:
            periods_labels: labels for each period in the dataset, of shape
                `(n_periods, period_length)`. With `period_length` depending on the period.
            periods_preds: binary predictions for each period, in the same format.

        Returns:
            F_{beta}-scores, Precision and Recalls, respectively.
        """
        filtered_p_labels, filtered_p_preds = self._get_filtered_periods(
            periods_labels, periods_preds
        )
        return self._compute_metrics(filtered_p_labels, filtered_p_preds)

    @abc.abstractmethod
    def _compute_metrics(
        self,
        periods_labels: List[NDArray[np.float32]],
        periods_preds: List[NDArray[np.float32]],
    ) -> (dict, float, dict):
        """Returns the F-score, Precision and Recall for the provided `periods_labels` and `periods_preds`.

        Recalls and F-scores are returned as dictionaries like described in the class documentation.

        Args:
            periods_labels: labels for each period in the dataset.
            periods_preds: binary predictions for each period.

        Returns:
            F_{beta}-scores, Precision and Recalls, respectively.
        """
