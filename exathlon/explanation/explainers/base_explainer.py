"""Explanation discovery classes.
"""
import abc
import operator
from typing import Optional, List
from timeit import default_timer as timer

import numpy as np

from data.helpers import save_files


class BaseExplainer:
    """Explainer base class.

    Gathers common functionalities of all explanation discovery methods.

    To be evaluated by an EDEvaluator, explainers must provide their explanation as a dictionary
    with a key called "important_fts", providing a sequence of "important" explanatory features.

    Args:
        output_path: path to save the explanation model and information to.
    """

    def __init__(self, output_path: str = "."):
        self.output_path = output_path

    @property
    @abc.abstractmethod
    def fitting_step(self) -> bool:
        """Whether the explanation method requires some *fitting*."""

    def fit(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        """Fits the explainer parameters to the provided sequences.

        Args:
            train: training sequences, each of shape `(seq_length, n_features)`.
            y_train: corresponding labels of shape `(seq_length,)` for each sequence,
              unused by unsupervised methods.
            val: validation sequences, each of shape `(seq_length, n_features)`.
            y_val: corresponding labels of shape `(seq_length,)` for each sequence,
              unused by unsupervised methods.
        """
        start = timer()
        self._fit(train, y_train, val, y_val)
        end = timer()
        save_files(self.output_path, {"fit_time": {"n_seconds": end - start}}, "json")

    def _fit(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def predict(
        self,
        instances: List[np.array],
        instances_labels: Optional[List[np.array]] = None,
    ) -> List[dict]:
        """Returns anomaly explanations for the provided instances and (optional) labels.

        TODO: should use a feature key format to handle "feature i, timestamp j" instead. For instance,
         string key "{ft_idx}_{time_idx}" in the window, and then optionally replace features with their names.
         Note: InterFusion defines the importance score of a feature as its maximum importance score across
         the anomalous range: this may be simpler.

        The explanation of an instance has to be in the format of the example shown below:

        {
            "feature_to_importance": {3: 0.5, 0: 0.2},
            "feature_to_intervals": {
                3: [(-inf, 0.2, False, True), (0.8, inf, True, False)],
                0: [(1, 2.2, True, True)]
            },
        }

        Keys correspond to feature indices, values are in the form `(start, end, included_start,
        included_end)`.

        We assume only "important" features are in "feature_to_importance" keys (i.e.,
        no "zero-weight" features).

        "feature_to_intervals" is optional. The corresponding decision rule is a
        conjunction of disjunctions. For the example above, this corresponds to:

        (FT_3 <= 0.2 OR FT_3 >= 0.8) AND 1 <= FT_0 <= 2.2.

        Args:
            instances: input instances.
            instances_labels: input sequences labels.

        Returns:
            Explanation for each instance.
        """


def predict_decision_rule(
    sequence: np.array,
    feature_to_intervals: dict,
) -> np.array:
    """Returns a binary anomaly prediction for each record of `sequence` based on `feature_to_interval`."""
    abnormal_masks = []
    for ft, intervals in feature_to_intervals.items():
        ft_abnormal_masks = []
        for s, e, s_included, e_included in intervals:
            start_operator = operator.ge if s_included else operator.gt
            end_operator = operator.le if s_included else operator.lt
            ft_abnormal_masks.append(
                np.logical_and(
                    start_operator(sequence[:, ft], s), end_operator(sequence[:, ft], e)
                )
            )
        abnormal_masks.append(np.logical_or.reduce(ft_abnormal_masks))
    return np.logical_and.reduce(abnormal_masks).astype(int)
