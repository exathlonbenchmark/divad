import numpy as np

from utils.guarding import check_value_in_choices, check_is_percentage
from data.helpers import get_numpy_from_numpy_list
from detection.metrics.helpers import (
    get_f_beta_score,
    extract_binary_ranges_ids,
    extract_multiclass_ranges_ids,
    get_overlapping_ids,
    get_overlapping_range,
    get_overlapping_ranges,
)
from detection.metrics.evaluators.base import BaseEvaluator


def threshold_period_scores(sequence_scores: np.array, threshold: float) -> np.array:
    return np.array(sequence_scores >= threshold, dtype=int)


def threshold_scores(sequences_scores: np.array, threshold: float) -> np.array:
    # flatten the scores prior to thresholding to improve efficiency
    flattened_preds = threshold_period_scores(
        np.concatenate(sequences_scores, axis=0), threshold
    )
    # recover periods separation
    sequences_preds = []
    cursor = 0
    for seq_length in [len(seq) for seq in sequences_scores]:
        sequences_preds.append(flattened_preds[cursor : cursor + seq_length])
        cursor += seq_length
    return get_numpy_from_numpy_list(sequences_preds)


# evaluation parameters corresponding to each AD level
AD_LEVEL_PARAMS = [
    # AD1
    {
        "recall_alpha": 1.0,
        "recall_omega": "default",
        "recall_delta": "flat",
        "recall_gamma": "dup",
        "precision_omega": "default",
        "precision_delta": "flat",
        "precision_gamma": "dup",
    },
    # AD2
    {
        "recall_alpha": 0.0,
        "recall_omega": "default",
        "recall_delta": "flat",
        "recall_gamma": "dup",
        "precision_omega": "default",
        "precision_delta": "flat",
        "precision_gamma": "dup",
    },
    # AD3
    {
        "recall_alpha": 0.0,
        "recall_omega": "flat.normalized",
        "recall_delta": "front",
        "recall_gamma": "dup",
        "precision_omega": "default",
        "precision_delta": "flat",
        "precision_gamma": "dup",
    },
    # AD4
    {
        "recall_alpha": 0.0,
        "recall_omega": "flat.normalized",
        "recall_delta": "front",
        "recall_gamma": "no.dup",
        "precision_omega": "default",
        "precision_delta": "flat",
        "precision_gamma": "no.dup",
    },
]


class RangeEvaluator(BaseEvaluator):
    """Range-based evaluation.

    The Precision and Recall metrics are defined for range-based time series anomaly detection.

    They can reward existence, ranges cardinality, range overlaps size and position, like introduced
    in https://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(
        self,
        n_thresholds: int = 0,
        evaluation_type: str = "ad2",
        recall_alpha: float = 0.0,
        recall_omega: str = "default",
        recall_delta: str = "flat",
        recall_gamma: str = "dup",
        precision_omega: str = "default",
        precision_delta: str = "flat",
        precision_gamma: str = "dup",
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        check_value_in_choices(
            evaluation_type,
            "evaluation_type",
            ["range", *[f"ad{i}" for i in range(1, 5)]],
        )
        self.n_thresholds = n_thresholds
        if "ad" in evaluation_type:
            pr_params = AD_LEVEL_PARAMS[int(evaluation_type[-1]) - 1]
            # recall specifications
            self.recall_alpha = pr_params["recall_alpha"]
            self.recall_omega = self.omega_functions[pr_params["recall_omega"]]
            self.recall_delta = self.delta_functions[pr_params["recall_delta"]]
            self.recall_gamma = self.gamma_functions[pr_params["recall_gamma"]]
            # precision specifications
            self.precision_omega = self.omega_functions[pr_params["precision_omega"]]
            self.precision_delta = self.delta_functions[pr_params["precision_delta"]]
            self.precision_gamma = self.gamma_functions[pr_params["precision_gamma"]]
        else:
            check_is_percentage(recall_alpha, "recall_alpha")
            # recall specifications
            self.recall_alpha = recall_alpha
            self.recall_omega = self.omega_functions[recall_omega]
            self.recall_delta = self.delta_functions[recall_delta]
            self.recall_gamma = self.gamma_functions[recall_gamma]
            # precision specifications
            self.precision_omega = self.omega_functions[precision_omega]
            self.precision_delta = self.delta_functions[precision_delta]
            self.precision_gamma = self.gamma_functions[precision_gamma]

    def _precision_recall_curves(
        self, periods_labels, periods_scores, return_f_scores=False
    ):
        """Returns the evaluator's precisions and recalls for `self.n_thresholds` thresholds.

        A Precision score is returned for each threshold. Recalls and F-scores follow the same format,
        except the lists are grouped inside dictionaries with keys described in the class documentation.

        Thresholds are set to `self.n_thresholds` equally-spaced values in the sorted list of considered outlier
        scores (always including the min and max values).

        If `self.n_thresholds` was set to zero, all outlier scores are considered as thresholds.

        Args:
            periods_labels: periods record-wise anomaly labels of shape.
            periods_scores: periods record-wise outlier scores.
            return_f_scores: whether to also return F-beta scores derived from the precisions and recalls.

        Returns:
            The corresponding (F-scores,) Precisions, Recalls and evaluated thresholds.
        """
        flattened_scores = np.concatenate(periods_scores, axis=0)
        # get threshold values
        if self.n_thresholds == 0:
            # add a threshold above the maximum score (Recalls are 0 and Precisions should be defined to 1)
            thresholds = np.concatenate([flattened_scores, np.inf])
        else:
            threshold_ids = np.round(
                np.linspace(0, len(flattened_scores) - 1, self.n_thresholds)
            ).astype(int)
            thresholds = np.concatenate(
                [np.sort(flattened_scores)[threshold_ids], [np.inf]]
            )

        # list of mixed Precision scores, lists of mixed and type-wise Recall and F-scores
        precisions, recalls_dict, f_scores_dict = [], dict(), dict()
        for threshold in thresholds:
            # compute predictions corresponding to the threshold and get metrics
            periods_preds = threshold_scores(periods_scores, threshold)
            t_f_scores, t_precision, t_recalls = self._compute_metrics(
                periods_labels, periods_preds
            )
            # append the type-wise and mixed Recall and F-scores for this threshold
            for k in t_f_scores:
                # not the first occurrence of the key: append it to the existing list
                if k in f_scores_dict:
                    recalls_dict[k].append(t_recalls[k])
                    f_scores_dict[k].append(t_f_scores[k])
                # first occurrence of the key: initialize the list with the score value
                else:
                    recalls_dict[k] = [t_recalls[k]]
                    f_scores_dict[k] = [t_f_scores[k]]
            precisions.append(t_precision)

        # convert all lists to numpy arrays and return the PR curves
        precisions = np.array(precisions)
        recalls_dict = {k: np.array(v) for k, v in recalls_dict.items()}
        if return_f_scores:
            f_scores_dict = {k: np.array(v) for k, v in f_scores_dict.items()}
            return f_scores_dict, precisions, recalls_dict, thresholds
        return precisions, recalls_dict, thresholds

    def _compute_metrics(self, periods_labels, periods_preds):
        """Returns range-based metrics for the provided `periods_labels` and `periods_preds`."""
        precisions, recalls_dict = [], dict()
        for y_true, y_pred in zip(periods_labels, periods_preds):
            # add the period's Precision and Recall scores to the full lists
            p_precisions, p_recalls = self.compute_period_metrics(
                y_true, y_pred, as_list=True
            )
            precisions += p_precisions
            # the Recall scores are added per key
            for k in p_recalls:
                if k in recalls_dict:
                    recalls_dict[k] += p_recalls[k]
                else:
                    recalls_dict[k] = p_recalls[k]
        # compute mixed Precision (define it to 1 if no positive predictions)
        n_precisions = len(precisions)
        precision = sum(precisions) / n_precisions if n_precisions > 0 else 1

        # compute mixed and label-wise Recall and F-score (define Recall to 1 if no positive labels)
        returned_recalls, returned_f_scores = dict(), dict()
        for k, recalls_list in dict(
            recalls_dict, **{"mixed": sum(recalls_dict.values(), [])}
        ).items():
            n_recalls = len(recalls_list)
            returned_recalls[k] = sum(recalls_list) / n_recalls if n_recalls > 0 else 1
            returned_f_scores[k] = get_f_beta_score(
                precision, returned_recalls[k], self.beta
            )

        # compute average Recall and F-score across label keys
        label_recalls = {
            k: v for k, v in returned_recalls.items() if k != "mixed"
        }.values()
        returned_recalls["avg"] = (
            sum(label_recalls) / len(label_recalls)
            if len(label_recalls) > 0
            else np.nan
        )
        returned_f_scores["avg"] = get_f_beta_score(
            precision, returned_recalls["avg"], self.beta
        )
        return returned_f_scores, precision, returned_recalls

    def compute_period_metrics(self, y_true, y_pred, as_list=False):
        """Returns the F-score, Precision and Recall for the provided `y_true` and `y_pred` arrays.

        If `as_list` is True, then the Precision and Recall scores for each predicted/real anomaly range
        are returned instead.

        Recalls and F-scores are returned for each non-zero labels encountered in the period.
        => As dictionaries whose keys are the labels and values are the scores.

        Args:
            y_true (ndarray): 1d-array of multiclass anomaly labels.
            y_pred (ndarray): corresponding array of binary anomaly predictions.
            as_list (bool): the F-score, average Precision and average Recall are returned if True, else the list
                of precisions and recalls are returned.

        Returns:
            (dict, float, dict)|(list, dict): F-scores, average Precision and average Recalls if
                `as_list` is False, else lists of Precision and Recall scores.
        """
        # extract contiguous real and predicted anomaly ranges from the period's arrays
        real_ranges_dict = extract_multiclass_ranges_ids(y_true)
        predicted_ranges = extract_binary_ranges_ids(y_pred)
        # compute the Recall and Precision score for each real and predicted anomaly range, respectively
        recalls_dict, precisions = {k: [] for k in real_ranges_dict}, []
        for k, real_ranges in real_ranges_dict.items():
            for real_range in real_ranges:
                recalls_dict[k].append(
                    self.compute_range_recall(real_range, predicted_ranges)
                )
        # consider anomalous ranges of all classes when computing Precision
        r_ranges_values = list(real_ranges_dict.values())
        all_real_ranges = (
            np.concatenate(r_ranges_values, axis=0)
            if len(r_ranges_values) > 0
            else np.array([])
        )
        for predicted_range in predicted_ranges:
            precisions.append(
                self.compute_range_precision(predicted_range, all_real_ranges)
            )
        # return the full lists if specified
        if as_list:
            return precisions, recalls_dict
        # else return the overall F-score, Precision and Recall
        precision = sum(precisions) / len(precisions)
        returned_recalls, returned_f_scores = dict(), dict()
        for k in recalls_dict:
            returned_recalls[k] = sum(recalls_dict[k]) / len(recalls_dict[k])
            returned_f_scores[k] = get_f_beta_score(
                precision, returned_recalls[k], self.beta
            )
        return returned_f_scores, precision, returned_recalls

    def compute_range_precision(self, predicted_range, real_ranges):
        """Returns the Precision score for the provided `predicted_range`, considering `real_ranges`.

        Args:
            predicted_range (ndarray): start and end indices of the predicted anomaly to score, `[start, end]`.
            real_ranges (ndarray): 2d-array for the start and end indices of all the real anomalies.

        Returns:
            float: Precision score of the predicted anomaly range.
        """
        return self.overlap_reward(
            predicted_range,
            real_ranges,
            self.precision_omega,
            self.precision_delta,
            self.precision_gamma,
        )

    def compute_range_recall(self, real_range, predicted_ranges):
        """Returns the Recall score for the provided `real_range`, considering `predicted_ranges`.

        Args:
            real_range (ndarray): start and end indices of the real anomaly to score, `[start, end]`.
            predicted_ranges (ndarray): 2d-array for the start and end indices of all the predicted anomalies.

        Returns:
            float: Recall score of the real anomaly range.
        """
        alpha, omega, delta, gamma = (
            self.recall_alpha,
            self.recall_omega,
            self.recall_delta,
            self.recall_gamma,
        )
        return alpha * RangeEvaluator.existence_reward(real_range, predicted_ranges) + (
            1 - alpha
        ) * RangeEvaluator.overlap_reward(
            real_range, predicted_ranges, omega, delta, gamma
        )

    @staticmethod
    def existence_reward(range_, other_ranges):
        """Returns the existence reward of `range_` with respect to `other_ranges`.

        Args:
            range_ (ndarray): start and end indices of the range whose existence reward to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the other ranges to test
                overlapping with.

        Returns:
            int: 1 if `range_` overlaps with at least one record of `other_ranges`, 0 otherwise.
        """
        return 0 if get_overlapping_ids(range_, other_ranges, True) is None else 1

    @staticmethod
    def overlap_reward(range_, other_ranges, omega_f, delta_f, gamma_f):
        """Returns the overlap reward of `range_` with respect to `other_ranges` and
            the provided functions.

        Args:
            range_ (ndarray): start and end indices of the range whose overlap reward to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the "target" ranges.
            omega_f (func): size function.
            delta_f (func): positional bias.
            gamma_f (func): cardinality function.

        Returns:
            float: the overlap reward of `range_`, between 0 and 1.
        """
        size_rewards = 0
        for other_range in other_ranges:
            size_rewards += omega_f(
                range_, get_overlapping_range(range_, other_range), delta_f
            )
        return (
            RangeEvaluator.cardinality_factor(range_, other_ranges, gamma_f)
            * size_rewards
        )

    """Omega (size) functions.

    Return the size reward of the overlap based on the positional bias of the target range.
    """

    @staticmethod
    def default_size_function(range_, overlap, delta_f):
        """Returns the reward as the overlap size weighted by the positional bias."""
        if overlap is None:
            return 0
        # normalized rewards of the range's indices
        range_rewards = delta_f(range_[1] - range_[0])
        # return the total normalized reward covered by the overlap
        return sum(range_rewards[slice(*(overlap - range_[0]))])

    @staticmethod
    def flat_normalized_size_function(range_, overlap, delta_f):
        """Returns the overlap reward normalized so as not to exceed what it would be under a flat bias."""
        if overlap is None:
            return 0
        # normalized rewards of the range's indices under the provided positional bias
        range_rewards = delta_f(range_[1] - range_[0])
        # normalized rewards of the range's indices under a flat positional bias
        flat_rewards = RangeEvaluator.delta_functions["flat"](range_[1] - range_[0])
        # total normalized rewards covered by the overlap for both the provided and flat biases
        overlap_slice = slice(*(overlap - range_[0]))
        original_reward, flat_reward = sum(range_rewards[overlap_slice]), sum(
            flat_rewards[overlap_slice]
        )
        # best achievable reward given the overlap size (under an "ideal" position)
        max_reward = sum(
            sorted(range_rewards, reverse=True)[: (overlap[1] - overlap[0])]
        )
        # return the original reward normalized so that the maximum is now the flat reward
        return (flat_reward * original_reward / max_reward) if max_reward != 0 else 0

    # dictionary gathering references to the defined `omega` size functions
    omega_functions = {
        "default": default_size_function.__func__,
        "flat.normalized": flat_normalized_size_function.__func__,
    }

    """Delta functions (positional biases). 

    Return the normalized rewards for each relative index in a range of length `range_length`.
    """

    @staticmethod
    def flat_bias(range_length):
        return np.ones(range_length) / range_length

    @staticmethod
    def front_end_bias(range_length):
        """The index rewards linearly decrease as we move forward in the range."""
        raw_rewards = np.flip(np.array(range(range_length)))
        return raw_rewards / sum(raw_rewards)

    @staticmethod
    def back_end_bias(range_length):
        """The index rewards linearly increase as we move forward in the range."""
        raw_rewards = np.array(range(range_length))
        return raw_rewards / sum(raw_rewards)

    # dictionary gathering references to the defined `delta` positional biases
    delta_functions = {
        "flat": flat_bias.__func__,
        "front": front_end_bias.__func__,
        "back": back_end_bias.__func__,
    }

    @staticmethod
    def cardinality_factor(range_, other_ranges, gamma_f):
        """Returns the cardinality factor of `range_` with respect to `other_ranges` and `gamma_f`.

        Args:
            range_ (ndarray): start and end indices of the range whose cardinality factor to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the "target" ranges.
            gamma_f (func): cardinality function.

        Returns:
            float: the cardinality factor of `range_`, between 0 and 1.
        """
        n_overlapping_ranges = len(get_overlapping_ranges(range_, other_ranges))
        if n_overlapping_ranges == 1:
            return 1
        return gamma_f(n_overlapping_ranges)

    """Gamma functions (cardinality)
    """

    @staticmethod
    def no_duplicates_cardinality(n_overlapping_ranges):
        return 0

    @staticmethod
    def allow_duplicates_cardinality(n_overlapping_ranges):
        return 1

    @staticmethod
    def inverse_polynomial_cardinality(n_overlapping_ranges):
        return 1 / n_overlapping_ranges

    # dictionary gathering references to the defined `gamma` cardinality functions
    gamma_functions = {
        "no.dup": no_duplicates_cardinality.__func__,
        "dup": allow_duplicates_cardinality.__func__,
        "inv.poly": inverse_polynomial_cardinality.__func__,
    }
