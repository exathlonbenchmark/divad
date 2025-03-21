"""
TODO: instability calculation could be simplified with a better perturbation mechanism:
 => E.g., add a small amount of noise so that the prediction of the model does not change.
 => A larger issue is that the perturbation mechanism should not vary between model and data
 explainers in order for them to be comparable.
"""
import logging
import random
from typing import Union, Tuple
from math import floor, ceil
from timeit import default_timer as timer

import numpy as np

from utils.guarding import check_value_in_choices
from data.helpers import get_sliding_windows
from detection.metrics.helpers import extract_binary_ranges_ids
from explanation.metrics.evaluators.base_explainer_evaluator import (
    BaseExplainerEvaluator,
)


class ModelExplainerEvaluator(BaseExplainerEvaluator):
    """Model explainer evaluator class.

    Evaluates explainers trying to explain predictions of an AD model.

    Such AD models and corresponding explainers are assumed to rely on a fixed input "sample length",
    which will typically differ from most anomaly lengths.

    - Anomalies smaller than the sample are either dropped or expanded with neighboring data,
        according to the value of `small_anomalies_expansion`.
    - Anomalies larger than the sample are either "fully" or partially covered,
        according to the value of `large_anomalies_coverage`.

    Args:
        window_size: window size.
        small_anomalies_expansion: expansion policy for anomalies smaller than sample length
          ("none" for dropping them).
        large_anomalies_coverage: coverage policy for anomalies larger than sample length.
        random_seed: random seed to use in case of "all" large anomalies coverage.
    """

    def __init__(
        self,
        window_size: int,
        small_anomalies_expansion: str,
        large_anomalies_coverage: str,
        random_seed: int = 0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        # check parameters
        check_value_in_choices(
            small_anomalies_expansion,
            "small_anomalies_expansion",
            ["none", "before", "after", "both"],
        )
        check_value_in_choices(
            large_anomalies_coverage,
            "large_anomalies_coverage",
            ["all", "center", "end"],
        )
        self.window_size = window_size
        self.small_anomalies_expansion = small_anomalies_expansion
        self.large_anomalies_coverage = large_anomalies_coverage
        # minimum evaluation instance length required for explanation and instability computation
        self.min_instance_length = (
            window_size + self.ed1_instability_n_perturbations - 1
        )
        # label for the target anomaly in an evaluation instance, in case neighboring data is not fully normal
        self.target_anomaly_class = -1
        self.random_seed = random_seed

    def get_evaluation_instance(self, sequence, sequence_labels, pos_ranges, range_idx):
        """Model explainer implementation.

        For model explainers, evaluation instances are the full anomaly ranges, with instances
        smaller than the sample length either being ignored or expanded with neighboring data, according
        to the value of `self.small_anomalies_expansion`.
        """
        pos_range = pos_ranges[range_idx]
        anomaly_length = pos_range[1] - pos_range[0]
        range_class = sequence_labels[pos_range[0]]
        # insufficient anomaly length or dropping policy of anomalies smaller than the sample length
        if anomaly_length < self.min_anomaly_length or (
            anomaly_length < self.window_size
            and self.small_anomalies_expansion == "none"
        ):
            logging.warning(
                f"Instance of type {range_class} dropped due to insufficient anomaly length."
            )
            return None, None
        # initialize the instance and labels to the anomaly range
        instance = sequence[slice(*pos_range)]
        instance_labels = self.target_anomaly_class * np.ones(anomaly_length)
        prepended_start, appended_end = pos_range
        if anomaly_length < self.min_instance_length:
            # prepend and/or append some neighboring records
            period_end = len(sequence)
            n_added = self.min_instance_length - anomaly_length
            if self.small_anomalies_expansion != "both":
                # try to prepend *or* append records
                expansion_type = (
                    "prepend"
                    if self.small_anomalies_expansion == "before"
                    else "append"
                )
                prepended_start, appended_end = self.get_expanded_indices(
                    expansion_type,
                    n_added,
                    prepended_start,
                    appended_end,
                    range_class,
                    period_end,
                )
            else:
                # try to prepend *and* append records (arbitrarily append more than prepended if odd `n_added`)
                half_n_added = n_added / 2
                for expansion_type, n_records in zip(
                    ["prepend", "append"], [floor(half_n_added), ceil(half_n_added)]
                ):
                    if not (prepended_start is None or appended_end is None):
                        prepended_start, appended_end = self.get_expanded_indices(
                            expansion_type,
                            n_records,
                            prepended_start,
                            appended_end,
                            range_class,
                            period_end,
                        )
            if prepended_start is None or appended_end is None:
                # instance expansion was needed and failed: drop the instance
                return None, None
        # return instance and instance labels
        prep_slice = slice(prepended_start, pos_range[0])
        app_slice = slice(pos_range[1], appended_end)
        expanded_instance = np.concatenate(
            [sequence[prep_slice], instance, sequence[app_slice]]
        )
        expanded_labels = np.concatenate(
            [sequence_labels[prep_slice], instance_labels, sequence_labels[app_slice]]
        )
        return expanded_instance, expanded_labels

    @staticmethod
    def get_expanded_indices(
        expansion_type: str,
        n_added: int,
        initial_start: int,
        initial_end: int,
        range_class: int,
        period_end: int,
    ) -> Union[Tuple[int, int], Tuple[None, None]]:
        """Returns the expanded indices of the evaluation instance initially at `[initial_start, initial_end]`
            inside a `[0, period_length]` period, according to `expansion_type`.

        Note: The added neighboring data is not required to be fully normal.

        Note: If no records are available before/after the original instance bounds, they will
        be taken on the "other side" instead, if possible, no matter `expansion_type`.

        Args:
            expansion_type: type of expansion to perform (either "prepend" or "append")
            n_added: number of neighboring records to append/prepend.
            initial_start: initial (included) start index of the instance.
            initial_end: initial (excluded) end index of the instance.
            range_class: anomaly type of the instance, only used when displaying warnings.
            period_end: (excluded) end index of the period the instance belongs to.

        Returns:
            The updated instance's (included) start and (excluded) end indices if the instance
              could be expanded, (None, None) otherwise.
        """
        check_value_in_choices(expansion_type, "expansion_type", ["prepend", "append"])
        prepended_start = initial_start
        appended_end = initial_end
        if expansion_type == "prepend":
            prepended_start -= n_added
            if prepended_start < 0:
                n_appended = -prepended_start
                prepended_start = 0
                logging.warning(
                    f"Failed to fully prepend {n_added} records to type {range_class} instance, "
                    f"trying to append the remaining {n_appended}..."
                )
                appended_end += n_appended
                if appended_end > period_end:
                    logging.warning(f"Failed. Instance dropped.")
                    return None, None
                logging.warning(f"Succeeded.")
        else:
            appended_end += n_added
            if appended_end > period_end:
                n_prepended = appended_end - period_end
                appended_end = period_end
                logging.warning(
                    f"Failed to fully append {n_added} records to type {range_class} instance, "
                    f"trying to prepend the remaining {n_prepended}..."
                )
                prepended_start -= n_prepended
                if prepended_start < 0:
                    logging.warning(f"Failed. instance dropped.")
                    return None, None
                logging.warning(f"Succeeded.")
        return prepended_start, appended_end

    def explain_instance(
        self, instance: np.array, instance_labels: np.array
    ) -> (dict, float):
        """Model explainer implementation.

        Model explainers can only explain samples of fixed length.

        For them, evaluation instances are the anomaly ranges, possibly expanded with some neighboring
        data for ranges that were smaller than `self.min_instance_length`.

        The explanation of an instance is defined depending on the length of its anomaly:

        - If it equals the sample length, it is the sample explanation.
        - If it is smaller than the sample length, it is the explanation of the sample obtained before
            adding extra records for ED1 instability computation, when expanding the anomaly.
        - If it is larger than the sample length, it is defined according to the value of
            `self.large_anomalies_coverage`.

        We define the shared explanation of a set of instances as the (duplicate-free) union
        of the "important" explanatory features found for each sample.

        Indeed, for a single sample, these important features constitute those that affected
        the outlier score function the most locally. Taking the union of these features hence
        gives us a sense of the features that were found most relevant throughout the samples.
        """
        instance_length = len(instance)
        sample_length = self.window_size

        # the whole instance is the target anomaly (and larger than the sample length)
        if instance_length > self.min_instance_length:
            if self.large_anomalies_coverage == ["center", "end"]:
                if self.large_anomalies_coverage == "center":
                    # lean more towards the instance start if different parities
                    sample_start = floor((instance_length - sample_length) / 2)
                else:
                    sample_start = instance_length - sample_length
                samples = np.array(
                    [instance[sample_start : sample_start + sample_length]]
                )
            else:
                # cover jumping samples spanning the whole anomaly range
                samples = get_sliding_windows(
                    instance, sample_length, sample_length, include_remainder=True
                )
            # single explanation for all the provided samples
            start = timer()
            explanations = self.predict_explainer_func(samples)
            explanation = dict()
            for e in explanations:
                if "feature_to_intervals" in e:
                    raise NotImplementedError(
                        "Feature intervals are not supported in model explainer evaluation."
                    )
                for ft, importance in e["feature_to_importance"].items():
                    if ft not in explanation:
                        explanation[ft] = 0
                    explanation[ft] += importance
            end = timer()
            n_explained = len(explanations)
        else:
            # the instance contains the target anomaly
            anomaly_start, anomaly_end = extract_binary_ranges_ids(
                (instance_labels == self.target_anomaly_class).astype(int)
            )[0]
            anomaly_length = anomaly_end - anomaly_start

            # equal anomaly and sample lengths (expansion was only performed for ED1 instability computation)
            if anomaly_length == sample_length:
                start = timer()
                explanation = self.predict_explainer_func(
                    [instance[anomaly_start:anomaly_end]]
                )[0]
                end = timer()
                n_explained = 1
            else:
                # anomaly is smaller than the sample: construct sample depending on the anomaly expansion
                n_prepended = anomaly_start
                n_appended = instance_length - anomaly_end
                # remove the records only added for ED1 consistency computation
                n_removed = self.min_instance_length - sample_length
                if n_prepended == 0:
                    sample_range = [0, instance_length - n_removed]
                elif n_appended == 0:
                    sample_range = [n_removed, instance_length]
                else:
                    half_n_removed = n_removed / 2
                    # try removing half before and half after the anomaly (more before if odd)
                    n_removed_before, n_removed_after = ceil(half_n_removed), floor(
                        half_n_removed
                    )
                    # remove after if not enough prepended records
                    if n_prepended - n_removed_before < 0:
                        n_removed_after += n_removed_before - n_prepended
                        n_removed_before = n_prepended
                    sample_range = [n_removed_before, instance_length - n_removed_after]
                start = timer()
                explanation = self.predict_explainer_func(
                    [instance[slice(*sample_range)]]
                )[0]
                end = timer()
                n_explained = 1
        return explanation, (end - start) / n_explained

    def get_perturbed_features(self, instance, instance_labels, explanation=None):
        """Model explainer implementation.

        For model explainers, "perturbations" are defined differently depending on
        the instance length and anomaly coverage policy:

        - For instances of minimal length, perturbations are defined as 1-step sliding windows to the right.
        - For others, perturbations are defined depending on the anomaly coverage policy:
            - "Center" coverage: as 1-step sliding windows alternately to the right and left.
            - "End" coverage: as 1-step sliding windows to the left.
            - "All" coverage: as random samples of `n` windows among all possible 1-step sliding ones
                in the instance. Where `n` is the number of windows used when explaining the original instance.

        In all above cases, the explanatory features of the original instance are (by convention)
        included in the instability computation.
        """
        instance_length = len(instance)
        sample_length = self.window_size
        n_perturbations = self.ed1_instability_n_perturbations
        if instance_length == self.min_instance_length:
            # instance of minimal length: no initialization and slide through the whole instance
            samples_fts = []
            sub_instance = instance
        else:
            # larger instance: initialize with the explanation of the original instance
            if explanation is not None:
                e = explanation
            else:
                e, _ = self.explain_instance(instance, instance_labels)
            samples_fts = [list(e["feature_to_importance"].keys())]
            if self.large_anomalies_coverage == "end":
                # slide through the end of the instance if "end" anomaly coverage
                sub_instance = instance[
                    (instance_length - sample_length - n_perturbations + 1) : -1
                ]
        if (
            instance_length == self.min_instance_length
            or self.large_anomalies_coverage == "end"
        ):
            samples = get_sliding_windows(sub_instance, sample_length, 1)
            samples_explanations = self.predict_explainer_func(samples)
            samples_fts += [
                list(s_e["feature_to_importance"].keys())
                for s_e in samples_explanations
            ]
        elif self.large_anomalies_coverage == "center":
            # alternately slide to the right and left of the anomaly center if "center" anomaly coverage
            center_start = floor((instance_length - sample_length) / 2)
            slide = 1
            slide_sign = 1
            for _ in range(self.ed1_instability_n_perturbations - 1):
                start = center_start + slide_sign * slide
                sample_explanation = self.predict_explainer_func(
                    [instance[start : (start + sample_length)]]
                )[0]
                samples_fts.append(
                    list(sample_explanation["feature_to_importance"].keys())
                )
                slide_sign *= -1
                if slide_sign == 1:
                    slide += 1
        else:
            # construct every possible samples from the instance if "all" anomaly coverage
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            samples_pool = get_sliding_windows(instance, sample_length, 1)
            # randomly sample the same number of samples as used for explaining the original instance
            for _ in range(n_perturbations):
                # any remaining sample was included when explaining the original instance
                samples = samples_pool[
                    random.sample(
                        range(len(samples_pool)), ceil(instance_length / sample_length)
                    )
                ]
                samples_explanations = self.predict_explainer_func(samples)
                samples_features = [
                    set(s_e["feature_to_importance"].keys())
                    for s_e in samples_explanations
                ]
                samples_fts.append(list(set().union(*samples_features)))
        return samples_fts

    def get_ed1_accuracy(self, instance: np.array, instance_labels: np.array = None):
        """For model explainers, ED1 accuracy is not defined."""
        return np.full(3, np.nan)

    def get_ed2_accuracy(self, explanations, instances, instances_labels=None):
        """For model explainers, ED2 accuracy is not defined."""
        return np.full(3, np.nan)
