import abc
import logging
from typing import Optional, Union, List, Tuple, Callable

import numpy as np
from scipy.stats import entropy

from data.helpers import get_numpy_from_numpy_list, get_nansum
from detection.metrics.helpers import extract_binary_ranges_ids


class NoInstanceError(Exception):
    """Exception raised in case no evaluation instance could be extracted from the data.

    Args:
        message (str): description of the error.
    """

    def __init__(self, message):
        self.message = message


class BaseExplainerEvaluator:
    """Explainer evaluator base class.

    Defines the base class for computing relevant ED metrics, including:

    - Proportion of "covered" instances (i.e., having sufficient length to be considered
        by the ED method / in the evaluation).
    - Among the covered instances, proportion of "explained" instances (i.e., for which the ED method
        found important features). Only those instances will be considered when computing other metrics.
    - Average inference (i.e., explanation) time.
    - ED1/2 conciseness.
    - ED1/2 normalized consistency.
    - For the methods supporting it, ED1/2 accuracy (i.e., point-based Precision, Recall and F1-score).

    * Proportion of covered/explained instances, inference time, ED1 consistency and ED1 accuracy
        are returned as dictionaries whose keys are:
        - "mixed", considering all non-zero labels as a single positive anomaly class.
        - every distinct non-zero label encountered in the data, considering the corresponding class only.
        - "avg", corresponding to the average scores across each positive class (excluding the "mixed" key).
    * ED2 metrics are returned as similar dictionaries, but without the "mixed" key.
        ED2 metrics are indeed only defined across anomalies of the same type.
    * ED1 conciseness is returned as a single value, corresponding to the "mixed" key only.
        Considering ED1 conciseness per class would indeed be redundant with ED2 conciseness.

    Args:
        predict_explainer_func: explanation function.
        min_anomaly_length: minimum anomaly length for an instance to be considered in the evaluation.
        ed1_instability_n_perturbations: number of perturbations to perform when computing instance-based instability.
    """

    def __init__(
        self,
        predict_explainer_func: Callable,
        min_anomaly_length: int = 1,
        ed1_instability_n_perturbations: int = 5,
    ):
        self.min_anomaly_length = min_anomaly_length
        self.ed1_instability_n_perturbations = ed1_instability_n_perturbations
        # instance-based metrics
        self.ed1_metric_names = [
            "time",
            "prop_covered",
            "prop_explained",
            "size",
            "perturbed_size",
            "instability",
            "f1_score",
            "precision",
            "recall",
        ]
        # type-wise cross-instance metrics
        self.ed2_metric_names = [
            "discordance",
            "f1_score",
            "precision",
            "recall",
        ]
        self.metric_names = [f"ed1_{m}" for m in self.ed1_metric_names] + [
            f"ed2_{m}" for m in self.ed2_metric_names
        ]
        self.predict_explainer_func = predict_explainer_func

    def compute_metrics(
        self,
        sequences: np.array,
        sequences_labels: np.array,
        sequences_info: List[list],
    ) -> (dict, dict):
        """Returns the relevant ED metrics and explanations for the provided `periods` and `periods_labels`.

        Metrics are returned as a single dictionary with as keys the metric names and as
        values the metric values, as described in the class documentation.

        Returned format explanations:

        ```
        {
            seq_id1: {
                # anomaly chronological rank + type index
                0_T1: {
                    "feature_to_importance": {"ft1": importance_score, "ft67": 2.3},
                    "other_format_key": {...}
                },
                1_T1: {...}
            },
            seq_id1: {...}
        }
        ```
        Args:
            sequences (ndarray): periods data of shape `(n_periods, period_length, n_features)`.
                With `period_length` depending on the period.
            sequences_labels (ndarray): labels for each period of shape `(n_periods, period_length)`.
                With `period_length` depending on the period.
            sequences_info (list): list of periods information.

        Returns:
            Relevant ED metrics with a key per anomaly type, "avg", "mixed", and formatted explanations.
        """
        pos_classes = list(
            np.delete(np.unique(np.concatenate(sequences_labels, axis=0)), 0)
        )
        include_ed2 = len(pos_classes) > 1
        # evaluation instances, labels and information per positive class (pc)
        (
            pc_to_instances,
            pc_to_instances_labels,
            pc_to_instances_info,
            pc_to_prop_covered,
        ) = self.get_evaluation_instances(
            sequences, sequences_labels, sequences_info, pos_classes=pos_classes
        )
        metric_to_type_to_value = {
            m: dict() for m in self.metric_names if include_ed2 or "ed2" not in m
        }
        pc_to_explained_instances = dict()
        pc_to_explained_instances_labels = dict()
        pc_to_explained_instances_info = dict()
        pc_to_explanations = dict()
        pc_to_n_explained = dict()
        pc_to_n_covered = dict()
        for pc, pc_instances in pc_to_instances.items():
            logging.info(f"Computing metrics for instances of type {pc}.")
            pc_instances_labels = pc_to_instances_labels[pc]
            # instances explanations, average inference time and explained instances mask
            (explanations, avg_exp_time, explanation_mask) = self.explain_instances(
                pc_instances, pc_instances_labels
            )
            prop_explained = sum(explanation_mask) / len(pc_instances)
            explained_pc_instances = pc_to_instances[pc][explanation_mask]
            pc_to_explained_instances[pc] = explained_pc_instances
            explained_pc_instances_labels = pc_to_instances_labels[pc][explanation_mask]
            pc_to_explained_instances_labels[pc] = explained_pc_instances_labels
            pc_to_explained_instances_info[pc] = list(
                np.array(pc_to_instances_info[pc])[explanation_mask]
            )
            pc_to_explanations[pc] = explanations
            n_explained_instances = len(explanations)
            pc_to_n_explained[pc] = n_explained_instances
            pc_to_n_covered[pc] = len(pc_instances)
            if prop_explained > 0:
                # compute instance-based (ED1) metrics
                ed1_sizes = []
                ed1_perturbed_sizes = []
                ed1_instabilities = []
                ed1_f1_scores = []
                ed1_precisions = []
                ed1_recalls = []
                for instance, instance_labels, explanation in zip(
                    explained_pc_instances, explained_pc_instances_labels, explanations
                ):
                    # explanation size
                    size = len(explanation["feature_to_importance"])
                    ed1_sizes.append(size)
                    # perturbed size and normalized instability
                    raw_instability = self.get_ed1_raw_instability(
                        instance, instance_labels, explanation
                    )
                    perturbed_size = 2**raw_instability
                    ed1_perturbed_sizes.append(perturbed_size)
                    ed1_instabilities.append(perturbed_size / size)
                    # accuracy
                    f1, p, r = self.get_ed1_accuracy(instance, instance_labels)
                    ed1_f1_scores.append(f1)
                    ed1_precisions.append(p)
                    ed1_recalls.append(r)
                for m, v in zip(
                    ["ed1_time", "ed1_prop_covered", "ed1_prop_explained"],
                    [avg_exp_time, pc_to_prop_covered[pc], prop_explained],
                ):
                    metric_to_type_to_value[m][pc] = v
                for m, l in zip(
                    [
                        "ed1_size",
                        "ed1_perturbed_size",
                        "ed1_instability",
                        "ed1_f1_score",
                        "ed1_precision",
                        "ed1_recall",
                    ],
                    [
                        ed1_sizes,
                        ed1_perturbed_sizes,
                        ed1_instabilities,
                        ed1_f1_scores,
                        ed1_precisions,
                        ed1_recalls,
                    ],
                ):
                    metric_to_type_to_value[m][pc] = np.sum(l) / n_explained_instances
                if include_ed2:
                    discordance = self.get_ed2_discordance(explanations)
                    ed2_f1, ed2_p, ed2_r = self.get_ed2_accuracy(
                        explanations,
                        explained_pc_instances,
                        explained_pc_instances_labels,
                    )
                    for m, v in zip(
                        [
                            "ed2_discordance",
                            "ed2_f1_score",
                            "ed2_precision",
                            "ed2_recall",
                        ],
                        [discordance, ed2_f1, ed2_p, ed2_r],
                    ):
                        metric_to_type_to_value[m][pc] = v
            logging.info("done.")

        # average metrics across anomaly types
        logging.info("Averaging metrics across anomaly types.")
        for m in metric_to_type_to_value.keys():
            # will set to NaN with a warning if all values are NaN for a metric here
            metric_to_type_to_value[m]["avg"] = np.nanmean(
                [metric_to_type_to_value[m][k] for k in metric_to_type_to_value[m]]
            )
        logging.info("done.")

        # deduce metrics for the "mixed" class (i.e., considering all anomaly types the same)
        logging.info('Deriving metrics for the "mixed" class.')
        tot_covered_instances = sum(pc_to_n_covered.values())
        tot_explained_instances = sum(pc_to_n_explained.values())
        for m in metric_to_type_to_value:
            # this "mixed" consideration does not make sense for type-based (ED2) metrics
            if "ed2" not in m:
                # only consider explained instances for all metrics except the proportion of explained instances
                if m != "prop_explained":
                    pc_to_n_relevant = pc_to_n_explained
                    tot_relevant = tot_explained_instances
                # for the proportion of explained instances, consider the number of covered instances
                else:
                    pc_to_n_relevant = pc_to_n_covered
                    tot_relevant = tot_covered_instances
                if tot_explained_instances > 0:
                    metric_to_type_to_value[m]["mixed"] = (
                        get_nansum(
                            [
                                pc_to_n_relevant[pc] * metric_to_type_to_value[m][pc]
                                for pc in pc_to_instances
                            ]
                        )
                        / tot_relevant
                    )
                else:
                    # no explained instances at all
                    metric_to_type_to_value[m]["mixed"] = (
                        0 if m == "prop_covered" else np.nan
                    )
        logging.info("done.")

        # format type-wise explanations to include the instances information
        seq_to_ano_to_explanation = dict()
        for pc, explanations in pc_to_explanations.items():
            for i, explanation in enumerate(explanations):
                instance_info = pc_to_explained_instances_info[pc][i]
                # the chronological rank of the instance in the period is always last
                instance_idx = instance_info[-1]
                # TODO: make it more general, for now only works for spark formatted information.
                # for spark data, the first element of the period information is the file name
                file_name = instance_info[0]
                # group explanations by file name
                if file_name not in seq_to_ano_to_explanation:
                    seq_to_ano_to_explanation[file_name] = dict()
                # within a file, label each explanation with its instance rank and type
                seq_to_ano_to_explanation[file_name][
                    f"{instance_idx}_T{pc}"
                ] = explanation

        # return explanation metrics and formatted explanations
        return metric_to_type_to_value, seq_to_ano_to_explanation

    def get_evaluation_instances(
        self,
        sequences: np.array,
        sequences_labels: np.array,
        sequences_info: list,
        pos_classes: Optional[np.array] = None,
    ) -> (dict, dict, dict, dict):
        """Returns evaluation instances, labels and information from the provided `periods`, `periods_labels`
            and `periods_info`.

        TODO: should be defined at the sequence level: just get a set of instances for a given sequence.

        Raises `NoInstanceError` if no evaluation instance could be extracted from the
        provided data. Else, provides along with the instances and labels the proportion
        of anomalies that will be covered in the evaluation (grouped by type).

        Instances information is returned as lists of the form `[*period_info, instance_idx]`, where
        `period_info` is the information of the period the instance belongs to, and `instance_idx` is
        the chronological rank of the instance in that period.

        Args:
            sequences: sequences records of shape `(n_sequences, seq_length, n_features)`.
                With `seq_length` depending on the period.
            sequences_labels (ndarray): multiclass sequences labels of shape `(n_sequences, seq_length)`.
                With `seq_length` depending on the period.
            sequences_info (list): list of sequences information.
            pos_classes: if already computed, can be passed so as not to compute them again.

        Returns:
            Instances, corresponding labels and information, as well as proportions of anomalies covered
              per anomaly type, with as keys the relevant (numerical) anomaly types and as values
              ndarrays/lists/floats.
        """
        if pos_classes is None:
            pos_classes = np.delete(
                np.unique(np.concatenate(sequences_labels, axis=0)), 0
            )
        pc_to_n_ranges = dict()
        pc_to_prop_covered = dict()
        pc_to_instances = dict()
        pc_to_instances_labels = dict()
        pc_to_instances_info = dict()
        for pc in pos_classes:
            pc_to_n_ranges[pc] = 0
            pc_to_instances[pc] = []
            pc_to_instances_labels[pc] = []
            pc_to_instances_info[pc] = []
        for seq, seq_labels, seq_info in zip(
            sequences, sequences_labels, sequences_info
        ):
            # instances, labels and information grouped by anomaly type for the sequence
            seq_pos_ranges = extract_binary_ranges_ids((seq_labels > 0).astype(int))
            seq_range_classes = np.array(
                [seq_labels[pos_range[0]] for pos_range in seq_pos_ranges]
            )
            for range_idx in range(len(seq_pos_ranges)):
                instance, instance_labels = self.get_evaluation_instance(
                    seq, seq_labels, seq_pos_ranges, range_idx
                )
                range_class = seq_range_classes[range_idx]
                pc_to_n_ranges[range_class] += 1
                if instance is not None:
                    pc_to_instances[range_class].append(instance)
                    pc_to_instances_labels[range_class].append(instance_labels)
                    pc_to_instances_info[range_class].append([*seq_info, range_idx])
        # turn data and labels lists to numpy arrays and return the results
        for pc in pc_to_instances:
            pc_to_instances[pc] = get_numpy_from_numpy_list(pc_to_instances[pc])
            pc_to_instances_labels[pc] = get_numpy_from_numpy_list(
                pc_to_instances_labels[pc]
            )
            # proportion of positive ranges covered in the evaluation (zero division should never happen)
            pc_to_prop_covered[pc] = pc_to_instances[pc].shape[0] / pc_to_n_ranges[pc]
        # raise error if no evaluation instance could be extracted
        if np.all([prop_covered == 0 for prop_covered in pc_to_prop_covered.values()]):
            raise NoInstanceError(
                "No evaluation instance could be extracted from the sequences."
            )
        return (
            pc_to_instances,
            pc_to_instances_labels,
            pc_to_instances_info,
            pc_to_prop_covered,
        )

    @abc.abstractmethod
    def get_evaluation_instance(
        self,
        sequence: np.array,
        sequence_labels: np.array,
        pos_ranges: np.array,
        range_idx: int,
    ) -> Union[Tuple[np.array, np.array], None]:
        """Returns the evaluation instance and labels corresponding to the anomaly range
            `pos_ranges[range_idx]` in `period`.

        If the evaluation instance cannot be extracted, due to the positive range violating
        a given constraint (e.g., minimum length), then None values should be returned.

        Args:
            sequence: sequence records of shape `(seq_length, n_features)`.
            sequence_labels: corresponding multiclass labels of shape `(seq_length,)`.
            pos_ranges: start and (excluded) end indices of every anomaly range in the
                sequence, of the form `[[start_1, end_1], [start_2, end_2], ...]`.
            range_idx (int): index in `pos_ranges` of the anomaly range for which to return the
                instance and labels.

        Returns:
            Evaluation instance and labels corresponding to the anomaly range,
              of respective shapes `(instance_length, n_features)` and `(instance_length,)`, or None
              if the range could not be extracted.
        """
        pass

    def explain_instances(
        self, instances: np.array, instances_labels: np.array
    ) -> (List[dict], float, list):
        """Returns the explanations of the provided `instances` along with the average
            explanation time and proportion of explained instances.

        If an instance could not be explained by the ED method (i.e., no "important features" were
        found for it), it will be ignored when computing ED metrics, but reflected in a metric called
        "proportion of explained instances".

        Args:
            instances (ndarray): evaluation instances of shape `(n_instances, instance_length, n_features)`.
                With `instance_length` depending on the instance.
            instances_labels (ndarray): instances labels of shape `(n_instances, instance_length)`.
                With `instance_length` depending on the instance.

        Returns:
            Instances explanations, average explanation time and explanation boolean mask.
        """
        explanations = []
        explanation_mask = []
        tot_time = 0
        for instance, instance_labels in zip(instances, instances_labels):
            explanation, explanation_time = self.explain_instance(
                instance, instance_labels
            )
            was_explained = len(explanation["feature_to_importance"]) > 0
            explanation_mask.append(was_explained)
            if was_explained:
                explanations.append(explanation)
                tot_time += explanation_time
        n_explained = len(explanations)
        avg_time = tot_time / n_explained if n_explained > 0 else 0
        return explanations, avg_time, explanation_mask

    @abc.abstractmethod
    def explain_instance(
        self, instance: np.array, instance_labels: np.array
    ) -> (dict, float):
        """Returns the explanation of the provided `instance` along with the explanation time.

        According to the type of evaluated method and instance definition, the explanation
        of an *instance* might be derived from the explanations of *samples* in various ways.

        Args:
            instance (ndarray): evaluation instance of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.

        Returns:
            Instance explanation and explanation time in seconds.
        """

    def get_ed1_raw_instability(
        self, instance: np.array, instance_labels: np.array, explanation: dict = None
    ) -> float:
        """Returns the ED1 consistency (i.e., stability) score of the provided `instance`.

        This metric is defined as the (possibly normalized) consistency of explanatory
        features across different "disturbances" of the explained instance.

        Args:
            instance (ndarray): instance data of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.
            explanation (dict|None): optional instance explanation, that can be used to normalize
                the consistency score of the instance.

        Returns:
            The ED1 consistency score of the instance.
        """
        # inconsistency of explanatory features across different instance "perturbations"
        return self.get_features_inconsistency(
            self.get_perturbed_features(instance, instance_labels, explanation)
        )

    @abc.abstractmethod
    def get_perturbed_features(
        self,
        instance: np.array,
        instance_labels: np.array,
        explanation: Optional[dict] = None,
    ) -> List[list]:
        """Returns the explanatory features found for different "disturbances" of `instance`.

        The number of disturbances to perform is given by the value of
        `self.ed1_consistency_n_disturbances`, which might include the original instance or not.

        Args:
            instance (ndarray): instance data of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.
            explanation (dict|None): optional, pre-computed, explanation of `instance`.

        Returns:
            The explanatory features lists for each instance "perturbation".
        """

    @staticmethod
    def get_features_inconsistency(features_lists: List[list]) -> float:
        """Returns the "inconsistency" of the provided features lists.

        This "inconsistency" aims to capture the degree of disagreement between the features lists.
        We define it as the entropy of the lists' duplicate-preserving union (i.e., turning the
        features lists into a features bag).

        The smaller the entropy value, the less uncertain will be the outcome of randomly
        drawing an explanatory feature from the bag, and hence the more the lists of features
        will agree with each other.

        Args:
            features_lists: list of features lists whose consistency to compute.

        Returns:
            Inconsistency of the features lists.
        """
        features_bag = [ft for features_list in features_lists for ft in features_list]
        # unnormalized probability distribution of feature ids
        p_features = []
        for feature_id in set(features_bag):
            p_features.append(features_bag.count(feature_id))
        return entropy(p_features, base=2)

    @abc.abstractmethod
    def get_ed1_accuracy(
        self, instance: np.array, instance_labels: np.array
    ) -> (float, float, float):
        """Returns the ED1 accuracy metrics of the provided `instance`.

        These metrics aim to capture the local predictive power of the explanations derived by the
        ED method. This predictive power is measured as the point-wise Precision, Recall and F1-score
        achieved by explanations when used as anomaly detection rules around the explained anomaly.

        Args:
            instance (ndarray): instance data of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.

        Returns:
            ndarray: F1-score, Precision and Recall  for the instance, respectively.
        """

    def get_ed2_discordance(self, explanations: List[dict], normalized: bool = True):
        """Returns the ED2 discordance score of the provided `explanations`.

        This metric is defined as the (possibly normalized) discordance of explanatory features across
        the provided explanations, it is therefore only defined if multiple explanations are provided.

        Args:
            explanations: instance explanations dicts whose discordance to compute.
            normalized: whether to "normalize" the discordance score with respect to the
                average explanation size.

        Returns:
            The ED2 discordance score of the explanations (or NaN if a single explanation was provided).
        """
        if len(explanations) == 1:
            return np.nan
        explanations_fts = [
            list(explanation["feature_to_importance"].keys())
            for explanation in explanations
        ]
        fts_consistency = self.get_features_inconsistency(explanations_fts)
        if not normalized:
            return fts_consistency
        avg_explanation_length = sum([len(fts) for fts in explanations_fts]) / len(
            explanations
        )
        return (2**fts_consistency) / avg_explanation_length

    @abc.abstractmethod
    def get_ed2_accuracy(
        self, explanations: List[dict], instances: np.array, instances_labels: np.array
    ) -> (float, float, float):
        """Returns the ED2 accuracy metrics for the provided `explanations` and `instances`.

        These metrics aim to capture the "global" predictive power of the explanations derived
        by the ED method. This predictive power is measured as the point-wise F1-score, Precision and
        Recall achieved by explanations when used as anomaly detection rules around other anomalies
        of the same type.

        Like ED2 discordance, this metric is only defined across multiple evaluated instances.

        Args:
            explanations: list of instance explanations dicts.
            instances: corresponding instances, assumed to be of the same anomaly type,
                of shape `(n_instances, instance_length, n_features)`. With `instance_length`
                depending on the instance.
            instances_labels: instances labels of shape `(n_instances, instance_length)`.
                With `instance_length` depending on the instance.

        Returns:
            F1-score, Precision and Recall and F1-score (or NaNs if a single instance was provided).
        """
