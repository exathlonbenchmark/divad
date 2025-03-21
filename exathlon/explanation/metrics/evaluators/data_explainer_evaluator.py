"""
TODO: use twice the anomaly length (or a parameter) instead of all available previous normal data,
  to be more fair across anomaly instances.
"""
import random
import logging
from typing import Optional, List
from timeit import default_timer as timer

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from explanation.explainers.base_explainer import predict_decision_rule
from explanation.metrics.evaluators.base_explainer_evaluator import (
    BaseExplainerEvaluator,
)
from explanation.explainers.data_explainers.helpers.general import (
    get_split_sample,
    get_merged_sample,
)


class DataExplainerEvaluator(BaseExplainerEvaluator):
    """Data explainer evaluator class.

    Evaluates data explainers, trying to explain differences between a set of normal records
    and a set of anomalous records.

    For such methods, evaluation instances are the same as explained samples: data intervals
    comprised of a "normal" interval followed by an "anomalous" interval: [N; A].

    The explanation of an instance/sample is defined by the explanation discovery method,
    and must contain a set of "explanatory" or "important" feature indices.

    Args:
        min_normal_length: minimum normal data points per instance to compute ED1 instability and accuracy.
        ed1_instability_sampled_prop: proportion of records to sample when computing ED1 instability.
        ed1_accuracy_n_splits: number of random splits to use when computing ED1 accuracy.
        ed1_accuracy_test_prop: proportion of records to use as test when computing ED1 accuracy.
        random_seed: random seed to use when computing ED1 instability and accuracy.
    """

    def __init__(
        self,
        min_normal_length: int,
        ed1_instability_sampled_prop: float,
        ed1_accuracy_n_splits: int,
        ed1_accuracy_test_prop: float,
        random_seed: int = 0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        # a minimum length of 2 points per instance part is needed for computing ED1 instability and accuracy
        self.min_anomaly_length = max(2, self.min_anomaly_length)
        self.min_normal_length = max(2, min_normal_length)
        self.ed1_instability_sampled_prop = ed1_instability_sampled_prop
        self.ed1_accuracy_n_splits = ed1_accuracy_n_splits
        self.ed1_accuracy_test_prop = ed1_accuracy_test_prop
        self.random_seed = random_seed

    def get_evaluation_instance(self, period, period_labels, pos_ranges, range_idx):
        """Data explainer implementation.

        For data explainers, evaluation instances are the full anomaly ranges prepended
        with (all) their preceding normal data.
        """
        instance = None
        instance_labels = None
        pos_range = pos_ranges[range_idx]
        range_class = period_labels[pos_range[0]]
        normal_data = (
            (pos_range[0] != 0)
            if range_idx == 0
            else (pos_range[0] != pos_ranges[range_idx - 1][1])
        )
        if normal_data:
            instance_start = 0 if range_idx == 0 else pos_ranges[range_idx - 1][1]
            normal_length = pos_range[0] - instance_start
            anomaly_length = pos_range[1] - pos_range[0]
            if (
                normal_length >= self.min_normal_length
                and anomaly_length >= self.min_anomaly_length
            ):
                instance = period[instance_start : pos_range[1]]
                instance_labels = np.concatenate(
                    [np.zeros(normal_length), range_class * np.ones(anomaly_length)]
                )
            else:
                logging.warning(
                    f"Instance of type {range_class} dropped due to insufficient normal and/or anomaly length."
                )
        else:
            logging.warning(
                f"Instance of type {range_class} dropped due to absence of preceding normal data"
            )
        return instance, instance_labels

    def explain_instance(self, instance, instance_labels):
        """Model-free implementation.

        For model-free explainers, evaluation instances are the same as explained samples.
        """
        start = timer()
        explanation = self.predict_explainer_func([instance], [instance_labels])[0]
        end = timer()
        return explanation, end - start

    def get_perturbed_features(
        self,
        instance: np.array,
        instance_labels: np.array,
        explanation: Optional[dict] = None,
    ) -> List[list]:
        """Data explainer implementation.

        For data explainers, "perturbations" are defined as random samples separately
        drawn from the instance's normal and anomalous data.

        The explanatory features of the original instance are therefore not included in the
        consistency computation.
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        # separate normal from anomalous records
        normal_records, anomalous_records = get_split_sample(instance, instance_labels)
        n_normal_records = len(normal_records)
        n_anomalous_records = len(anomalous_records)
        # important (explanatory) features found for `ed1_instability_n_perturbations` random samples
        samples_fts = []
        # round below except if the resulting size is zero
        normal_sample_size = max(
            int(self.ed1_instability_sampled_prop * n_normal_records), 1
        )
        anomalous_sample_size = max(
            int(self.ed1_instability_sampled_prop * n_anomalous_records), 1
        )
        for _ in range(self.ed1_instability_n_perturbations):
            # draw a random sample from the normal and anomalous records
            normal_ids = random.sample(range(n_normal_records), normal_sample_size)
            anomalous_ids = random.sample(
                range(n_anomalous_records), anomalous_sample_size
            )
            # get important explanatory features for the sample
            sampled_instance = np.concatenate(
                [normal_records[normal_ids], anomalous_records[anomalous_ids]]
            )
            sampled_labels = np.concatenate(
                [np.zeros(len(normal_ids)), np.ones(len(anomalous_ids))]
            )
            sample_explanation = self.predict_explainer_func(
                [sampled_instance], [sampled_labels]
            )[0]
            samples_fts.append(list(sample_explanation["feature_to_importance"].keys()))
        return samples_fts

    def get_ed1_accuracy(
        self, instance: np.array, instance_labels: np.array
    ) -> (float, float, float):
        """Data explainer implementation.

        For data explainers, the normal and anomalous records of the instance are
        randomly split into a training and a test set (according to `self.ed1_accuracy_test_prop`).

        The explanation is derived on the training set and evaluated on the test set. The
        final performance is returned averaged across `self.ed1_accuracy_n_splits` random splits.
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        # extract the normal and anomalous records from the instance
        normal_records, anomalous_records = get_split_sample(instance, instance_labels)
        # get accuracy scores averaged across `ed1_accuracy_n_splits` random splits
        accuracy_scores = np.zeros(3)
        for _ in range(self.ed1_accuracy_n_splits):
            # get training and test instances
            (
                normal_train,
                normal_test,
            ) = train_test_split(normal_records, test_size=self.ed1_accuracy_test_prop)
            (
                anomalous_train,
                anomalous_test,
            ) = train_test_split(
                anomalous_records, test_size=self.ed1_accuracy_test_prop
            )
            test_instance, test_instance_binary_labels = get_merged_sample(
                normal_test, anomalous_test, 1
            )
            # derive explanation for the training instance
            train_instance = np.concatenate([normal_train, anomalous_train])
            train_instance_labels = np.concatenate(
                [np.zeros(len(normal_train)), np.ones(len(anomalous_train))]
            )
            train_explanation = self.predict_explainer_func(
                [train_instance], [train_instance_labels]
            )[0]
            # evaluate classification performance on the test instance
            test_instance_preds = predict_decision_rule(
                test_instance, train_explanation["feature_to_intervals"]
            )
            p, r, f1 = precision_recall_fscore_support(
                test_instance_binary_labels,
                test_instance_preds,
                average="binary",
                zero_division=0,
            )[:3]
            accuracy_scores += [f1, p, r]
        # return the average f1-score, precision and recall across splits
        return accuracy_scores / self.ed1_accuracy_n_splits

    def get_ed2_accuracy(self, explanations, instances, instances_labels):
        """Data explainer implementation.

        Since instances of data explainers contain both normal and anomalous records,
        an explanation derived for an instance can directly be evaluated on others.
        """
        if len(instances) == 1:
            return np.full(3, np.nan)
        accuracy_scores = np.zeros(3)
        n_explanations = len(explanations)
        for i, explanation in enumerate(explanations):
            explanation_accuracy_scores = np.zeros(3)
            for j in range(n_explanations):
                if j != i:
                    y_preds = predict_decision_rule(
                        instances[i], explanation["feature_to_intervals"]
                    )
                    p, r, f1 = precision_recall_fscore_support(
                        (instances_labels[i] > 0).astype(int),
                        y_preds,
                        average="binary",
                        zero_division=0,
                    )[:3]
                    explanation_accuracy_scores += [f1, p, r]
            # add average performance of the explanation across test instances
            accuracy_scores += explanation_accuracy_scores / (n_explanations - 1)
        # return average performance across explanations
        return accuracy_scores / n_explanations
