"""Data explainers helpers module.
"""
import numpy as np


def get_split_sample(sample: np.array, sample_labels: np.array) -> (np.array, np.array):
    """Returns the original `sample` split into its "normal" and "anomalous" parts.

    Args:
        sample: sample of shape `(sample_length, n_features)`.
        sample_labels: sample labels of shape `(sample_length,)`.

    Returns:
        The normal and anomalous records of the sample, respectively.
    """
    return sample[sample_labels == 0], sample[sample_labels > 0]


def get_merged_sample(
    normal_records: np.array, anomalous_records: np.array, positive_class: int = 1
) -> (np.array, np.array):
    """Returns the provided normal and anomalous records as a single explanation sample and labels.

    Args:
        normal_records: normal records of shape `(n_normal_records, n_features)`.
        anomalous_records: anomalous records of shape `(n_anomalous_records, n_features)`.
        positive_class: positive class to use for the anomaly labels (default 1).

    Returns:
        The explanation sample and corresponding labels, respectively.
    """
    sample = np.concatenate([normal_records, anomalous_records])
    sample_labels = np.concatenate(
        [
            np.zeros(len(normal_records)),
            positive_class * np.ones(len(anomalous_records)),
        ]
    ).astype(int)
    return sample, sample_labels
