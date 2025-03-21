import abc
from typing import Optional, List

import numpy as np

from utils.guarding import check_is_not_none
from explanation.explainers.base_explainer import BaseExplainer
from explanation.explainers.data_explainers.helpers.general import get_split_sample


class BaseDataExplainer(BaseExplainer):
    """Data explainer base class.

    Gathers common functionalities of all data explainers.

    These methods try to explain differences between a set of normal records and a set of
    anomalous records, independently of any anomaly detection model.
    """

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def predict(
        self,
        instances: List[np.array],
        instances_labels: Optional[List[np.array]] = None,
    ) -> List[dict]:
        # labels are mandatory for data explainers
        check_is_not_none(instances_labels, "instances_labels")
        explanations = []
        for instance, instance_labels in zip(instances, instances_labels):
            normal_records, anomalous_records = get_split_sample(
                instance, instance_labels
            )
            explanation = self.predict_instance(normal_records, anomalous_records)
            if "feature_to_importance" not in explanation:
                raise ValueError(
                    'Explanations should contain a "feature_to_importance" key.'
                )
            explanations.append(explanation)
        return explanations

    @abc.abstractmethod
    def predict_instance(
        self, normal_records: np.array, anomalous_records: np.array
    ) -> dict:
        """Returns the explanation for the normal and anomalous records."""
