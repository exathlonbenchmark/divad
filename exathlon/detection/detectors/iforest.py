import logging
from typing import Union

from sklearn.ensemble import IsolationForest

from detection.detectors.helpers.general import (
    get_flattened_windows,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.base import BaseDetector


class Iforest(BaseDetector):
    relevant_steps = [
        "make_window_datasets",
        "train_window_scorer",
        "train_online_scorer",
        "evaluate_online_scorer",
        "train_online_detector",
        "evaluate_online_detector",
    ]
    fitting_steps = ["train_window_scorer", "train_online_detector"]
    window_model_param_names = []
    window_scorer_param_names = [
        "iforest",
        "drop_anomalies",
        "n_estimators",
        "max_samples",
        "max_features",
        "contamination",
        "random_state",
    ]
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.iforest = None
        self.drop_anomalies = None
        self.n_estimators = None
        self.max_samples = None
        self.max_features = None
        self.contamination = None
        self.random_state = None

    def set_window_scorer_params(
        self,
        drop_anomalies: bool = False,
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = "auto",
        max_features: Union[int, float] = 1.0,
        contamination: Union[str, float] = "auto",
        random_state: int = 0,
    ) -> None:
        """Sets hyperparameters relevant to the window scorer."""
        self.drop_anomalies = drop_anomalies
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state

    def _fit_window_scorer(self, X_train, y_train=None, X_val=None, y_val=None):
        if self.drop_anomalies:
            X_train, y_train, X_val, y_val = get_normal_windows(
                X_train, y_train, X_val, y_val
            )
            logging.info("Memory used after removing anomalies:")
            log_windows_memory(X_train, X_val)
        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.random_state,
            bootstrap=False,
        )
        self.iforest.fit(get_flattened_windows(X_train))

    def _predict_window_scorer(self, X):
        return -self.iforest.score_samples(get_flattened_windows(X))
