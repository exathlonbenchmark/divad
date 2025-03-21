import logging
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA as sklearn_PCA

from utils.guarding import check_value_in_choices
from features.transformers import plot_explained_variance
from detection.detectors.helpers.general import (
    get_flattened_windows,
    get_packed_windows,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.base import BaseDetector


class Pca(BaseDetector):
    relevant_steps = [
        "make_window_datasets",
        "train_window_model",
        "train_window_scorer",
        "train_online_scorer",
        "evaluate_online_scorer",
        "train_online_detector",
        "evaluate_online_detector",
    ]
    fitting_steps = ["train_window_model", "train_online_detector"]
    window_model_param_names = ["pca_"]
    window_scorer_param_names = ["method_", "n_selected_components_"]
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.pca_ = None
        self.method_ = None
        self.n_selected_components_ = None

    def _fit_window_model(
        self,
        X_train: np.array,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ) -> None:
        X_train, y_train, X_val, y_val = get_normal_windows(
            X_train, y_train, X_val, y_val
        )
        logging.info("Memory used after removing anomalies:")
        log_windows_memory(X_train, X_val)
        self.pca_ = sklearn_PCA()
        self.pca_.fit(get_flattened_windows(X_train))
        plot_explained_variance(
            self.pca_, explained_variance=1.0, output_path=self.window_model_path
        )

    def _predict_window_model(self, X):
        return get_packed_windows(self.pca_.transform(get_flattened_windows(X)))

    def set_window_scorer_params(
        self,
        method: str,
        n_selected_components: Optional[Union[int, float]] = None,
    ) -> None:
        """Sets hyperparameters relevant to the window scorer.

        Args:
            method: scoring method (either "reconstruction" or "mahalanobis").
            n_selected_components: number of components to consider in scoring (the
              highest-variance axes for "reconstruction", the lowest-variance axes for "mahalanobis").
        """
        check_value_in_choices(method, "method", ["reconstruction", "mahalanobis"])
        if not (
            n_selected_components is None
            or isinstance(n_selected_components, int)
            or 0 < n_selected_components < 1
        ):
            raise ValueError(
                f"Received invalid `n_selected_components`: {n_selected_components}."
            )
        self.method_ = method
        self.n_selected_components_ = n_selected_components

    def _get_n_selected_components(self):
        """Returns the number of components based on `self.n_selected_components_`."""
        if self.n_selected_components_ == -1:
            n_selected_components = self.pca_.components_.shape[0]
        elif isinstance(self.n_selected_components_, int):
            n_selected_components = self.n_selected_components_
        else:
            cumulative_variance = np.cumsum(self.pca_.explained_variance_ratio_)
            n_selected_components = (
                np.where(cumulative_variance > self.n_selected_components_)[0][0] + 1
            )
        return n_selected_components

    def transform_windows(self, X):
        transformed_X = None
        n_selected_components = self._get_n_selected_components()
        reshaped_X = get_flattened_windows(X)
        if self.method_ == "reconstruction":
            # select `n_selected_components` with the largest variance and place them as columns
            V = self.pca_.components_[:n_selected_components].T
            # transformed X in the reduced component space (projection)
            transformed_X = reshaped_X.dot(V)
        elif self.method_ == "mahalanobis":
            # select `n_selected_components` with the smallest variance and place them as columns
            V = self.pca_.components_[-n_selected_components:].T
            lambdas = self.pca_.explained_variance_[-n_selected_components:]
            # transformed X in the reduced component space (whitening)
            transformed_X = reshaped_X.dot(V) / np.sqrt(np.maximum(lambdas, 1e-7))
        return transformed_X

    def _predict_window_scorer(self, X):
        n_selected_components = self._get_n_selected_components()
        reshaped_X = get_flattened_windows(X)
        if self.method_ == "reconstruction":
            # select `n_selected_components` with the largest variance and place them as columns
            V = self.pca_.components_[:n_selected_components].T
            # transformed/projected X expressed in the original space
            projected_X = reshaped_X.dot(V).dot(V.T)
            # squared norm without computing the square roots for numerical stability
            window_scores = np.sum(np.square(reshaped_X - projected_X), axis=1)
        elif self.method_ == "mahalanobis":
            # select `n_selected_components` with the smallest variance and place them as columns
            V = self.pca_.components_[-n_selected_components:].T
            lambdas = self.pca_.explained_variance_[-n_selected_components:]
            # transformed X in the reduced component space
            transformed_X = reshaped_X.dot(V)
            # squared norm without computing square roots for numerical stability
            window_scores = np.sum(np.square(transformed_X) / lambdas, axis=1)
        else:
            raise NotImplementedError(
                'Only "reconstruction" and "mahalanobis" methods are supported.'
            )
        return window_scores
