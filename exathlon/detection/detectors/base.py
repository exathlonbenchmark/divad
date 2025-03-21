"""
TODO:
 - Rename to base_detector.py.
 - Add (return) type hints for `relevant_steps` and `fitting_steps`.
 - Remove param_names attributes, and simply check for the main model of the step in the base class.

TODO: For explanation capabilities, should provide `set_explainer_params(), `fit_explainer()` and
 `predict_explainer()` methods (predict or "instances" or "samples", see that. Check that methods are there
 in `train_explainer` and `evaluate_explainer` scripts. Should also have a `feature_names_` attribute.
"""
import abc

from omegaconf import DictConfig
from typing import Optional, List, Tuple
from timeit import default_timer as timer

import numpy as np
import pandas as pd

# from run import STEPS
from utils.guarding import (
    check_value_in_choices,
    check_all_values_in_choices,
    check_is_not_none,
)
from data.helpers import get_sliding_windows, get_downsampled_windows, save_files
from detection.detectors.helpers.general import log_windows_memory
from detection.metrics.evaluation import get_best_score_threshold
from detection.metrics.evaluators.point import PointEvaluator
from detection.metrics.evaluators.range import RangeEvaluator


class BaseDetector(metaclass=abc.ABCMeta):
    """Detector base class.

    Args:
        window_model_path: output path of the window model, if relevant.
        window_scorer_path: output path of the window scorer, if relevant.
        online_scorer_path: output path of the online scorer.
        online_detector_path: output path of the online detector.
    """

    def __init__(
        self,
        window_model_path: Optional[str] = None,
        window_scorer_path: Optional[str] = None,
        online_scorer_path: str = ".",
        online_detector_path: str = ".",
    ):
        self.window_model_path = window_model_path
        self.window_scorer_path = window_scorer_path
        self.online_scorer_path = online_scorer_path
        self.online_detector_path = online_detector_path
        # only relevant for window-based detectors
        # window size actually used by models
        self.window_size_ = None
        # window size before upstream downsampling, and downsampling parameters
        self.pre_downsampling_window_size_ = None
        self.downsampling_size_ = None
        self.downsampling_step_ = None
        self.downsampling_func_ = None
        # default online scoring parameters (shared by all methods if used)
        if type(self).set_online_scorer_params == BaseDetector.set_online_scorer_params:
            self.online_scorer_param_names.append("scores_avg_beta_")
        self.scores_avg_beta_ = None
        # default online detection parameters (shared by all methods if used)
        if (
            type(self).set_online_detector_params
            == BaseDetector.set_online_detector_params
        ):
            for p in ["online_scorer_evaluator_", "maximized_f_score_"]:
                self.online_detector_param_names.append(p)
            if type(self)._fit_online_detector == BaseDetector._fit_online_detector:
                self.online_detector_param_names.append("score_threshold_")
        self.online_scorer_evaluator_ = None
        self.maximized_f_score_ = None
        self.score_threshold_ = None
        # check_all_values_in_choices(self.relevant_steps, "self.relevant_steps", STEPS)
        self.component_to_param_names = {
            "window_model": self.window_model_param_names,
            "window_scorer": self.window_scorer_param_names,
            "online_scorer": self.online_scorer_param_names,
            "online_detector": self.online_detector_param_names,
        }

    @property
    @abc.abstractmethod
    def fitting_steps(self):
        """Ordered sequence of steps for which the detection method requires some *fitting*.

        This enables not to load training data for steps that are only relevant to set hyperparameters.
        """

    @property
    @abc.abstractmethod
    def relevant_steps(self):
        """Ordered sequence of steps relevant to the detection method (including evaluation).

        A pipeline step is "relevant" to a detector object if it evaluates it, or changes it
        by either setting hyperparameters or fitting parameters.
        """

    @classmethod
    def get_previous_relevant_steps(cls, step: str) -> list:
        """Returns the relevant steps strictly before `step` for the detector (empty list if none).

        Previous relevant steps exclude evaluation steps, which are not required.
        """
        # check_value_in_choices(step, "step", STEPS)
        try:
            step_idx = cls.relevant_steps.index(step)
            if step_idx == 0:
                return []
            # evaluation scripts are not mandatory *previous* relevant steps
            return [s for s in cls.relevant_steps[:step_idx] if "evaluate" not in s]
        except IndexError:
            if step not in cls.relevant_steps:
                raise ValueError(
                    f"Step {step} is not relevant to {type(cls.__name__)}."
                )

    @property
    @abc.abstractmethod
    def window_model_param_names(self) -> list:
        """Parameter names for the window model (empty if not relevant or no parameters)."""

    @property
    @abc.abstractmethod
    def window_scorer_param_names(self) -> list:
        """Parameter names for the window scorer (empty if not relevant or no parameters)."""

    @property
    @abc.abstractmethod
    def online_scorer_param_names(self) -> list:
        """Parameter names for the online scorer (empty if not relevant or no parameters)."""

    @property
    @abc.abstractmethod
    def online_detector_param_names(self) -> list:
        """Parameter names for the online detector (empty if not relevant or no parameters)."""

    def check_components(self, components: list) -> None:
        """Checks whether all parameters of the provided components have been fit/set (i.e., not `None`)."""
        check_all_values_in_choices(
            components, "components", list(self.component_to_param_names.keys())
        )
        for c in components:
            for p_name in self.component_to_param_names[c]:
                if getattr(self, p_name) is None:
                    raise ValueError(f"Missing required {c} ({p_name} is None).")

    def set_window_model_params(self, **params) -> None:
        """Sets hyperparameters relevant to the window model."""
        pass

    def fit_window_model(
        self,
        X_train: np.array,
        y_train: Optional[np.array] = None,
        X_val: Optional[np.array] = None,
        y_val: Optional[np.array] = None,
        train_info: Optional[dict] = None,
        val_info: Optional[dict] = None,
    ) -> None:
        """Fits parameters of the window model to a surrogate task assumed useful to anomaly detection.

        Args:
            X_train: training windows of shape `(n_train, window_size, n_features)`.
            y_train: training labels of shape `(n_train,)`, unused by unsupervised methods.
            X_val: validation windows of shape `(n_val, window_size, n_features)`.
            y_val: validation labels of shape `(n_val,)`, unused by unsupervised methods.
            train_info: training windows "information", as defined earlier in the pipeline.
            val_info: validation windows "information", as defined earlier in the pipeline.
        """
        self.window_size_ = X_train.shape[1]
        if train_info is not None:
            self.pre_downsampling_window_size_ = train_info["window_size"][0]
            self.downsampling_size_ = train_info["downsampling_size"][0]
            self.downsampling_step_ = train_info["downsampling_step"][0]
            self.downsampling_func_ = train_info["downsampling_func"][0]
        elif self.pre_downsampling_window_size_ is None:
            raise ValueError(
                f"`train_info` should be passed with a "
                f'"window_size" key to fill {self.pre_downsampling_window_size_}'
            )
        log_windows_memory(X_train, X_val)
        start = timer()
        self._fit_window_model(X_train, y_train, X_val, y_val, train_info, val_info)
        end = timer()
        save_files(
            self.window_model_path,
            {"fit_metadata": {"num_seconds": end - start}},
            "json",
        )

    def _fit_window_model(
        self,
        X_train: np.array,
        y_train: Optional[np.array] = None,
        X_val: Optional[np.array] = None,
        y_val: Optional[np.array] = None,
        train_info: Optional[List[Tuple]] = None,
        val_info: Optional[List[Tuple]] = None,
    ) -> None:
        pass

    def predict_window_model(self, X: np.array) -> np.array:
        """Returns predictions for the provided windows.

        Args:
            X: input windows of shape `(n_windows, window_size, n_features)`.

        Returns:
            Window predictions, which can be anything used to assign anomaly scores/predictions.
        """
        self.check_components(["window_model"])
        return self._predict_window_model(X)

    def _predict_window_model(self, X: np.array) -> np.array:
        pass

    def set_window_scorer_params(self, **params) -> None:
        """Sets hyperparameters relevant to the window scorer."""
        pass

    def fit_window_scorer(
        self,
        X_train: np.array,
        y_train: Optional[np.array] = None,
        X_val: Optional[np.array] = None,
        y_val: Optional[np.array] = None,
        train_info: Optional[dict] = None,
        val_info: Optional[dict] = None,
    ) -> None:
        """Fits parameters of the window scorer to the provided windows.

        Args:
            X_train: training windows of shape `(n_train, window_size, n_features)`.
            y_train: training labels of shape `(n_train,)`, unused by unsupervised methods.
            X_val: validation windows of shape `(n_val, window_size, n_features)`.
            y_val: validation labels of shape `(n_val,)`, unused by unsupervised methods.
            train_info: training windows "information", as defined earlier in the pipeline.
            val_info: validation windows "information", as defined earlier in the pipeline.
        """
        self.check_components(["window_model"])
        self.window_size_ = X_train.shape[1]
        self.pre_downsampling_window_size_ = train_info["window_size"][0]
        self.downsampling_size_ = train_info["downsampling_size"][0]
        self.downsampling_step_ = train_info["downsampling_step"][0]
        self.downsampling_func_ = train_info["downsampling_func"][0]
        start = timer()
        self._fit_window_scorer(X_train, y_train, X_val, y_val, train_info, val_info)
        end = timer()
        save_files(
            self.window_scorer_path, {"fit_time": {"n_seconds": end - start}}, "json"
        )

    def _fit_window_scorer(
        self,
        X_train: np.array,
        y_train: Optional[np.array] = None,
        X_val: Optional[np.array] = None,
        y_val: Optional[np.array] = None,
        train_info: Optional[np.array] = None,
        val_info: Optional[np.array] = None,
    ) -> None:
        pass

    def predict_window_scorer(self, X: np.array) -> np.array:
        """Returns anomaly scores for the provided windows.

        Args:
            X: windows to score.

        Returns:
            Real-valued anomaly scores of shape `(n_windows,)`, where higher scores denote
              more outlyingness.
        """
        self.check_components(["window_model", "window_scorer"])
        return self._predict_window_scorer(X)

    def _predict_window_scorer(self, X: np.array) -> np.array:
        pass

    def set_online_scorer_params(self, scores_avg_beta: float = 0.9867):
        """Sets hyperparameters relevant to the online scorer."""
        self.scores_avg_beta_ = scores_avg_beta

    def fit_online_scorer(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        """Fits parameters of the online scorer to the provided sequences.

        Args:
            train: training sequences, each of shape `(seq_length, n_features)`.
            y_train: corresponding labels of shape `(seq_length,)` for each sequence,
              unused by unsupervised methods.
            val: validation sequences, each of shape `(seq_length, n_features)`.
            y_val: corresponding labels of shape `(seq_length,)` for each sequence,
              unused by unsupervised methods.
        """
        self.check_components(["window_model", "window_scorer"])
        start = timer()
        self._fit_online_scorer(train, y_train, val, y_val)
        end = timer()
        save_files(
            self.online_scorer_path, {"fit_time": {"n_seconds": end - start}}, "json"
        )

    def _fit_online_scorer(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        pass

    def predict_online_scorer(self, sequences: List[np.array]) -> List[np.array]:
        """Returns anomaly scores for the provided sequences.

        In this default implementation, sequence scores are assigned by calling
        the method's window scorer, and applying exponential smoothing on the
        resulting window scores. The first `window_size` records of every sequence are
        assigned a score of negative infinity (i.e., never deemed anomalous).

        Args:
            sequences: input sequences whose records to assign anomaly scores.

        Returns:
            Anomaly scores for each sequence record, where higher scores denote more outlyingness.
        """
        self.check_components(["window_model", "window_scorer", "online_scorer"])
        return self._predict_online_scorer(sequences)

    def _predict_online_scorer(self, sequences: List[np.array]) -> List[np.array]:
        """Default, window-based implementation."""
        check_is_not_none(self.window_size_, "self.window_size_")
        try:
            check_is_not_none(
                self.pre_downsampling_window_size_, "self.pre_downsampling_window_size_"
            )
            check_is_not_none(self.downsampling_size_, "self.downsampling_size_")
            check_is_not_none(self.downsampling_step_, "self.downsampling_step_")
            check_is_not_none(self.downsampling_func_, "self.downsampling_func_")
        except AttributeError:
            self.pre_downsampling_window_size_ = 1
            self.downsampling_size_ = 1
            self.downsampling_step_ = 1
            self.downsampling_func_ = 1
        neg_inf = np.core.getlimits.finfo(sequences[0][0].dtype).min
        sequences_scores = []
        for seq in sequences:
            X = get_sliding_windows(seq, self.pre_downsampling_window_size_, 1)
            if self.downsampling_size_ > 1:
                X = get_downsampled_windows(
                    X,
                    self.downsampling_size_,
                    self.downsampling_step_,
                    self.downsampling_func_,
                )
            seq_scores = self.predict_window_scorer(X)
            # apply exponentially weighted average of the sequence's record-wise outlier scores
            seq_scores = (
                pd.Series(seq_scores)
                .ewm(alpha=1 - self.scores_avg_beta_, adjust=True)
                .mean()
                .values
            )
            scores_padding = np.full(self.pre_downsampling_window_size_ - 1, neg_inf)
            seq_scores = np.nan_to_num(np.concatenate([scores_padding, seq_scores]))
            sequences_scores.append(seq_scores)
        return sequences_scores

    def set_online_detector_params(
        self, scorer_evaluator_cfg: DictConfig, maximized_f_score: str
    ) -> None:
        """Sets hyperparameters relevant to the online detector."""
        check_value_in_choices(
            maximized_f_score, "maximized_f_score", ["mixed", "balanced"]
        )
        evaluator_type = scorer_evaluator_cfg.pop("evaluation_type")
        evaluator_class = (
            PointEvaluator if evaluator_type == "point" else RangeEvaluator
        )
        self.online_scorer_evaluator_ = evaluator_class(**scorer_evaluator_cfg)
        self.maximized_f_score_ = maximized_f_score

    def fit_online_detector(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        """Fits parameters of the online detector to the provided sequences.

        Args:
            train: training sequences, each of shape `(seq_length, n_features)`.
            y_train: corresponding labels of shape `(seq_length,)` for each sequence,
              unused by unsupervised methods.
            val: validation sequences, each of shape `(seq_length, n_features)`.
            y_val: corresponding labels of shape `(seq_length,)` for each sequence,
              unused by unsupervised methods.
        """
        self.check_components(["window_model", "window_scorer", "online_scorer"])
        start = timer()
        self._fit_online_detector(train, y_train, val, y_val)
        end = timer()
        save_files(
            self.online_detector_path, {"fit_time": {"n_seconds": end - start}}, "json"
        )

    def _fit_online_detector(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        """Default behavior: supervised threshold selection.

        For supervised threshold selection, `train` corresponds to the pipeline-level validation
          sequences, and `val` should not be provided.

        For other techniques (overloading this method), `train` and `val` correspond to the
          pipeline-level training and validation sequences.
        """
        if not (val is None and y_val is None):
            raise ValueError(
                "No validation sequences should be passed to the default `_fit_online_detector()`."
            )
        scores = self.predict_online_scorer(train)
        self.score_threshold_ = get_best_score_threshold(
            scoring_evaluator=self.online_scorer_evaluator_,
            sequences_scores=scores,
            sequences_labels=y_train,
            f_score_type=self.maximized_f_score_,
        )

    def predict_online_detector(self, sequences: List[np.array]) -> List[np.array]:
        """Returns (binary) anomaly predictions for the provided sequences.

        Args:
            sequences: input sequences whose records to assign anomaly predictions.

        Returns:
            Binary anomaly predictions for each sequence record: 0 for normal, 1 for anomaly.
        """
        self.check_components(
            ["window_model", "window_scorer", "online_scorer", "online_detector"]
        )
        return self._predict_online_detector(sequences)

    def _predict_online_detector(self, sequences: List[np.array]) -> List[np.array]:
        """A record is an anomaly iff its score is greater or equal to `self.score_threshold_`."""
        sequences_scores = self.predict_online_scorer(sequences)
        return [(s >= self.score_threshold_).astype(np.int8) for s in sequences_scores]

    def __getstate__(self):
        """Called when pickling/saving the detector object (override for custom behaviors)."""
        return self.__dict__

    def __setstate__(self, d):
        """Called when unpickling/loading the detector object (override for custom behaviors)."""
        self.__dict__ = d
