import os

import numpy as np
import pandas as pd
import keras_tuner
from keras_tuner import HyperModel
from xgboost import XGBClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import (
    make_scorer,
    fbeta_score,
    precision_recall_fscore_support,
    recall_score,
)

import matplotlib.pyplot as plt

from utils.guarding import (
    check_is_not_none,
    check_is_percentage,
    check_value_in_choices,
)
from detection.detectors.helpers.general import get_flattened_windows
from detection.detectors.base import BaseDetector


def get_evaluation_col_to_values(model, X_flat, y_binary, y, set_name):
    col_to_values = dict()
    upper_set_name = set_name.upper()
    preds = model.predict(X_flat)
    p, r, f1 = precision_recall_fscore_support(
        y_binary, preds, average="binary", zero_division=0
    )[:3]
    for k, v in zip(["F1", "P", "R"], [f1, p, r]):
        col_to_values[f"{upper_set_name}_{k}"] = [v]
    ano_classes = np.array([c for c in np.unique(y) if c != 0.0], dtype=np.int32)
    for c in ano_classes:
        y_true_class = np.array(y == c, dtype=np.int32)
        class_r = recall_score(y_true_class, preds, average="binary", zero_division=0)
        col_to_values[f"{upper_set_name}_R_T{c}"] = class_r
    return col_to_values


def save_evaluation(
    model,
    X_train_flat,
    y_train_binary,
    y_train,
    X_val_flat,
    y_val_binary,
    y_val,
    output_path,
    curves=True,
):
    os.makedirs(output_path, exist_ok=True)
    train_col_to_values = get_evaluation_col_to_values(
        model, X_train_flat, y_train_binary, y_train, set_name="train"
    )
    if X_val_flat is not None:
        val_col_to_values = get_evaluation_col_to_values(
            model,
            X_val_flat,
            y_val_binary,
            y_val,
            set_name="val",
        )
        if curves:
            curves_data = model.evals_result()
            # plot learning curves
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(curves_data["validation_0"]["logloss"], label="train")
            ax.plot(curves_data["validation_1"]["logloss"], label="val")
            ax.legend()
            ax.grid(True)
            fig.savefig(os.path.join(output_path, "curves.png"))
            plt.close(fig)
    else:
        val_col_to_values = dict()
    evaluation_df = pd.DataFrame(dict(train_col_to_values, **val_col_to_values))
    evaluation_df.to_csv(os.path.join(output_path, "results.csv"), index=False)


class Xgboost(BaseDetector, HyperModel):
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
        "xgb_classifier",
        "binary_mode",
        "imbalance_handling",
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "subsample",
        "learning_rate",
        "gamma",
        "max_delta_step",
        "random_state",
    ]
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.xgb_classifier = None
        self.binary_mode = None
        self.imbalance_handling = None
        self.n_estimators = None
        self.max_depth = None
        self.min_child_weight = None
        self.subsample = None
        self.learning_rate = None
        self.gamma = None
        self.max_delta_step = None
        self.random_state = None
        # used for balancing if relevant
        self.n_train_normal = None
        self.n_train_ano = None

    def set_window_scorer_params(
        self,
        binary_mode: bool = True,
        imbalance_handling: str = "none",
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        gamma: float = 0.0,
        max_delta_step: int = 0,
        random_state: int = 0,
    ) -> None:
        """Sets hyperparameters relevant to the window scorer.

        Args:
            binary_mode: only binary mode is supported for now.
            imbalance_handling: either "none" for no handling, or "weights" for handling data imbalance
             through setting `scale_pos_weight := num_negative / num_positive`.
            n_estimators: see XGBoost docs: https://xgboost.readthedocs.io/en/stable/parameter.html.
            max_depth: see XGBoost docs.
            min_child_weight: see XGBoost docs.
            subsample: see XGBoost docs.
            learning_rate: see XGBoost docs.
            gamma: see XGBoost docs.
            max_delta_step: maximum delta step we allow each leaf output to be.
             If the value is set to 0, it means there is no constraint.
             If it is set to a positive value, it can help making the update step more conservative.
             Usually this parameter is not needed, but it might help in logistic regression when class
             is extremely imbalanced. Set it to value of 1-10 might help control the update.
            random_state: seed parameter of XGBoost.
        """
        check_value_in_choices(
            imbalance_handling, "imbalance_handling", ["none", "weights"]
        )
        check_is_percentage(subsample, "subsample")
        check_is_percentage(learning_rate, "learning_rate")
        self.binary_mode = binary_mode
        self.imbalance_handling = imbalance_handling
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.random_state = random_state

    def _fit_window_scorer(self, X_train, y_train=None, X_val=None, y_val=None):
        check_is_not_none(y_train, "y_train")
        if not self.binary_mode:
            raise NotImplementedError("Only binary mode is currently supported.")
        self.n_train_normal = np.sum(y_train == 0.0)
        self.n_train_ano = np.sum(y_train > 0.0)
        y_train_binary = np.copy(y_train)
        y_train_binary[y_train_binary > 0.0] = 1.0
        objective = "binary:logistic"
        # num_class should not be specified in binary mode
        num_class_kwarg = dict()
        if self.imbalance_handling == "weights":
            scale_pos_weight = self.n_train_normal / self.n_train_ano
        else:
            scale_pos_weight = 1.0
        self.xgb_classifier = XGBClassifier(
            booster="gbtree",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            max_delta_step=self.max_delta_step,
            objective=objective,
            scale_pos_weight=scale_pos_weight,
            seed=self.random_state,
            **num_class_kwarg,
        )
        X_train_flat = get_flattened_windows(X_train)
        if X_val is not None:
            check_is_not_none(y_val, "y_val")
            X_val_flat = get_flattened_windows(X_val)
            y_val_binary = np.copy(y_val)
            y_val_binary[y_val_binary > 0.0] = 1.0
            eval_set_kwarg = {
                "eval_set": [(X_train_flat, y_train_binary), (X_val_flat, y_val_binary)]
            }
        else:
            X_val_flat = None
            y_val_binary = None
            eval_set_kwarg = dict()
        self.xgb_classifier.fit(X_train_flat, y_train_binary, **eval_set_kwarg)
        # performance evaluation
        save_evaluation(
            self.xgb_classifier,
            X_train_flat,
            y_train_binary,
            y_train,
            X_val_flat,
            y_val_binary,
            y_val,
            self.window_scorer_path,
            curves=True,
        )

    def build(self, hp):
        config = {
            "n_estimators": hp.Choice("n_estimators", [50, 100, 200]),
            "max_depth": hp.Int("max_depth", min_value=1, max_value=12),
            "min_child_weight": hp.Choice(
                "min_child_weight", [1, 3, 5, 10, 20, 100, 500]
            ),
            "subsample": hp.Float("subsample", min_value=0.5, max_value=1.0),
            "learning_rate": hp.Float(
                "learning_rate", min_value=1e-4, max_value=0.5, sampling="log"
            ),
            "gamma": self.gamma,
            "max_delta_step": hp.Choice("max_delta_step", [0, 1, 5]),
            "seed": self.random_state,
        }
        imbalance_handling = hp.Choice("imbalance_handling", ["none", "weights"])
        if imbalance_handling == "weights":
            config["scale_pos_weight"] = self.n_train_normal / self.n_train_ano
        else:
            config["scale_pos_weight"] = 1.0
        self.xgb_classifier = XGBClassifier(
            booster="gbtree", objective="binary:logistic", **config
        )
        return self.xgb_classifier

    def tune_window_scorer(self, X_train, y_train=None, X_val=None, y_val=None):
        check_is_not_none(X_val, "X_val")
        check_is_not_none(y_val, "y_val")
        self.n_train_normal = np.sum(y_train == 0.0)
        self.n_train_ano = np.sum(y_train > 0.0)
        X_train_flat = get_flattened_windows(X_train)
        y_train_binary = np.copy(y_train)
        y_train_binary[y_train_binary > 0.0] = 1.0
        X_val_flat = get_flattened_windows(X_val)
        y_val_binary = np.copy(y_val)
        y_val_binary[y_val_binary > 0.0] = 1.0
        X = np.concatenate([X_train_flat, X_val_flat], axis=0)
        y = np.concatenate([y_train_binary, y_val_binary], axis=0)
        val_fold = np.concatenate(
            [np.repeat(-1, X_train_flat.shape[0]), np.repeat(0, X_val_flat.shape[0])]
        )
        search_output_path = f"{self.window_scorer_path}_searches"
        os.makedirs(search_output_path, exist_ok=True)
        f1_scorer = make_scorer(fbeta_score, beta=1.0)
        max_trials = 2000
        oracle = keras_tuner.oracles.BayesianOptimizationOracle(
            objective=keras_tuner.Objective("score", "max"), max_trials=max_trials
        )
        tuner_id = f"bayesian_{max_trials}"
        tuner = keras_tuner.tuners.SklearnTuner(
            oracle=oracle,
            hypermodel=self,
            scoring=f1_scorer,
            cv=PredefinedSplit(val_fold),
            directory=search_output_path,
            project_name=tuner_id,
            overwrite=True,
        )
        tuner.search(X, y)
        # performance summary
        hp_col_to_values = dict()
        train_col_to_values = dict()
        val_col_to_values = dict()
        n_models = max_trials
        best_models = tuner.get_best_models(num_models=n_models)
        best_hps = tuner.get_best_hyperparameters(num_trials=n_models)
        for model, hps in zip(best_models, best_hps):
            model_hps = hps.get_config()["values"]
            model_train_col_to_values = get_evaluation_col_to_values(
                model, X_train_flat, y_train_binary, y_train, set_name="train"
            )
            model_val_col_to_values = get_evaluation_col_to_values(
                model,
                X_val_flat,
                y_val_binary,
                y_val,
                set_name="val",
            )
            if len(train_col_to_values) == 0:
                hp_col_to_values = {k: [v] for k, v in model_hps.items()}
                train_col_to_values = model_train_col_to_values
                val_col_to_values = model_val_col_to_values
            else:
                for k, v in model_hps.items():
                    hp_col_to_values[k].append(v)
                for k, v in model_train_col_to_values.items():
                    train_col_to_values[k] += v
                for k, v in model_val_col_to_values.items():
                    val_col_to_values[k] += v
        evaluation_df = pd.DataFrame(
            dict(hp_col_to_values, **train_col_to_values, **val_col_to_values)
        )
        evaluation_df.to_csv(
            os.path.join(search_output_path, f"{tuner_id}_results.csv"), index=False
        )

    def _predict_window_scorer(self, X):
        # (1 - p(normal))
        return 1.0 - self.xgb_classifier.predict_proba(get_flattened_windows(X))[:, 0]
