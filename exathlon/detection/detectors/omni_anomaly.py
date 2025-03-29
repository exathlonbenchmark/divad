import os
import time
import random
import logging
import argparse
from pprint import pprint

import numpy as np
import tensorflow as tf

from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict

from utils.guarding import check_value_in_choices
from detection.detectors.helpers.general import (
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.helpers.tf_omni_anomaly import TensorFlowOmniAnomaly
from detection.detectors.helpers.tf_omni_anomaly_helpers.training import Trainer
from detection.detectors.helpers.tf_omni_anomaly_helpers.prediction import Predictor
from detection.detectors.base import BaseDetector

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class OmniAnomaly(BaseDetector):
    relevant_steps = [
        "make_window_datasets",
        "train_window_model",
        "train_online_scorer",
        "evaluate_online_scorer",
        "train_online_detector",
        "evaluate_online_detector",
    ]
    fitting_steps = ["train_window_model", "train_online_detector"]
    window_model_param_names = [
        "model_",
        "trainer_",
        "predictor_",
        "saver_",
        "arch_hps_",
        "train_hps_",
        "pred_hps_",
    ]
    window_scorer_param_names = []
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_ = None
        self.trainer_ = None
        self.predictor_ = None
        self.saver_ = None
        self.arch_hps_ = None
        self.train_hps_ = None
        self.pred_hps_ = None

    def set_window_model_params(
        self,
        z_dim: int = 3,
        dense_dim: int = 500,
        rnn_num_hidden: int = 500,
        use_connected_z_p: bool = True,
        use_connected_z_q: bool = True,
        std_epsilon: float = 1e-4,
        posterior_flow_type: str = "nf",
        nf_layers: int = 20,
        initial_lr: float = 0.001,
        lr_anneal_epoch_freq: int = 40,
        lr_anneal_factor: float = 0.5,
        gradient_clip_norm: float = 10.0,
        valid_step_freq: int = 100,
        max_epoch: int = 20,
        batch_size: int = 50,
        test_batch_size: int = 50,
        test_n_z: int = 1,
        get_score_on_dim: bool = False,
    ):
        """Sets hyperparameters relevant to the window model.

        Args:
            z_dim: dimension of the latent vector representation (encoding).
            dense_dim: number of dense units.
            rnn_num_hidden: number of GRU units.
            use_connected_z_p: whether to link z_p's temporally.
            use_connected_z_q: whether to link z_q's temporally.
            std_epsilon: epsilon for VAE standard deviations.
            posterior_flow_type: type of posterior flow (either "nf" or "none").
            nf_layers: number of planar normalizing flow layers.
            initial_lr: learning rate for the Adam optimizer.
            lr_anneal_epoch_freq: period of learning rate annealing in epochs.
            lr_anneal_factor: learning rate annealing factor.
            gradient_clip_norm: norm value to which to clip the gradients.
            valid_step_freq: period of validation metrics computation in steps, if negative, validation
              metrics will be computed -`valid_step_freq` per epoch (e.g., -5 <=> validate 5 times per epoch).
            max_epoch: maximum number of epochs to train the model for.
            batch_size: training mini-batch size.
            test_batch_size: test (and validation) mini-batch size.
            test_n_z: number of test z values to sample when computing reconstruction probabilities.
            get_score_on_dim: whether to get score on dim. If `True`, the score will be a 2-dim ndarray.
        """
        check_value_in_choices(
            posterior_flow_type, "posterior_flow_type", ["nf", "none"]
        )
        self.arch_hps_ = {
            "z_dim": z_dim,
            "dense_dim": dense_dim,
            "rnn_num_hidden": rnn_num_hidden,
            "use_connected_z_p": use_connected_z_p,
            "use_connected_z_q": use_connected_z_q,
            "std_epsilon": std_epsilon,
            "posterior_flow_type": posterior_flow_type,
            "nf_layers": nf_layers,
            "get_score_on_dim": get_score_on_dim,
        }
        self.train_hps_ = {
            "initial_lr": initial_lr,
            "lr_anneal_epochs": lr_anneal_epoch_freq,
            "lr_anneal_factor": lr_anneal_factor,
            "grad_clip_norm": gradient_clip_norm,
            "valid_step_freq": valid_step_freq,
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "valid_batch_size": test_batch_size,
        }
        self.pred_hps_ = {"test_n_z": test_n_z}

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
        n_train_samples, window_size, n_features = X_train.shape
        if self.train_hps_["valid_step_freq"] < 0:
            # because we drop the last batch in training
            steps_per_epoch = n_train_samples // self.train_hps_["batch_size"]
            logging.info(
                f"Validating {-self.train_hps_['valid_step_freq']} times per epoch."
            )
            logging.info(f"n_train_samples: {n_train_samples}")
            logging.info(f"batch_size: {self.train_hps_['batch_size']}")
            logging.info(f"steps_per_epoch: {steps_per_epoch}")
            self.train_hps_["valid_step_freq"] = steps_per_epoch // (
                -self.train_hps_["valid_step_freq"]
            )
            logging.info(f"valid_step_freq: {self.train_hps_['valid_step_freq']}")
        for k, v in zip(["window_length", "x_dim"], [window_size, n_features]):
            self.arch_hps_[k] = v

        # construct the model under `variable_scope` named 'model'
        with tf.variable_scope("model") as model_vs:
            self.model_ = TensorFlowOmniAnomaly(
                config=argparse.Namespace(**self.arch_hps_), name="model"
            )

            # construct the trainer
            self.trainer_ = Trainer(
                model=self.model_, model_vs=model_vs, **self.train_hps_
            )

            # construct the predictor
            self.predictor_ = Predictor(
                self.model_,
                batch_size=self.train_hps_["valid_batch_size"],
                n_z=self.pred_hps_["test_n_z"],
                last_point_only=True,
            )

            with tf.Session().as_default():
                # train the model
                train_start = time.time()
                best_valid_metrics = self.trainer_.fit(
                    X_train, X_val, summary_dir=self.window_model_path
                )
                train_time = (time.time() - train_start) / self.train_hps_["max_epoch"]
                best_valid_metrics.update({"train_time": train_time})
                if self.window_model_path is not None:
                    # save the variables
                    var_dict = get_variables_as_dict(model_vs)
                    self.saver_ = VariableSaver(var_dict, self.window_model_path)
                    self.saver_.save()
                print("=" * 30 + "result" + "=" * 30)
                pprint(best_valid_metrics)

    def _predict_window_model(self, X):
        pass

    def _predict_window_scorer(self, X):
        with tf.variable_scope("model") as model_vs:
            with tf.Session().as_default():
                self.saver_.restore()
                # get score of test set
                scores, test_z, pred_speed = self.predictor_.get_window_scores(X)
                if self.arch_hps_["get_score_on_dim"]:
                    # get the joint score
                    scores = np.sum(scores, axis=-1)
        # OmniAnomaly defines scores as reconstruction probabilities
        return -scores

    def __getstate__(self):
        # saving TF objects causes errors
        removed = ["model_", "trainer_", "predictor_", "saver_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        # Fixes `ValueError: Variable model/trainer/global_step already exists, disallowed`
        # https://www.programmersought.com/article/380863523/.
        tf.reset_default_graph()
        with tf.variable_scope("model") as model_vs:
            self.model_ = TensorFlowOmniAnomaly(
                config=argparse.Namespace(**self.arch_hps_), name="model"
            )
            # creating trainer and predictor is needed to initialize the variables
            # https://github.com/NetManAIOps/donut.
            self.trainer_ = Trainer(
                model=self.model_, model_vs=model_vs, **self.train_hps_
            )
            self.predictor_ = Predictor(
                self.model_,
                batch_size=self.train_hps_["valid_batch_size"],
                n_z=self.pred_hps_["test_n_z"],
                last_point_only=True,
            )
            with tf.Session().as_default():
                try:
                    self.saver_ = VariableSaver(
                        get_variables_as_dict(model_vs), self.window_model_path
                    )
                except OSError:  # works both if not found and permission denied
                    # if not found, expect the model to be next to the detector file
                    backup_window_model_path = os.path.abspath(os.curdir)
                    self.saver_ = VariableSaver(
                        get_variables_as_dict(model_vs), backup_window_model_path
                    )
                    self.window_model_path = backup_window_model_path


def _print_variables(sess):
    """Used for debugging.

    From: https://github.com/google/prettytensor/issues/6#issuecomment-380919368.
    """
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)  # Prints the name of the variable alongside its value.
