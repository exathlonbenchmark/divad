import os
import logging
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from utils.guarding import check_value_in_choices
from data.helpers import save_files
from detection.detectors.helpers.general import (
    get_parsed_integer_list_str,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.helpers.tf_helpers import (
    get_callbacks,
    sample_squared_euclidean_distance,
)
from detection.detectors.helpers.tf_deep_svdd import (
    TensorFlowDeepSVDD,
    compile_deep_svdd,
    get_deep_svdd_dataset,
)
from detection.detectors.base import BaseDetector


class DeepSvdd(BaseDetector):
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
        # deep SVDD model, centroid and training callbacks
        "deep_svdd_",
        "c_",  # hypersphere centroid
        "c_epsilon_",
        "callbacks_",
        # architecture, optimization, training and callbacks hyperparameters
        "arch_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        # needed to load the model (in addition to window size)
        "n_features_",
        # "needed" to properly re-set callbacks after loading
        "n_train_samples_",
    ]
    window_scorer_param_names = []
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "deep_svdd"
        self.deep_svdd_ = None
        self.c_ = None
        self.c_epsilon_ = None
        self.callbacks_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.n_features_ = None
        self.n_train_samples_ = None

    def set_window_model_params(
        self,
        type_: str = "dense",
        dense_hidden_activations: str = "relu",
        rec_unit_type: str = "lstm",
        rec_dropout: float = 0.0,
        conv1d_strides: Union[int, str] = "",
        n_hidden_neurons: Union[int, str] = "",
        output_dim: int = 10,
        output_activation: str = "linear",
        batch_normalization: bool = False,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        loss: str = "one_class",
        nu: float = 0.01,
        optimizer: str = "adam",
        adamw_weight_decay: float = 0.0,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        shuffling_buffer_prop: float = 1.0,
        n_epochs: int = 400,
        lr_scheduling: str = "none",
        lrs_pwc_red_factor: float = 2,
        lrs_pwc_red_freq: int = 10,
        lrs_oc_start_lr: float = 1e-4,
        lrs_oc_max_lr: float = 1e-3,
        lrs_oc_min_mom: float = 0.85,
        lrs_oc_max_mom: float = 0.95,
        early_stopping_target: str = "val_loss",
        early_stopping_patience: int = 20,
        c_epsilon: float = 0.01,
        random_seed: Optional[int] = None,
    ):
        """Sets hyperparameters relevant to the window model.

        Args:
            type_: type of deep SVDD to build (either "dense" or "rec").
            dense_hidden_activations: hidden layers activation for dense architectures.
            rec_unit_type: type of recurrent unit (either "lstm" or "gru") for recurrent architecture.
            rec_dropout: dropout rate for the hidden state of recurrent layers.
            conv1d_strides: strides for each Conv1D layer before the hidden layers, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            n_hidden_neurons: number of units for each hidden layer before the output, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            output_dim: number of units for the output layer.
            output_activation: output layer activation.
            batch_normalization: whether to apply batch normalization (before layer activations).
            input_dropout: dropout rate for the input layer.
            hidden_dropout: dropout rate for other feed-forward layers (except the output).
            loss: loss function to optimize (either "one_class" or "soft_boundary").
            nu: nu parameter for the "soft_boundary" loss.
            optimizer: optimization algorithm used for training the network.
            adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
            learning_rate: learning rate used by the optimization algorithm.
            batch_size: mini-batch size.
            shuffling_buffer_prop: proportion of training data to use as a shuffling buffer.
            n_epochs: number of epochs to train the model for.
            lr_scheduling: learning rate scheduling to apply (either "none", "pw_constant" or "one_cycle").
            lrs_pwc_red_factor: "pw_constant" learning rate reduction factor.
            lrs_pwc_red_freq: "pw_constant" learning rate reduction frequency (in epochs).
            lrs_oc_start_lr: "one_cycle" starting learning rate.
            lrs_oc_max_lr: "one_cycle" maximum learning rate.
            lrs_oc_min_mom: "one_cycle" minimum momentum.
            lrs_oc_max_mom: "one_cycle" maximum momentum.
            early_stopping_target: early stopping target (either "loss" or "val_loss").
            early_stopping_patience: early stopping patience (in epochs).
            c_epsilon: minimum absolute value for a coordinate of the hypersphere centroid.
            random_seed: random seed for reproducibility (e.g., of the hypersphere
              centroid) across calls.
        """
        check_value_in_choices(type_, "type_", ["dense", "rec"])
        if type_ == "rec":
            check_value_in_choices(rec_unit_type, "rec_unit_type", ["lstm", "gru"])
        check_value_in_choices(
            lr_scheduling, "lr_scheduling", ["none", "pw_constant", "one_cycle"]
        )
        check_value_in_choices(
            early_stopping_target, "early_stopping_target", ["loss", "val_loss"]
        )
        check_value_in_choices(loss, "loss", ["one_class", "soft_boundary"])
        if loss == "soft_boundary":
            raise NotImplementedError("Soft boundary deep SVDD loss not supported yet.")

        # turn list parameters to actual lists
        if isinstance(conv1d_strides, int):
            conv1d_strides = str(conv1d_strides)
        if isinstance(n_hidden_neurons, int):
            n_hidden_neurons = str(n_hidden_neurons)
        conv1d_strides = get_parsed_integer_list_str(conv1d_strides)
        n_hidden_neurons = get_parsed_integer_list_str(n_hidden_neurons)
        self.c_epsilon_ = c_epsilon
        self.arch_hps_ = {
            "type_": type_,
            "conv1d_strides": conv1d_strides,
            "n_hidden_neurons": n_hidden_neurons,
            "output_dim": output_dim,
            "output_activation": output_activation,
            "batch_normalization": batch_normalization,
            "input_dropout": input_dropout,
            "hidden_dropout": hidden_dropout,
            "random_seed": random_seed,
            "loss_": loss,
        }
        if type_ == "dense":
            self.arch_hps_["dense_hidden_activations"] = dense_hidden_activations
        elif type_ == "rec":
            self.arch_hps_["rec_unit_type"] = rec_unit_type
            self.arch_hps_["rec_dropout"] = rec_dropout
        if loss == "soft_boundary":
            self.arch_hps_["nu"] = nu
        self.opt_hps_ = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
        }
        if optimizer == "adamw":
            self.opt_hps_["adamw_weight_decay"] = adamw_weight_decay
        self.train_hps_ = {
            "batch_size": batch_size,
            "shuffling_buffer_prop": shuffling_buffer_prop,
            "n_epochs": n_epochs,
        }
        # training callback hyperparameters
        self.callbacks_hps_ = {
            "lr_scheduling": lr_scheduling,
            "early_stopping_target": early_stopping_target,
            "early_stopping_patience": early_stopping_patience,
        }
        if lr_scheduling == "pw_constant":
            self.callbacks_hps_["lrs_pwc_red_factor"] = lrs_pwc_red_factor
            self.callbacks_hps_["lrs_pwc_red_freq"] = lrs_pwc_red_freq
        elif lr_scheduling == "one_cycle":
            self.callbacks_hps_["lrs_oc_start_lr"] = lrs_oc_start_lr
            self.callbacks_hps_["lrs_oc_max_lr"] = lrs_oc_max_lr
            self.callbacks_hps_["lrs_oc_min_mom"] = lrs_oc_min_mom
            self.callbacks_hps_["lrs_oc_max_mom"] = lrs_oc_max_mom

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
        if self.arch_hps_["random_seed"] is not None:
            tf.random.set_seed(self.arch_hps_["random_seed"])
        n_train_samples, window_size, n_features = X_train.shape
        self.n_train_samples_ = n_train_samples
        self.n_features_ = n_features
        self.deep_svdd_ = TensorFlowDeepSVDD(window_size, n_features, **self.arch_hps_)
        # set centroid as the mean of outputs for an initial forward pass
        self.c_ = tf.reduce_mean(self.deep_svdd_.predict(X_train), axis=0).numpy()
        # prevent centroid coordinates from being too small
        small_abs_c_mask = np.abs(self.c_) < self.c_epsilon_
        self.c_[small_abs_c_mask & (self.c_ < 0)] = -self.c_epsilon_
        self.c_[small_abs_c_mask & (self.c_ > 0)] = self.c_epsilon_
        self.deep_svdd_.set_centroid(self.c_)
        self.callbacks_ = get_callbacks(
            callbacks_type="train",
            output_path=self.window_model_path,
            model_file_name=self.model_file_name,
            save_weights_only=True,
            n_train_samples=n_train_samples,
            batch_size=self.train_hps_["batch_size"],
            n_epochs=self.train_hps_["n_epochs"],
            **self.callbacks_hps_,
        )
        compile_deep_svdd(self.deep_svdd_, **self.opt_hps_)
        train_dataset = get_deep_svdd_dataset(
            X_train,
            shuffling_buffer_prop=self.train_hps_["shuffling_buffer_prop"],
            batch_size=self.train_hps_["batch_size"],
        )
        val_dataset = None if X_val is None else get_deep_svdd_dataset(X_val)
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        self.deep_svdd_.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.train_hps_["n_epochs"],
            verbose=1,
            callbacks=self.callbacks_,
        )

    def _predict_window_model(self, X):
        return self.deep_svdd_.predict(X)

    def _predict_window_scorer(self, X):
        """Anomaly scores are squared distances from the centroid in output space."""
        encoded = self.predict_window_model(X)
        return sample_squared_euclidean_distance(encoded, self.c_)

    def __getstate__(self):
        # saving callbacks_ causes errors
        removed = ["deep_svdd_", "callbacks_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        self.deep_svdd_ = TensorFlowDeepSVDD(
            self.window_size_, self.n_features_, **self.arch_hps_
        )
        self.deep_svdd_.set_centroid(self.c_)
        self.callbacks_ = get_callbacks(
            callbacks_type="train",
            output_path=self.window_model_path,
            model_file_name=self.model_file_name,
            save_weights_only=True,
            n_train_samples=self.n_train_samples_,
            batch_size=self.train_hps_["batch_size"],
            n_epochs=self.train_hps_["n_epochs"],
            **self.callbacks_hps_,
        )
        compile_deep_svdd(self.deep_svdd_, **self.opt_hps_)
        try:
            self.deep_svdd_.load_weights(
                os.path.join(self.window_model_path, self.model_file_name)
            )
        except OSError:  # works both if not found and permission denied
            # if not found, expect the keras model to be next to the detector file
            self.deep_svdd_.load_weights(os.path.join(os.curdir, self.model_file_name))
            self.window_model_path = os.curdir
