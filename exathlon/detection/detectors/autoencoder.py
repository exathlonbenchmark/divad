import os
import logging
from typing import Union

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import mean_squared_error

from utils.guarding import check_value_in_choices
from data.helpers import save_files
from detection.detectors.helpers.general import (
    get_parsed_integer_list_str,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.helpers.tf_helpers import get_callbacks, LayerBlock
from detection.detectors.helpers.tf_autoencoder import (
    get_autoencoder,
    compile_autoencoder,
    get_autoencoder_dataset,
)
from detection.detectors.base import BaseDetector


class Autoencoder(BaseDetector):
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
        # autoencoder model and training callbacks
        "autoencoder_",
        "callbacks_",
        # architecture, optimization, training and callbacks hyperparameters
        "arch_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        # "needed" to properly re-set callbacks after loading
        "n_train_samples_",
    ]
    window_scorer_param_names = []
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "autoencoder.h5"
        self.autoencoder_ = None
        self.callbacks_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.n_train_samples_ = None

    def set_window_model_params(
        self,
        latent_dim: int = 10,
        type_: str = "dense",
        enc_conv1d_filters: Union[int, str] = "",
        enc_conv1d_kernel_sizes: Union[int, str] = "",
        enc_conv1d_strides: Union[int, str] = "",
        conv1d_pooling: bool = False,
        conv1d_batch_norm: bool = False,
        enc_n_hidden_neurons: Union[int, str] = "",
        dense_layers_activation: str = "relu",
        linear_latent_activation: bool = False,
        dec_last_activation: str = "linear",
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rec_unit_type: str = "lstm",
        activation_rec: str = "tanh",
        rec_dropout: float = 0.0,
        rec_latent_type: str = "rec",
        conv_add_dense_for_latent: bool = False,
        loss: str = "mse",
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
    ):
        """Sets hyperparameters relevant to the window model.

        Args:
            latent_dim: dimension of the latent vector representation (coding).
            type_: type of autoencoder to build.
            enc_conv1d_filters: number of filters for each Conv1D layer before the hidden layers, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            enc_conv1d_kernel_sizes: kernel sizes for each Conv1D layer before the hidden layers, in the
             same format as `enc_conv1d_kernel_sizes`.
            enc_conv1d_strides: strides for each Conv1D layer before the hidden layers, in the
             same format as `enc_conv1d_kernel_sizes`.
            conv1d_pooling: whether to perform downsampling and upsampling through pooling and upsampling
             layers rather than strided convolutions (the last decoder layer will always use strided convolutions).
            conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
            enc_n_hidden_neurons: number of units for each hidden layer before the coding, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            dense_layers_activation: intermediate layers activation for dense architectures.
            linear_latent_activation: whether to consider a linear activation for the latent layer.
            dec_last_activation: last decoder layer activation (either "linear" or "sigmoid").
            input_dropout: dropout rate for the input layer.
            hidden_dropout: dropout rate for other feed-forward layers (except the output).
            rec_unit_type: type of recurrent unit (either "lstm" or "gru").
            activation_rec: activation function to use for recurrent layers (not the "recurrent activation").
            rec_dropout: dropout rate for the hidden state of recurrent layers.
            rec_latent_type: type of latent layers for recurrent architectures.
            conv_add_dense_for_latent: whether to add a dense layer to output the coding for fully
             convolutional architectures.
            loss: loss function to optimize (either "mse" or "bce").
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
        """
        check_value_in_choices(type_, "type_", ["dense", "rec"])
        check_value_in_choices(
            lr_scheduling, "lr_scheduling", ["none", "pw_constant", "one_cycle"]
        )
        check_value_in_choices(
            early_stopping_target, "early_stopping_target", ["loss", "val_loss"]
        )
        if type_ == "rec":
            check_value_in_choices(rec_unit_type, "rec_unit_type", ["lstm", "gru"])
        # turn list parameters to actual lists
        if isinstance(enc_conv1d_filters, int):
            enc_conv1d_filters = str(enc_conv1d_filters)
        if isinstance(enc_conv1d_kernel_sizes, int):
            enc_conv1d_kernel_sizes = str(enc_conv1d_kernel_sizes)
        if isinstance(enc_conv1d_strides, int):
            enc_conv1d_strides = str(enc_conv1d_strides)
        if isinstance(enc_n_hidden_neurons, int):
            enc_n_hidden_neurons = str(enc_n_hidden_neurons)
        enc_conv1d_filters = get_parsed_integer_list_str(enc_conv1d_filters)
        enc_conv1d_kernel_sizes = get_parsed_integer_list_str(enc_conv1d_kernel_sizes)
        enc_conv1d_strides = get_parsed_integer_list_str(enc_conv1d_strides)
        enc_n_hidden_neurons = get_parsed_integer_list_str(enc_n_hidden_neurons)
        self.arch_hps_ = {
            "latent_dim": latent_dim,
            "type_": type_,
            "enc_conv1d_filters": enc_conv1d_filters,
            "enc_conv1d_kernel_sizes": enc_conv1d_kernel_sizes,
            "enc_conv1d_strides": enc_conv1d_strides,
            "conv1d_pooling": conv1d_pooling,
            "conv1d_batch_norm": conv1d_batch_norm,
            "enc_n_hidden_neurons": enc_n_hidden_neurons,
            "dec_last_activation": dec_last_activation,
            "input_dropout": input_dropout,
            "hidden_dropout": hidden_dropout,
            "dense_layers_activation": dense_layers_activation,
            "linear_latent_activation": linear_latent_activation,
        }
        if type_ == "rec":
            self.arch_hps_["rec_unit_type"] = rec_unit_type
            self.arch_hps_["activation_rec"] = activation_rec
            self.arch_hps_["rec_dropout"] = rec_dropout
            self.arch_hps_["rec_latent_type"] = rec_latent_type
            self.arch_hps_["conv_add_dense_for_latent"] = conv_add_dense_for_latent
        self.opt_hps_ = {
            "loss": loss,
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
        n_train_samples, window_size, n_features = X_train.shape
        self.n_train_samples_ = n_train_samples
        self.autoencoder_ = get_autoencoder(window_size, n_features, **self.arch_hps_)
        self.callbacks_ = get_callbacks(
            callbacks_type="train",
            output_path=self.window_model_path,
            model_file_name=self.model_file_name,
            save_weights_only=False,
            n_train_samples=n_train_samples,
            batch_size=self.train_hps_["batch_size"],
            n_epochs=self.train_hps_["n_epochs"],
            **self.callbacks_hps_,
        )
        compile_autoencoder(self.autoencoder_, **self.opt_hps_)
        train_dataset = get_autoencoder_dataset(
            X_train,
            self.train_hps_["shuffling_buffer_prop"],
            self.train_hps_["batch_size"],
        )
        val_dataset = None if X_val is None else get_autoencoder_dataset(X_val)
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        self.autoencoder_.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.train_hps_["n_epochs"],
            verbose=1,
            callbacks=self.callbacks_,
        )

    def _predict_window_model(self, X):
        return self.autoencoder_.predict(X)

    def _predict_window_scorer(self, X):
        """MSE scoring."""
        return np.mean(
            mean_squared_error(X, self.predict_window_model(X)).numpy(), axis=1
        )

    def __getstate__(self):
        # saving callbacks_ causes errors
        removed = ["autoencoder_", "callbacks_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        # TODO: wrap loading in function, maybe to move to helpers.
        try:
            self.autoencoder_ = load_model(
                os.path.join(self.window_model_path, self.model_file_name),
                custom_objects={"LayerBlock": LayerBlock, "LeakyReLU": LeakyReLU},
            )
        except OSError:  # works both if not found and permission denied
            # if not found, expect the keras model to be next to the detector file
            backup_window_model_path = os.path.abspath(os.curdir)
            self.autoencoder_ = load_model(
                os.path.join(backup_window_model_path, self.model_file_name),
                custom_objects={"LayerBlock": LayerBlock, "LeakyReLU": LeakyReLU},
            )
            self.window_model_path = backup_window_model_path
        self.callbacks_ = get_callbacks(
            callbacks_type="train",
            output_path=self.window_model_path,
            model_file_name=self.model_file_name,
            save_weights_only=False,
            n_train_samples=self.n_train_samples_,
            batch_size=self.train_hps_["batch_size"],
            n_epochs=self.train_hps_["n_epochs"],
            **self.callbacks_hps_,
        )
