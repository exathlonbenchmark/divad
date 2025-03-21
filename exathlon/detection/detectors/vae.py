import os
import gc
import random
import logging
from typing import Union

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

from utils.guarding import check_value_in_choices
from data.helpers import save_files
from detection.detectors.helpers.general import (
    get_parsed_integer_list_str,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.helpers.tf_helpers import get_callbacks, LayerBlock
from detection.detectors.helpers.tf_vae import get_vae, compile_vae, KLLossCallback
from detection.detectors.base import BaseDetector


class Vae(BaseDetector):
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
    window_model_param_names = [
        # vae model and training callbacks
        "vae_",
        "callbacks_",
        # architecture, optimization, training and callbacks hyperparameters
        "arch_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        # "needed" to properly re-set callbacks after loading
        "n_train_samples_",
    ]
    window_scorer_param_names = ["reco_prob_n_samples", "scoring_seed"]
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "vae.h5"
        self.vae_ = None
        self.callbacks_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.n_train_samples_ = None
        self.reco_prob_n_samples = None
        self.scoring_seed = None

    def set_window_model_params(
        self,
        type_: str = "dense",
        enc_conv1d_filters: Union[int, str] = "",
        enc_conv1d_kernel_sizes: Union[int, str] = "",
        enc_conv1d_strides: Union[int, str] = "",
        conv1d_pooling: bool = False,
        conv1d_batch_norm: bool = False,
        enc_n_hidden_neurons: Union[int, str] = "",
        dec_n_hidden_neurons: Union[int, str] = "",
        dec_conv1d_filters: Union[int, str] = "",
        dec_conv1d_kernel_sizes: Union[int, str] = "",
        dec_conv1d_strides: Union[int, str] = "",
        latent_dim: int = 10,
        dec_output_dist: str = "normal",
        dense_layers_activation: str = "relu",
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rec_unit_type: str = "lstm",
        activation_rec: str = "tanh",
        rec_dropout: float = 0.0,
        rec_latent_type: str = "dense",
        kl_weight: float = 1.0,
        softplus_shift: float = 1e-4,
        softplus_scale: float = 1.0,
        optimizer: str = "adam",
        adamw_weight_decay: float = 0.0,
        learning_rate: float = 3e-4,
        grad_norm_limit: float = 10.0,
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
            type_: type of vae to build.
            enc_conv1d_filters: number of filters for each Conv1D layer before the hidden layers, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            enc_conv1d_kernel_sizes: kernel size for each Conv1D layer before the hidden layers, specified as
             `enc_conv1d_filters`.
            enc_conv1d_strides: strides for each Conv1D layer before the hidden layers, specified as
             `enc_conv1d_filters`.
              space-separated integers for multiple layers.
            conv1d_pooling: whether to perform downsampling and upsampling through pooling and upsampling
             layers rather than strided convolutions (the last decoder layer will always use strided convolutions).
            conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
            enc_n_hidden_neurons: number of units for each hidden layer before the coding, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            dec_n_hidden_neurons: number of units for each hidden layer after the coding, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            dec_conv1d_filters: number of filters for each Conv1DTranspose layer after the hidden layers,
             specified as `enc_conv1d_filters`.
            dec_conv1d_kernel_sizes: kernel size for each Conv1DTranspose layer after the hidden layers,
             specified as `enc_conv1d_filters`.
            dec_conv1d_strides: strides for each Conv1DTranspose layer after the hidden layers,
             specified as `enc_conv1d_filters`.
            latent_dim: dimension of the latent vector representation (coding).
            dec_output_dist: decoder output distribution (either independent "bernoulli" or "normal").
            dense_layers_activation: intermediate layers activation for dense architectures.
            input_dropout: dropout rate for the input layer.
            hidden_dropout: dropout rate for other feed-forward layers (except the output).
            rec_unit_type: type of recurrent unit (either "lstm" or "gru").
            activation_rec: activation function to use for recurrent layers (not the "recurrent activation").
            rec_dropout: dropout rate for the hidden state of recurrent layers.
            rec_latent_type: type of latent layers for recurrent architectures.
            kl_weight: KL divergence term weight in the loss, set here as an activity regularizer.
            softplus_shift: (epsilon) shift to apply after softplus when computing standard deviations.
             The purpose is to stabilize training and prevent NaN probabilities by making sure
             standard deviations of normal distributions are non-zero.
             https://github.com/tensorflow/probability/issues/751 suggests 1e-5.
             https://arxiv.org/pdf/1802.03903.pdf uses 1e-4.
            softplus_scale: scale to apply in the softplus to stabilize training.
             See https://github.com/tensorflow/probability/issues/703 for more details.
            optimizer: optimization algorithm used for training the network.
            adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
            learning_rate: learning rate used by the optimization algorithm.
            grad_norm_limit: gradient norm clipping value.
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
            dec_output_dist, "dec_output_dist", ["bernoulli", "normal"]
        )
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
        if isinstance(dec_n_hidden_neurons, int):
            dec_n_hidden_neurons = str(dec_n_hidden_neurons)
        if isinstance(dec_conv1d_filters, int):
            dec_conv1d_filters = str(dec_conv1d_filters)
        if isinstance(dec_conv1d_kernel_sizes, int):
            dec_conv1d_kernel_sizes = str(dec_conv1d_kernel_sizes)
        if isinstance(dec_conv1d_strides, int):
            dec_conv1d_strides = str(dec_conv1d_strides)
        enc_conv1d_filters = get_parsed_integer_list_str(enc_conv1d_filters)
        enc_conv1d_kernel_sizes = get_parsed_integer_list_str(enc_conv1d_kernel_sizes)
        enc_conv1d_strides = get_parsed_integer_list_str(enc_conv1d_strides)
        enc_n_hidden_neurons = get_parsed_integer_list_str(enc_n_hidden_neurons)
        dec_n_hidden_neurons = get_parsed_integer_list_str(dec_n_hidden_neurons)
        dec_conv1d_filters = get_parsed_integer_list_str(dec_conv1d_filters)
        dec_conv1d_kernel_sizes = get_parsed_integer_list_str(dec_conv1d_kernel_sizes)
        dec_conv1d_strides = get_parsed_integer_list_str(dec_conv1d_strides)
        self.arch_hps_ = {
            "type_": type_,
            "enc_conv1d_filters": enc_conv1d_filters,
            "enc_conv1d_kernel_sizes": enc_conv1d_kernel_sizes,
            "enc_conv1d_strides": enc_conv1d_strides,
            "conv1d_pooling": conv1d_pooling,
            "conv1d_batch_norm": conv1d_batch_norm,
            "enc_n_hidden_neurons": enc_n_hidden_neurons,
            "dec_n_hidden_neurons": dec_n_hidden_neurons,
            "dec_conv1d_filters": dec_conv1d_filters,
            "dec_conv1d_kernel_sizes": dec_conv1d_kernel_sizes,
            "dec_conv1d_strides": dec_conv1d_strides,
            "latent_dim": latent_dim,
            "dec_output_dist": dec_output_dist,
            "input_dropout": input_dropout,
            "hidden_dropout": hidden_dropout,
            "dense_layers_activation": dense_layers_activation,
            "kl_weight": kl_weight,
            "softplus_shift": softplus_shift,
            "softplus_scale": softplus_scale,
        }
        if type_ == "rec":
            self.arch_hps_["rec_unit_type"] = rec_unit_type
            self.arch_hps_["activation_rec"] = activation_rec
            self.arch_hps_["rec_dropout"] = rec_dropout
            self.arch_hps_["rec_latent_type"] = rec_latent_type
        self.opt_hps_ = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "grad_norm_limit": grad_norm_limit,
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
        self.vae_ = get_vae(window_size, n_features, **self.arch_hps_)
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
        # TODO: log KL loss in TensorBoard.
        self.callbacks_.append(KLLossCallback())
        compile_vae(self.vae_, **self.opt_hps_)
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        self.vae_.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=self.train_hps_["n_epochs"],
            verbose=1,
            callbacks=self.callbacks_,
        )

    def _predict_window_model(self, X):
        return self.vae_.predict(X)

    def set_window_scorer_params(
        self, reco_prob_n_samples: int = 1024, scoring_seed: int = 0
    ):
        """Sets hyperparameters relevant to the window scorer.

        Args:
            reco_prob_n_samples: number of samples to use when computing the reconstruction probability.
            scoring_seed: random seed used for reconstruction probability scoring.
        """
        self.reco_prob_n_samples = reco_prob_n_samples
        self.scoring_seed = scoring_seed

    def _predict_window_scorer(self, X):
        """Reconstruction probability scoring.

        TODO: replace loop with broadcasting to improve efficiency.
        """
        # set relevant random seeds to improve reproducibility
        os.environ["PYTHONHASHSEED"] = str(self.scoring_seed)
        random.seed(self.scoring_seed)
        tf.random.set_seed(self.scoring_seed)
        np.random.seed(self.scoring_seed)

        n_windows, window_size, n_features = X.shape
        # batch_shape=[n_windows], event_shape=[latent_dim]
        z_dists = self.vae_.get_layer("encoder")(X)
        scores = np.zeros(n_windows)
        for _ in tqdm(range(self.reco_prob_n_samples)):
            # shape=[sample_shape=n_windows, latent_dim]
            z_samples = z_dists.sample()
            # batch_shape=[n_windows], event_shape=[window_size, n_features]
            x_dists = self.vae_.get_layer("decoder")(z_samples)
            neg_log_probs = -x_dists.log_prob(X)
            if np.any(np.isnan(neg_log_probs)):
                raise ValueError("Scoring returned NaN negative log-likelihoods.")
            scores += neg_log_probs
            # alleviate memory leak from  model predictions
            del neg_log_probs
            del z_samples
            del x_dists
            _ = gc.collect()
        scores /= self.reco_prob_n_samples
        return scores

    def __getstate__(self):
        # saving callbacks_ causes errors
        removed = ["vae_", "callbacks_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        # TODO: wrap loading in function, maybe to move to helpers.
        try:
            # no need to load the custom loss function
            self.vae_ = load_model(
                os.path.join(self.window_model_path, self.model_file_name),
                custom_objects={"LayerBlock": LayerBlock, "LeakyReLU": LeakyReLU},
                compile=False,
            )
        except OSError:  # works both if not found and permission denied
            # if not found, expect the keras model to be next to the detector file
            self.vae_ = load_model(
                os.path.join(os.curdir, self.model_file_name),
                custom_objects={"LayerBlock": LayerBlock, "LeakyReLU": LeakyReLU},
                compile=False,
            )
            self.window_model_path = os.curdir
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
        self.callbacks_.append(KLLossCallback())
