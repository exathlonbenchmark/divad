import os
import random
import logging
from typing import Union

import numpy as np
import sklearn
from numpy.typing import NDArray
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
import keras_tuner
from keras_tuner import HyperModel
import tensorflow as tf

from utils.guarding import check_value_in_choices, check_is_not_none
from data.helpers import save_files, get_batches
from detection.detectors.helpers.general import (
    get_parsed_list_argument,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.helpers.tf_helpers import get_callbacks
from detection.detectors.helpers.tf_divad import TensorFlowDivad
from detection.detectors.base import BaseDetector
from detection.detectors.pca import Pca


def get_domains(data_info: dict, domain_key: str) -> NDArray[str]:
    if domain_key == "file_name":
        return data_info["file_name"]
    elif domain_key == "rate":
        return np.array([str(r) for r in data_info["input_rate"]])
    elif domain_key == "type-rate":
        trace_types = data_info["trace_type"]
        input_rates = data_info["input_rate"]
        return np.array([f"{t}_{r}" for t, r in zip(trace_types, input_rates)])
    elif domain_key == "settings-rate":
        settings = data_info["settings"]
        input_rates = data_info["input_rate"]
        return np.array([f"{s}_{r}" for s, r in zip(settings, input_rates)])
    elif domain_key == "app-type-rate":
        app_ids = data_info["app_id"]
        trace_types = data_info["trace_type"]
        input_rates = data_info["input_rate"]
        return np.array(
            [f"{a}_{t}_{r}" for a, t, r in zip(app_ids, trace_types, input_rates)]
        )
    elif domain_key == "app-settings-rate":
        app_ids = data_info["app_id"]
        settings = data_info["settings"]
        input_rates = data_info["input_rate"]
        return np.array(
            [f"{a}_{s}_{r}" for a, s, r in zip(app_ids, settings, input_rates)]
        )
    raise ValueError


class Divad(BaseDetector, HyperModel):
    relevant_steps = [
        "make_window_datasets",
        "train_window_model",
        "train_window_scorer",
        "train_online_scorer",
        "evaluate_online_scorer",
        "train_online_detector",
        "evaluate_online_detector",
    ]
    fitting_steps = [
        "train_window_model",
        "train_window_scorer",
        "train_online_detector",
    ]
    window_model_param_names = [
        # models and training callbacks
        "divad_",
        "domain_transformer_",
        "callbacks_",
        # architecture, optimization, training and callbacks hyperparameters
        "domain_key_",
        "time_freq_",
        "arch_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        # "needed" to properly re-set callbacks after loading
        "n_train_samples_",
        # needed to load the model (in addition to window size)
        "n_features_",
        "n_domains_",
    ]
    window_scorer_param_names = [
        "scoring_method_",
        "fit_val_",
        "agg_post_dist_",
        "agg_post_gm_n_components_",
        "agg_post_gm_seed_",
        "mean_nll_n_samples_",
        "mean_nll_seed_",
    ]
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "divad"
        self.divad_ = None
        self.domain_transformer_ = None
        self.callbacks_ = None
        self.search_callbacks_ = None
        self.searched_callbacks_ = None
        self.domain_key_ = None
        self.time_freq_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.scoring_method_ = None
        # only relevant for aggregated posterior scoring
        self.fit_val_ = None
        self.agg_post_dist_ = None
        self.agg_post_gm_n_components_ = None
        self.agg_post_gm_seed_ = None
        self.pca_detector_ = None
        self.gm_model_ = None
        # only relevant for "mean NLL"-based scoring
        self.mean_nll_n_samples_ = None
        self.mean_nll_seed_ = None
        self.n_train_samples_ = None
        self.n_features_ = None
        self.n_domains_ = None

    def set_window_model_params(
        self,
        domain_key: str = "app-settings-rate",
        type_: str = "dense",
        pzy_dist: str = "standard",
        pzy_kl_n_samples: int = 1,
        pzy_gm_n_components: int = 16,
        pzy_gm_softplus_scale: float = 1.0,
        pzy_vamp_n_components: int = 16,
        qz_x_conv1d_filters: Union[int, str] = "",
        qz_x_conv1d_kernel_sizes: Union[int, str] = "",
        qz_x_conv1d_strides: Union[int, str] = "",
        qz_x_n_hidden: Union[int, str] = "",
        pzd_d_n_hidden: Union[int, str] = "",
        px_z_conv1d_filters: Union[int, str] = "",
        px_z_conv1d_kernel_sizes: Union[int, str] = "",
        px_z_conv1d_strides: Union[int, str] = "",
        px_z_n_hidden: Union[int, str] = "",
        time_freq: bool = True,
        sample_normalize_x: bool = False,
        sample_normalize_mag: bool = True,
        apply_hann: bool = False,
        n_freq_modes: int = -1,
        phase_encoding: str = "cyclical",
        phase_cyclical_decoding: bool = True,
        qz_x_freq_conv1d_filters: Union[int, str] = "",
        qz_x_freq_conv1d_kernel_sizes: Union[int, str] = "",
        qz_x_freq_conv1d_strides: Union[int, str] = "",
        px_z_freq_conv1d_filters: Union[int, str] = "",
        px_z_freq_conv1d_kernel_sizes: Union[int, str] = "",
        px_z_freq_conv1d_strides: Union[int, str] = "",
        conv1d_pooling: bool = True,
        conv1d_batch_norm: bool = True,
        latent_dim: int = 10,
        rec_unit_type: str = "lstm",
        activation_rec: str = "tanh",
        rec_weight_decay: float = 0.0,
        dec_output_dist: str = "bernoulli",
        min_beta: float = 0.0,
        max_beta: float = 1.0,
        beta_n_epochs: int = 100,
        loss_weighting: str = "fixed",
        d_classifier_weight: float = 2000.0,
        softplus_shift: float = 1e-4,
        softplus_scale: float = 1.0,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        optimizer: str = "adam",
        adamw_weight_decay: float = 0.0,
        learning_rate: float = 3e-4,
        grad_norm_limit: float = 10.0,
        batch_size: int = 32,
        n_epochs: int = 300,
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
            domain_key: key defining the concept of domain: either "file_name", "rate", type-rate",
             "settings-rate", "app-type-rate" or "app-settings-rate".
            type_: type of `qz_x` and `px_z` networks (either "dense" or "rec").
            pzy_dist: prior distribution for zy (either "standard", "gm" or "vamp").
            pzy_kl_n_samples: number of MC samples to estimate the KL with pzy if `pzy_dist` is not "standard".
            pzy_gm_n_components: number of GM components if `pzy_dist` is "gm".
            pzy_gm_softplus_scale: softplus scale to apply to GM components to stabilize training if
             `pzy_dist` is "gm".
            pzy_vamp_n_components: number of pseudo inputs (i.e., mixture components) if `pzy_dist` is "vamp".
            qz_x_conv1d_filters: 1d convolution filters of the `qz_x` networks ("time" path if relevant).
            qz_x_conv1d_kernel_sizes: 1d convolution kernel sizes of the `qz_x` networks ("time" path if relevant).
            qz_x_conv1d_strides: `qz_x` 1d convolution strides of the `qz_x` networks ("time" path if relevant).
            qz_x_n_hidden: number of units for each hidden layer before the coding in qz_x networks, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            pzd_d_n_hidden: number of units for each hidden layer before the coding in the pzd_d network,
              specified similarly as `qz_x_n_hidden`.
            px_z_conv1d_filters: 1d convolution filters of the `px_z` network (the last number of filters
             should be the number of features, "time" path if relevant).
            px_z_conv1d_kernel_sizes: 1d convolution kernel sizes of the `px_z` network ("time" path if relevant).
            px_z_conv1d_strides: `qz_x` 1d convolution strides of the `px_z` network ("time" path if relevant).
            px_z_n_hidden: number of units for each hidden layer after the coding concatenation and before
              the output in the px_z network, specified similarly as `qz_x_n_hidden`.
            time_freq: whether to use a "time-frequency" architecture. If False, only modeling windows
              in the time domain.
            sample_normalize_x: whether to normalize x values per sample (and feature) before the FFT.
            sample_normalize_mag: whether to normalize the FFT magnitudes per sample after the FFT.
            apply_hann: whether to apply hann apodization to the input windows before the FFT.
            n_freq_modes: number of low-frequency modes to keep for the FFT (the mean is always excluded,
             -1 for keeping all other modes).
            phase_encoding: FFT phase encoding strategy ("none" for not encoding the phases, "raw" for encoding
             the phases directly, or "cyclical" for encoding the sine and cosines of phases).
            phase_cyclical_decoding: whether to consider a cyclical phase decoding, assuming the layer
             activations are the phases sines and cosines, or not, assuming they are the phase values directly.
            qz_x_freq_conv1d_filters: 1d convolution filters of the `qz_x` networks ("frequency" path).
            qz_x_freq_conv1d_kernel_sizes: 1d convolution kernel sizes of the `qz_x` networks ("frequency" path).
            qz_x_freq_conv1d_strides: `qz_x` 1d convolution strides of the `qz_x` networks ("frequency" path).
            px_z_freq_conv1d_filters: 1d convolution filters of the `px_z` network (not accounting for the
             last Conv1DTranspose layers, as they may be further split into two, "frequency" path).
            px_z_freq_conv1d_kernel_sizes: 1d convolution kernel sizes of the `px_z` network (accounting for the
             last layers, "frequency" path).
            px_z_freq_conv1d_strides: `qz_x` 1d convolution strides of the `px_z` network (accounting for the
             last layers, "frequency" path).
            conv1d_pooling: whether to perform downsampling and upsampling through pooling and upsampling
             layers rather than strided convolutions (the last decoder layer will always use strided convolutions).
            conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
            latent_dim: dimension of the latent vector representation (coding).
            rec_unit_type: type of recurrent unit used for recurrent DIVAD (either "lstm" or "gru").
            activation_rec: activation function used by recurrent DIVAD units (not the "recurrent activation").
            rec_weight_decay: L2 weight decay to apply to recurrent layers.
            dec_output_dist: output distribution of the decoder (either independent "bernoulli" or "normal").
            softplus_shift: (epsilon) shift to apply after softplus when computing standard deviations.
             The purpose is to stabilize training and prevent NaN probabilities by making sure
             standard deviations of normal distributions are non-zero.
             https://github.com/tensorflow/probability/issues/751 suggests 1e-5.
             https://arxiv.org/pdf/1802.03903.pdf uses 1e-4.
            softplus_scale: scale to apply in the softplus to stabilize training.
             See https://github.com/tensorflow/probability/issues/703 for more details.
            weight_decay: L2 weight decay to apply to feed-forward layers.
            dropout: dropout rate to apply to dense layer.
            min_beta: minimum beta value (KL divergence weight) for both z_y and z_d.
            max_beta: maximum beta value (KL divergence weight) for both z_y and z_d.
            beta_n_epochs: number of epochs during which to linearly increase beta from `min_beta` to `max_beta`.
            loss_weighting: loss weighting strategy (either "cov_weighting" or "fixed").
            d_classifier_weight: domain classification weight in the loss if `loss_weighting`
             is "fixed" (elbo is one).
            optimizer: optimization algorithm used for training the network.
            adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
            learning_rate: learning rate used by the optimization algorithm.
            grad_norm_limit: gradient norm clipping value.
            batch_size: mini-batch size.
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
        # turn list parameters to actual lists
        qz_x_conv1d_filters = get_parsed_list_argument(qz_x_conv1d_filters)
        qz_x_conv1d_kernel_sizes = get_parsed_list_argument(qz_x_conv1d_kernel_sizes)
        qz_x_conv1d_strides = get_parsed_list_argument(qz_x_conv1d_strides)
        qz_x_n_hidden = get_parsed_list_argument(qz_x_n_hidden)
        pzd_d_n_hidden = get_parsed_list_argument(pzd_d_n_hidden)
        px_z_conv1d_filters = get_parsed_list_argument(px_z_conv1d_filters)
        px_z_conv1d_kernel_sizes = get_parsed_list_argument(px_z_conv1d_kernel_sizes)
        px_z_conv1d_strides = get_parsed_list_argument(px_z_conv1d_strides)
        px_z_n_hidden = get_parsed_list_argument(px_z_n_hidden)
        qz_x_freq_conv1d_filters = get_parsed_list_argument(qz_x_freq_conv1d_filters)
        qz_x_freq_conv1d_kernel_sizes = get_parsed_list_argument(
            qz_x_freq_conv1d_kernel_sizes
        )
        qz_x_freq_conv1d_strides = get_parsed_list_argument(qz_x_freq_conv1d_strides)
        px_z_freq_conv1d_filters = get_parsed_list_argument(px_z_freq_conv1d_filters)
        px_z_freq_conv1d_kernel_sizes = get_parsed_list_argument(
            px_z_freq_conv1d_kernel_sizes
        )
        px_z_freq_conv1d_strides = get_parsed_list_argument(px_z_freq_conv1d_strides)
        check_value_in_choices(
            domain_key,
            "domain_key",
            [
                "file_name",
                "rate",
                "type-rate",
                "type-settings",
                "settings-rate",
                "app-type-rate",
                "app-settings-rate",
            ],
        )
        check_value_in_choices(type_, "type_", ["dense", "rec"])
        check_value_in_choices(
            phase_encoding, "phase_encoding", ["none", "raw", "cyclical"]
        )
        check_value_in_choices(pzy_dist, "pzy_dist", ["standard", "gm", "vamp"])
        if type_ == "rec":
            check_value_in_choices(rec_unit_type, "rec_unit_type", ["lstm", "gru"])
        check_value_in_choices(
            dec_output_dist, "dec_output_dist", ["bernoulli", "normal"]
        )
        check_value_in_choices(
            loss_weighting, "loss_weighting", ["cov_weighting", "fixed"]
        )
        check_value_in_choices(
            lr_scheduling, "lr_scheduling", ["none", "pw_constant", "one_cycle"]
        )
        check_value_in_choices(
            early_stopping_target,
            "early_stopping_target",
            ["loss", "val_loss", "val_neg_elbo"],
        )
        self.domain_key_ = domain_key
        self.time_freq_ = time_freq
        self.arch_hps_ = {
            "type_": type_,
            "pzy_dist": pzy_dist,
            "pzy_kl_n_samples": pzy_kl_n_samples,
            "pzy_gm_n_components": pzy_gm_n_components,
            "pzy_gm_softplus_scale": pzy_gm_softplus_scale,
            "pzy_vamp_n_components": pzy_vamp_n_components,
            "qz_x_conv1d_filters": qz_x_conv1d_filters,
            "qz_x_conv1d_kernel_sizes": qz_x_conv1d_kernel_sizes,
            "qz_x_conv1d_strides": qz_x_conv1d_strides,
            "qz_x_n_hidden": qz_x_n_hidden,
            "pzd_d_n_hidden": pzd_d_n_hidden,
            "px_z_conv1d_filters": px_z_conv1d_filters,
            "px_z_conv1d_kernel_sizes": px_z_conv1d_kernel_sizes,
            "px_z_conv1d_strides": px_z_conv1d_strides,
            "px_z_n_hidden": px_z_n_hidden,
            "latent_dim": latent_dim,
            "dec_output_dist": dec_output_dist,
            "softplus_shift": softplus_shift,
            "softplus_scale": softplus_scale,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "min_beta": min_beta,
            "loss_weighting": loss_weighting,
            "d_classifier_weight": d_classifier_weight,
            "conv1d_pooling": conv1d_pooling,
            "conv1d_batch_norm": conv1d_batch_norm,
        }
        if time_freq:
            time_freq_arch_hps = {
                "sample_normalize_x": sample_normalize_x,
                "sample_normalize_mag": sample_normalize_mag,
                "apply_hann": apply_hann,
                "n_freq_modes": n_freq_modes,
                "phase_encoding": phase_encoding,
                "phase_cyclical_decoding": phase_cyclical_decoding,
                "qz_x_freq_conv1d_filters": qz_x_freq_conv1d_filters,
                "qz_x_freq_conv1d_kernel_sizes": qz_x_freq_conv1d_kernel_sizes,
                "qz_x_freq_conv1d_strides": qz_x_freq_conv1d_strides,
                "px_z_freq_conv1d_filters": px_z_freq_conv1d_filters,
                "px_z_freq_conv1d_kernel_sizes": px_z_freq_conv1d_kernel_sizes,
                "px_z_freq_conv1d_strides": px_z_freq_conv1d_strides,
            }
            self.arch_hps_ = dict(self.arch_hps_, **time_freq_arch_hps)
        if type_ == "rec":
            self.arch_hps_["rec_unit_type"] = rec_unit_type
            self.arch_hps_["activation_rec"] = activation_rec
            self.arch_hps_["rec_weight_decay"] = rec_weight_decay
        self.opt_hps_ = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "grad_norm_limit": grad_norm_limit,
        }
        if optimizer == "adamw":
            self.opt_hps_["adamw_weight_decay"] = adamw_weight_decay
        self.train_hps_ = {
            "min_beta": min_beta,
            "max_beta": max_beta,
            "beta_n_epochs": beta_n_epochs,
            "batch_size": batch_size,
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
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ):
        X_train, one_hot_d_train, X_val, one_hot_d_val = self._get_data(
            X_train, y_train, X_val, y_val, train_info, val_info
        )
        self._set_model()
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
        self.callbacks_ += [
            BetaLinearScheduler(
                n_epochs=self.train_hps_["beta_n_epochs"],
                min_beta=self.train_hps_["min_beta"],
                max_beta=self.train_hps_["max_beta"],
            ),
            BetaLogger(),
        ]
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        if X_val is not None:
            validation_data = ((X_val, one_hot_d_val),)
        else:
            validation_data = None
        self.divad_.fit(
            (X_train, one_hot_d_train),
            validation_data=validation_data,
            batch_size=self.train_hps_["batch_size"],
            epochs=self.train_hps_["n_epochs"],
            verbose=1,
            callbacks=self.callbacks_,
        )

    def _get_data(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ):
        X_train, y_train, X_val, y_val, train_info, val_info = get_normal_windows(
            X_train, y_train, X_val, y_val, train_info, val_info
        )
        logging.info("Memory used after removing anomalies:")
        log_windows_memory(X_train, X_val)
        # domain encoding
        train_domains = get_domains(train_info, self.domain_key_)
        val_domains = get_domains(val_info, self.domain_key_)
        unique_train_domains = np.unique(train_domains)
        if X_val is not None:
            unique_val_domains = np.unique(val_domains)
            if set(unique_train_domains) != set(unique_val_domains):
                raise ValueError("Training and validation domains should be the same.")
        n_domains = len(unique_train_domains)
        sparse_kwarg = (
            {"sparse": False}
            if sklearn.__version__ == "1.0.2"
            else {"sparse_output": False}
        )
        self.domain_transformer_ = OneHotEncoder(dtype=np.float32, **sparse_kwarg)
        one_hot_d_train = self.domain_transformer_.fit_transform(
            train_domains.reshape(-1, 1)
        )
        if X_val is not None:
            one_hot_d_val = self.domain_transformer_.transform(
                val_domains.reshape(-1, 1)
            )
        else:
            one_hot_d_val = None
        n_train_samples, window_size, n_features = X_train.shape
        self.n_train_samples_ = n_train_samples
        self.window_size_ = window_size
        self.n_features_ = n_features
        self.n_domains_ = n_domains
        return X_train, one_hot_d_train, X_val, one_hot_d_val

    def _set_model(self):
        if self.time_freq_:
            self.divad_ = TensorFlowTimeFrequencyDivad(
                self.window_size_, self.n_features_, self.n_domains_, **self.arch_hps_
            )
        else:
            self.divad_ = TensorFlowDivad(
                self.window_size_, self.n_features_, self.n_domains_, **self.arch_hps_
            )
        compile_divad(self.divad_, **self.opt_hps_)

    def tune_window_model(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ):
        X_train, one_hot_d_train, X_val, one_hot_d_val = self._get_data(
            X_train, y_train, X_val, y_val, train_info, val_info
        )
        if X_val is not None:
            validation_data = ((X_val, one_hot_d_val),)
        else:
            validation_data = None
        # fixed callbacks used when searching hyperparameters
        search_output_path = f"{self.window_model_path}_searches"
        self.search_callbacks_ = get_callbacks(
            callbacks_type="search",
            output_path=search_output_path,
            model_file_name=self.model_file_name,
            save_weights_only=True,
            n_train_samples=self.n_train_samples_,
            batch_size=self.train_hps_["batch_size"],
            n_epochs=self.train_hps_["n_epochs"],
            **self.callbacks_hps_,
        )
        tuner = keras_tuner.Hyperband(
            hypermodel=self,
            objective="val_loss",
            max_epochs=150,
            factor=3,
            hyperband_iterations=1,
            executions_per_trial=1,
            overwrite=False,
            directory=search_output_path,
            project_name="hyperband_300_2",
        )
        tuner.search(
            (X_train, one_hot_d_train),
            validation_data=validation_data,
            batch_size=self.train_hps_["batch_size"],
            epochs=self.train_hps_["n_epochs"],
            verbose=1,
            callbacks=self.search_callbacks_,
        )

    def build(self, hp):
        # architecture hyperparameters
        if self.window_size_ == 300:
            n_freq_modes_choices = [32, 64, 128, -1]
        else:
            # default (meant for a window size of 120)
            n_freq_modes_choices = [16, 32, -1]
        arch = hp.Choice("arch", ["default", "smaller", "even_smaller"])
        arch_hps = {
            "phase_encoding": hp.Choice("phase_encoding", ["none", "raw"]),
            "latent_dim": hp.Choice("latent_dim", [16, 64, 128]),
            "pzy_dist": hp.Choice("pzy_dist", ["gm", "standard"]),
            "dropout": hp.Choice("dropout", [0.0, 0.5]),
            "conv1d_batch_norm": hp.Boolean("conv1d_batch_norm"),
            "apply_hann": hp.Boolean("apply_hann"),
            "n_freq_modes": hp.Choice("n_freq_modes", n_freq_modes_choices),
            "sample_normalize_mag": True,
        }
        arch_hps["sample_normalize_x"] = arch_hps["apply_hann"]
        gru_layer = hp.Boolean("gru_layer")
        if arch == "default":
            other_arch_hps = {
                "qz_x_conv1d_filters": [64, 64, 32],
                "qz_x_conv1d_kernel_sizes": [11, 7, 5],
                "qz_x_conv1d_strides": [1, 1, 1],
                "qz_x_n_hidden": [64] if gru_layer else [],
                "pzd_d_n_hidden": [32],
                "px_z_conv1d_filters": [32, 64, self.n_features_],
                "px_z_conv1d_kernel_sizes": [5, 7, 11],
                "px_z_conv1d_strides": [1, 1, 1],
                "px_z_n_hidden": [64] if gru_layer else [],
                # TODO: add frequency path params.
            }
        elif arch == "smaller":
            other_arch_hps = {
                "qz_x_conv1d_filters": [32, 32, 32],
                "qz_x_conv1d_kernel_sizes": [5, 5, 5],
                "qz_x_conv1d_strides": [1, 1, 1],
                "qz_x_n_hidden": [32] if gru_layer else [],
                "pzd_d_n_hidden": [32],
                "px_z_conv1d_filters": [32, 32, self.n_features_],
                "px_z_conv1d_kernel_sizes": [5, 5, 5],
                "px_z_conv1d_strides": [1, 1, 1],
                "px_z_n_hidden": [32] if gru_layer else [],
                # TODO: add frequency path params.
            }
        else:
            # even smaller
            other_arch_hps = {
                "qz_x_conv1d_filters": [32, 32],
                "qz_x_conv1d_kernel_sizes": [5, 5],
                "qz_x_conv1d_strides": [1, 1],
                "qz_x_n_hidden": [32] if gru_layer else [],
                "pzd_d_n_hidden": [32],
                "px_z_conv1d_filters": [32, self.n_features_],
                "px_z_conv1d_kernel_sizes": [5, 5],
                "px_z_conv1d_strides": [1, 1],
                "px_z_n_hidden": [32] if gru_layer else [],
                # TODO: add frequency path params.
            }
        arch_hps = dict(arch_hps, **other_arch_hps)
        for k, v in arch_hps.items():
            self.arch_hps_[k] = v
        # optimization hyperparameters
        opt_hps = {"learning_rate": hp.Choice("learning_rate", [1e-4, 3e-4])}
        for k, v in opt_hps.items():
            self.opt_hps_[k] = v
        max_beta = hp.Choice("max_beta", [1.0, 3.0, 5.0])
        kl_warmup = hp.Boolean("kl_warmup")
        train_hps = {
            "min_beta": 0.0 if kl_warmup else max_beta,
            "max_beta": max_beta,
            "beta_n_epochs": 100,
        }
        # training hyperparameters
        for k, v in train_hps.items():
            self.train_hps_[k] = v
        self._set_model()
        # "searched" callbacks (updated at each hyperparameter search trial)
        self.searched_callbacks_ = get_callbacks(
            callbacks_type="searched",
            output_path=self.window_model_path,
            model_file_name=self.model_file_name,
            save_weights_only=True,
            n_train_samples=self.n_train_samples_,
            batch_size=self.train_hps_["batch_size"],
            n_epochs=self.train_hps_["n_epochs"],
            **self.callbacks_hps_,
        )
        self.searched_callbacks_ += [
            BetaLinearScheduler(
                n_epochs=self.train_hps_["beta_n_epochs"],
                min_beta=self.train_hps_["min_beta"],
                max_beta=self.train_hps_["max_beta"],
            ),
            BetaLogger(),
        ]
        return self.divad_

    def fit(self, hp, model, *args, **kwargs):
        """HyperModel's `fit` overriding, required to update training hyperparameters."""
        # hyperband already handles the `epoch` argument (error if passed multiple times)
        extended_callbacks = self.searched_callbacks_ + kwargs.pop("callbacks", [])
        return model.fit(
            *args,
            callbacks=extended_callbacks,
            **kwargs,
        )

    def _predict_window_model(self, X):
        return self.divad_.predict(X)

    def set_window_scorer_params(
        self,
        scoring_method: str = "prior_nll_of_mean",
        fit_val: bool = True,
        mean_nll_n_samples: int = 1024,
        mean_nll_seed: int = 0,
        agg_post_dist: str = "gm",
        agg_post_gm_n_components: int = 32,
        agg_post_gm_seed: int = 0,
    ):
        """Sets hyperparameters relevant to the window scorer.

        Scoring methods:

        - "prior_mean_nll": score(x) := E_{qzy_x(zy|x)}[-pzy.log_prob(zy)].
        - "agg_post_mean_nll": score := E_{qzy_x(zy|x)}[-qzy.log_prob(zy)].
        - "mean_nll_n_samples": score := -pzy.log_prob(E_{qzy_x(zy|x)}[zy]).
        - "agg_post_nll_of_mean": score := -qzy.log_prob(E_{qzy_x(zy|x)}[zy]).

        With qzy the aggregated posterior distribution of the training and (optionally val) normal zy's,
        E_{qzy_x(zy|x)}[zy] readily available as qzy_x(x).mean(), and E_{qzy_x(zy|x)}[-{p}.log_prob(zy)]
        estimated using `mean_nll_n_samples` MC samples.

        Args:
            scoring_method: scoring method (see above).
            fit_val: whether to fit validation data for "aggregated posterior"-based scoring.
            mean_nll_n_samples: number of MC samples for "mean NLL"-based scoring.
            mean_nll_seed: random seed for "mean NLL"-based scoring.
            agg_post_dist: distribution assumed followed by the aggregated posterior (either "normal" or "gm").
            agg_post_gm_n_components: number of GMM components if `agg_post_dist` is "gm".
            agg_post_gm_seed: random seed to use if `agg_post_dist` is "gm".
        """
        check_value_in_choices(
            scoring_method,
            "scoring_method",
            [
                "prior_mean_nll",
                "agg_post_mean_nll",
                "prior_nll_of_mean",
                "agg_post_nll_of_mean",
            ],
        )
        check_value_in_choices(agg_post_dist, "agg_post_dist", ["normal", "gm"])
        self.scoring_method_ = scoring_method
        self.fit_val_ = fit_val
        self.mean_nll_n_samples_ = mean_nll_n_samples
        self.mean_nll_seed_ = mean_nll_seed
        self.agg_post_dist_ = agg_post_dist
        self.agg_post_gm_n_components_ = agg_post_gm_n_components
        self.agg_post_gm_seed_ = agg_post_gm_seed

    def _fit_window_scorer(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ):
        """If relevant, fits the mean and covariance of the aggregated posterior qzy."""
        if "agg_post" in self.scoring_method_:
            # fy(x) := E_{qzy_x(zy|x)}[zy]
            X_train, y_train, X_val, y_val = get_normal_windows(
                X_train, y_train, X_val, y_val
            )
            # cannot use `.predict()` to get distributions as output: batch data manually
            batch_size = 256
            train_batches = get_batches(X_train, batch_size=batch_size)
            zy_batches = []
            for train_batch in train_batches:
                zy_batches.append(
                    self.divad_.qzy_x(train_batch, training=False).mean().numpy()
                )
            if self.fit_val_ and X_val is not None:
                val_batches = get_batches(X_val, batch_size=batch_size)
                for val_batch in val_batches:
                    zy_batches.append(
                        self.divad_.qzy_x(val_batch, training=False).mean().numpy()
                    )
            elif self.fit_val_:
                raise ValueError("`X_val` must be provided to fit validation data.")
            zy = np.concatenate(zy_batches, axis=0)
            if self.agg_post_dist_ == "normal":
                self.pca_detector_ = Pca(
                    window_model_path=self.window_scorer_path,
                    window_scorer_path=self.window_scorer_path,
                    online_scorer_path="",
                    online_detector_path="",
                )
                # fit individual records: window size of one
                self.pca_detector_.fit_window_model(
                    np.expand_dims(zy, axis=1), train_info=train_info, val_info=val_info
                )
                self.pca_detector_.set_window_scorer_params(
                    method="mahalanobis", n_selected_components=-1
                )
            elif self.agg_post_dist_ == "gm":
                self.gm_model_ = GaussianMixture(
                    n_components=self.agg_post_gm_n_components_,
                    random_state=self.agg_post_gm_seed_,
                )
                self.gm_model_.fit(zy)
            else:
                raise ValueError(
                    f"Received invalid aggregated posterior distribution {self.agg_post_dist_}."
                )

    def _predict_window_scorer(self, X):
        """DIVAD window scoring according to `self.scoring_method`."""
        # cannot use `.predict()` to get distributions as output: batch data manually
        batch_size = 256
        batches = get_batches(X, batch_size=batch_size)
        window_score_batches = []
        for batch in batches:
            zy_dist = self.divad_.qzy_x(batch, training=False)
            window_scores = None
            if "nll_of_mean" in self.scoring_method_:
                zy = zy_dist.mean().numpy()
                if self.scoring_method_ == "prior_nll_of_mean":
                    window_scores = -self.divad_.pzy.log_prob(zy)
                    if np.any(np.isnan(window_scores)):
                        raise ValueError(
                            "Scoring returned NaN negative log-likelihoods."
                        )
                elif self.scoring_method_ == "agg_post_nll_of_mean":
                    if self.agg_post_dist_ == "normal":
                        check_is_not_none(self.pca_detector_, "self.pca_detector_")
                        window_scores = self.pca_detector_.predict_window_scorer(
                            np.expand_dims(zy, axis=1)
                        )
                    elif self.agg_post_dist_ == "gm":
                        check_is_not_none(self.gm_model_, "self.gm_model_")
                        window_scores = -self.gm_model_.score_samples(zy)
                    else:
                        raise ValueError(
                            f"Received invalid aggregated posterior distribution {self.agg_post_dist_}."
                        )
                else:
                    raise ValueError(
                        f"Received invalid scoring method {self.scoring_method_}."
                    )
            elif "mean_nll" in self.scoring_method_:
                # set relevant random seeds to improve reproducibility
                os.environ["PYTHONHASHSEED"] = str(self.mean_nll_seed_)
                random.seed(self.mean_nll_seed_)
                tf.random.set_seed(self.mean_nll_seed_)
                np.random.seed(self.mean_nll_seed_)
                window_scores = np.zeros(batch.shape[0])
                for _ in tqdm(range(self.mean_nll_n_samples_)):
                    # shape=[sample_shape=n_windows, latent_dim]
                    zy_samples = zy_dist.sample()
                    if self.scoring_method_ == "prior_mean_nll":
                        window_scores += -self.divad_.pzy.log_prob(zy_samples)
                        if np.any(np.isnan(window_scores)):
                            raise ValueError(
                                "Scoring returned NaN negative log-likelihoods."
                            )
                    elif self.scoring_method_ == "agg_post_mean_nll":
                        if self.agg_post_dist_ == "normal":
                            window_scores += self.pca_detector_.predict_window_scorer(
                                np.expand_dims(zy_samples, axis=1)
                            )
                        elif self.agg_post_dist_ == "gm":
                            window_scores -= self.gm_model_.score_samples(zy_samples)
                        else:
                            raise ValueError(
                                f"Received invalid aggregated posterior distribution {self.agg_post_dist_}."
                            )
                    else:
                        raise ValueError(
                            f"Received invalid scoring method {self.scoring_method_}."
                        )
                window_scores /= self.mean_nll_n_samples_
            window_score_batches.append(window_scores)
        window_scores = np.concatenate(window_score_batches, axis=0)
        return window_scores

    def __getstate__(self):
        # saving callbacks_ causes errors
        removed = ["divad_", "callbacks_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        self._set_model()
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
        self.callbacks_ += [
            BetaLinearScheduler(
                n_epochs=self.train_hps_["beta_n_epochs"],
                min_beta=self.train_hps_["min_beta"],
                max_beta=self.train_hps_["max_beta"],
            ),
            BetaLogger(),
        ]
        try:
            self.divad_.load_weights(
                os.path.join(self.window_model_path, self.model_file_name)
            )
        # works both if not found and permission denied
        # if not found, expect the keras model to be in the current directory
        except (OSError, tf.errors.NotFoundError):
            self.divad_.load_weights(os.path.join(os.curdir, self.model_file_name))
            self.window_model_path = os.curdir
