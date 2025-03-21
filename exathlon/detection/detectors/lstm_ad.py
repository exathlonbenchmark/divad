import os
import logging
from typing import Union, Optional, List

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

from utils.guarding import check_value_in_choices
from data.helpers import get_sliding_windows, save_files
from detection.detectors.helpers.general import (
    get_parsed_integer_list_str,
    get_nll,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.helpers.tf_helpers import get_callbacks, LayerBlock
from detection.detectors.helpers.tf_rnn import get_rnn, compile_rnn, get_rnn_dataset
from detection.detectors.base import BaseDetector


class LstmAd(BaseDetector):
    relevant_steps = [
        "make_window_datasets",
        "train_window_model",
        "train_online_scorer",
        "evaluate_online_scorer",
        "train_online_detector",
        "evaluate_online_detector",
    ]
    fitting_steps = [
        "train_window_model",
        "train_online_scorer",
        "train_online_detector",
    ]
    window_model_param_names = [
        # rnn model and training callbacks
        "rnn_",
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
    online_scorer_param_names = ["error_averaging_", "error_mean_", "error_cov_"]
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "rnn.h5"
        # window model
        self.rnn_ = None
        self.callbacks_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.n_train_samples_ = None
        # online scorer
        self.error_averaging_ = None
        self.error_mean_ = None
        self.error_cov_ = None

    def set_window_model_params(
        self,
        n_forward: int = 20,
        conv1d_filters: Union[int, str] = "",
        conv1d_kernel_sizes: Union[int, str] = "",
        conv1d_strides: Union[int, str] = "",
        conv1d_pooling: bool = True,
        conv1d_batch_norm: bool = True,
        unit_type: str = "lstm",
        n_hidden_neurons: Union[int, str] = "",
        dropout: float = 0.0,
        rec_dropout: float = 0.0,
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
            n_forward: number of time steps to forecast ahead.
            conv1d_filters: number of filters for each Conv1D layer before the hidden layers, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            conv1d_kernel_sizes: kernel sizes for each Conv1D layer before the hidden layers, in the
             same format as `enc_conv1d_kernel_sizes`.
            conv1d_strides: strides for each Conv1D layer before the hidden layers, in the
             same format as `enc_conv1d_kernel_sizes`.
            conv1d_pooling: whether to perform downsampling pooling rather than strided convolutions.
            conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
            unit_type: type of recurrent unit (either "lstm" or "gru").
            n_hidden_neurons: number of units for each hidden layer before the output, as
              an empty string for no layer, an integer for a single layer, or a string of
              space-separated integers for multiple layers.
            dropout: dropout rate for feed-forward layers.
            rec_dropout: recurrent dropout rate.
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
        check_value_in_choices(unit_type, "unit_type", ["lstm", "gru"])
        check_value_in_choices(
            lr_scheduling, "lr_scheduling", ["none", "pw_constant", "one_cycle"]
        )
        check_value_in_choices(
            early_stopping_target, "early_stopping_target", ["loss", "val_loss"]
        )
        # turn list parameters to actual lists
        if isinstance(conv1d_filters, int):
            conv1d_filters = str(conv1d_filters)
        if isinstance(conv1d_kernel_sizes, int):
            conv1d_kernel_sizes = str(conv1d_kernel_sizes)
        if isinstance(conv1d_strides, int):
            conv1d_strides = str(conv1d_strides)
        if isinstance(n_hidden_neurons, int):
            n_hidden_neurons = str(n_hidden_neurons)
        conv1d_filters = get_parsed_integer_list_str(conv1d_filters)
        conv1d_kernel_sizes = get_parsed_integer_list_str(conv1d_kernel_sizes)
        conv1d_strides = get_parsed_integer_list_str(conv1d_strides)
        n_hidden_neurons = get_parsed_integer_list_str(n_hidden_neurons)
        self.arch_hps_ = {
            "n_forward": n_forward,
            "conv1d_filters": conv1d_filters,
            "conv1d_kernel_sizes": conv1d_kernel_sizes,
            "conv1d_strides": conv1d_strides,
            "conv1d_pooling": conv1d_pooling,
            "conv1d_batch_norm": conv1d_batch_norm,
            "unit_type": unit_type,
            "n_hidden_neurons": n_hidden_neurons,
            "dropout": dropout,
            "rec_dropout": rec_dropout,
        }
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
        n_train_samples, window_size, n_features = X_train.shape
        # window size actually used by the model
        self.window_size_ = window_size - self.arch_hps_["n_forward"]
        self.n_train_samples_ = n_train_samples
        self.rnn_ = get_rnn(self.window_size_, n_features, **self.arch_hps_)
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
        compile_rnn(self.rnn_, **self.opt_hps_)
        train_dataset = get_rnn_dataset(
            X_train,
            self.arch_hps_["n_forward"],
            self.train_hps_["shuffling_buffer_prop"],
            self.train_hps_["batch_size"],
        )
        if X_val is None:
            val_dataset = None
        else:
            val_dataset = get_rnn_dataset(X_val, self.arch_hps_["n_forward"])
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        self.rnn_.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.train_hps_["n_epochs"],
            verbose=1,
            callbacks=self.callbacks_,
        )

    def _predict_window_model(self, X):
        return self.rnn_.predict(X)

    def set_online_scorer_params(
        self, error_averaging: str = "none", scores_avg_beta: float = 0.9867
    ):
        """Sets hyperparameters relevant to the online scorer."""
        check_value_in_choices(
            error_averaging, "error_averaging", ["none", "time", "features"]
        )
        self.error_averaging_ = error_averaging
        self.scores_avg_beta_ = scores_avg_beta

    def _fit_online_scorer(
        self,
        train: List[np.array],
        y_train: Optional[List[np.array]] = None,
        val: Optional[List[np.array]] = None,
        y_val: Optional[List[np.array]] = None,
    ) -> None:
        if val is not None:
            sequences = val
        else:
            logging.warning(
                "No validation sequences to fit LSTM-AD, fitting on train sequences instead."
            )
            sequences = train
        errors = np.array([], dtype=np.float32)
        for seq in sequences:
            seq_errors = self.get_sequence_errors(seq)
            errors = (
                np.concatenate([errors, seq_errors]) if len(errors) > 0 else seq_errors
            )
        # fit multivariate Gaussian to error vectors
        self.error_mean_ = np.mean(errors, axis=0)
        self.error_cov_ = np.cov(errors, rowvar=False)

    def get_sequence_errors(self, seq: np.array) -> np.array:
        """Returns error vectors for the records of `seq`.

        Error vectors will not be returned for neither the first `(window_size + n_forward - 1)`
        record nor the last `(n_forward - 1)` records of `seq`. The returned sequence of errors will
        hence contain `n_errors = seq.shape[0] - (window_size + 2 * n_forward - 2)` records.

        The dimension of error vectors `error_dim` depends on `self.error_averaging_`:

        - If "none": `error_dim` will be `n_forward * n_features`.
        - If "time": `error_dim` will be `n_features` (average of errors through time for each feature).
        - If "features": `error_dim` will be `n_forward` (average of feature errors for each time step).

        Args:
            seq: input sequence, to return error vectors for.

        Returns:
            Error vectors for the records of `seq`, of shape `(n_errors, error_dim)`, as described above.
        """
        window_size = self.window_size_  # called "w" in the following
        n_forward = self.arch_hps_["n_forward"]  # called "l" in the following
        window_preds = self.predict_window_model(
            get_sliding_windows(seq, window_size, 1)
        )
        true_windows = get_sliding_windows(seq, n_forward, 1)
        # align true windows with window predictions
        true_windows = true_windows[window_size:]  # w records used to predict
        window_preds = window_preds[:-n_forward]  # l too many predictions at the end
        assert true_windows.shape == window_preds.shape
        window_errors = window_preds - true_windows
        n_features = seq.shape[1]
        n_records = window_errors.shape[0] + (n_forward - 1)  # l-1 records at the end
        record_to_error_window = np.full((n_records, n_forward, n_features), np.nan)
        for window_idx, window_error in enumerate(window_errors):
            for row_idx, row_error in enumerate(window_error):
                record_idx = window_idx + row_idx
                record_to_error_window[record_idx, row_idx, :] = row_error
        # the first and last l-1 records have incomplete error vectors
        n_removed = n_forward - 1
        record_to_error_window = record_to_error_window[n_removed:-n_removed]
        assert not np.any(np.isnan(record_to_error_window))
        n_errors = record_to_error_window.shape[0]
        # we removed w+l-1 records from the start of the sequence, and l-1 from the end
        assert seq.shape[0] - (window_size + n_removed) - n_removed == n_errors
        if self.error_averaging_ == "none":
            # flatten error windows per record
            errors = record_to_error_window.reshape((n_errors, n_forward * n_features))
        else:
            averaged_axis = 1 if self.error_averaging_ == "time" else 2
            errors = np.mean(record_to_error_window, axis=averaged_axis)
        return errors

    def _predict_online_scorer(self, sequences: List[np.array]) -> List[np.array]:
        window_size = self.window_size_
        n_forward = self.arch_hps_["n_forward"]
        neg_inf = np.core.getlimits.finfo(sequences[0][0].dtype).min
        sequences_scores = []
        for seq in sequences:
            seq_errors = self.get_sequence_errors(seq)
            seq_scores = get_nll(seq_errors, self.error_mean_, self.error_cov_)
            # apply exponentially weighted average of the sequence's record-wise outlier scores
            seq_scores = (
                pd.Series(seq_scores)
                .ewm(alpha=1 - self.scores_avg_beta_, adjust=True)
                .mean()
                .values
            )
            # add back first w+l-1 records and last l-1 records of the sequence as lowest scores
            seq_scores = np.nan_to_num(
                np.concatenate(
                    [
                        np.full(window_size + n_forward - 1, neg_inf),
                        seq_scores,
                        np.full(n_forward - 1, neg_inf),
                    ]
                )
            )
            assert seq_scores.shape[0] == seq.shape[0]
            sequences_scores.append(seq_scores)
        return sequences_scores

    def __getstate__(self):
        # saving callbacks_ causes errors
        removed = ["rnn_", "callbacks_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        # TODO: wrap loading in function, maybe to move to helpers.
        try:
            self.rnn_ = load_model(
                os.path.join(self.window_model_path, self.model_file_name),
                custom_objects={"LayerBlock": LayerBlock, "LeakyReLU": LeakyReLU},
            )
        except OSError:  # works both if not found and permission denied
            # if not found, expect the keras model to be next to the detector file
            self.rnn_ = load_model(
                os.path.join(os.curdir, self.model_file_name),
                custom_objects={"LayerBlock": LayerBlock, "LeakyReLU": LeakyReLU},
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
