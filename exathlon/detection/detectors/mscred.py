import os
import time
import random
import logging
from typing import Union

import torch
import tensorflow as tf
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.guarding import check_value_in_choices, check_is_not_empty
from data.helpers import save_files
from detection.detectors.helpers.general import (
    get_parsed_integer_list_str,
    get_normal_windows,
    log_windows_memory,
)
from detection.detectors.base import BaseDetector
from detection.detectors.helpers.torch_helpers import (
    get_optimizer,
    Checkpointer,
    EarlyStopper,
)
from detection.detectors.helpers.torch_mscred import TorchMSCRED, get_mscred_loader
from detection.detectors.helpers.torch_mscred_helpers.correlation_matrix import (
    get_nearest_power_of_two,
)

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class Mscred(BaseDetector):
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
        "train_online_detector",
    ]
    window_model_param_names = [
        # model
        "model_",
        # architecture, optimization, training, callbacks and data hyperparameters
        "arch_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        "data_hps_",
        # "needed" to properly re-set callbacks after loading
        "n_train_samples_",
        "n_features_",
    ]
    window_scorer_param_names = []
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "model.pt"
        # model and training components
        self.model_ = None
        self.optimizer_ = None
        self.last_epoch_ = None
        self.last_checkpointed_loss_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.data_hps_ = None
        # needed to load the model
        self.n_train_samples_ = None
        self.n_features_ = None

    def set_window_model_params(
        self,
        signature_lengths: Union[int, str] = "20 10 5",
        filters: Union[int, str] = "32 64 128 256",
        kernel_sizes: Union[int, str] = "12 7 5 5",
        strides: Union[int, str] = "1 2 4 4",
        optimizer: str = "adamw",
        adamw_weight_decay: float = 0.01,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        n_epochs: int = 300,
        early_stopping_target: str = "val_loss",
        early_stopping_patience: int = 20,
    ):
        """Sets hyperparameters relevant to the window model.

        Args:
            signature_lengths: length of the signature matrices to compute correlations over, as an integer
              for a single matrix, or a string of space-separated integers for multiple matrices.
              Note: every signature matrix length should be smaller than the window size used.
            filters: number of filters for each Conv2D layer, as an integer for a single layer, or a
              string of space-separated integers for multiple layers.
            kernel_sizes: kernel sizes for each Conv2D layer, in the same format as `filters`.
            strides: strides for each Conv2D layer, in the same format as `filters`.
            optimizer: optimization algorithm used for training the network.
            adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
            learning_rate: learning rate used by the optimization algorithm.
            batch_size: mini-batch size.
            n_epochs: number of epochs to train the model for.
            early_stopping_target: early stopping target (either "loss" or "val_loss").
            early_stopping_patience: early stopping patience (in epochs).
        """
        # turn list parameters to actual lists
        if isinstance(signature_lengths, int):
            signature_lengths = str(signature_lengths)
        if isinstance(filters, int):
            filters = str(filters)
        if isinstance(kernel_sizes, int):
            kernel_sizes = str(kernel_sizes)
        if isinstance(strides, int):
            strides = str(strides)
        signature_lengths = get_parsed_integer_list_str(signature_lengths)
        filters = get_parsed_integer_list_str(filters)
        kernel_sizes = get_parsed_integer_list_str(kernel_sizes)
        strides = get_parsed_integer_list_str(strides)
        for v, s in zip(
            [signature_lengths, filters, kernel_sizes, strides],
            ["signature_lengths", "filters", "kernel_sizes", "strides"],
        ):
            check_is_not_empty(v, s)
        check_value_in_choices(
            optimizer,
            "optimizer",
            ["nag", "rmsprop", "adam", "nadam", "adadelta", "adamw"],
        )
        check_value_in_choices(
            early_stopping_target, "early_stopping_target", ["loss", "val_loss"]
        )
        self.arch_hps_ = {
            "n_signatures": len(signature_lengths),
            "filters": filters,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
        }
        self.opt_hps_ = {"optimizer": optimizer, "learning_rate": learning_rate}
        if optimizer == "adamw":
            self.opt_hps_["adamw_weight_decay"] = adamw_weight_decay
        self.train_hps_ = {"batch_size": batch_size, "n_epochs": n_epochs}
        # training callback hyperparameters
        self.callbacks_hps_ = {
            "early_stopping_target": early_stopping_target,
            "early_stopping_patience": early_stopping_patience,
        }
        self.data_hps_ = {
            "signature_lengths": signature_lengths,
            "input_reduction_factor": np.prod(strides),
        }

    def _fit_window_model(
        self,
        X_train: np.array,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ) -> None:
        logging.info("Memory used before removing anomalies:")
        log_windows_memory(X_train, X_val)
        X_train, y_train, X_val, y_val, train_info, val_info = get_normal_windows(
            X_train, y_train, X_val, y_val, train_info, val_info
        )
        logging.info("Memory used after removing anomalies:")
        log_windows_memory(X_train, X_val)
        if self.callbacks_hps_["early_stopping_target"] == "val_loss" and X_val is None:
            raise ValueError(
                "Validation data must be provided when specifying early stopping on validation loss."
            )
        n_train_samples, window_size, n_features = X_train.shape
        # the input features will get padded to the nearest (larger) power of two
        n_features = get_nearest_power_of_two(
            n_features, self.data_hps_["input_reduction_factor"]
        )
        self.n_train_samples_ = n_train_samples
        self.n_features_ = n_features
        self.model_ = TorchMSCRED(n_features=n_features, **self.arch_hps_)
        train_loader = get_mscred_loader(
            X_train,
            **self.data_hps_,
            batch_size=self.train_hps_["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        if X_val is not None:
            val_loader = get_mscred_loader(
                X_val,
                **self.data_hps_,
                batch_size=self.train_hps_["batch_size"],
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = None
        self.optimizer_ = get_optimizer(self.model_.parameters(), **self.opt_hps_)
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        early_stopper = EarlyStopper(
            patience=self.callbacks_hps_["early_stopping_patience"], min_delta=0
        )
        checkpointer = Checkpointer(
            model=self.model_,
            optimizer=self.optimizer_,
            scheduler=None,
            full_output_path=os.path.join(self.window_model_path, self.model_file_name),
        )
        writer = SummaryWriter(
            log_dir=os.path.join(
                self.window_model_path, time.strftime("%Y_%m_%d-%H_%M_%S")
            )
        )
        for epoch in range(self.train_hps_["n_epochs"]):
            n_train_batches = len(train_loader)
            pbar = tf.keras.utils.Progbar(target=n_train_batches)
            print(f'Epoch {epoch + 1}/{self.train_hps_["n_epochs"]}')
            # training
            train_loss_sum = 0.0
            self.model_.train()
            for train_idx, train_batch in enumerate(train_loader):
                self.optimizer_.zero_grad()
                train_batch_loss = self.model_.get_batch_loss(train_batch)
                train_batch_loss.backward(retain_graph=True)
                self.optimizer_.step()
                train_loss_sum += train_batch_loss.item()
                # loss metrics are the average training losses from the start of the epoch
                pbar.update(
                    train_idx, values=[("loss", train_loss_sum / (train_idx + 1))]
                )
            train_loss_mean = train_loss_sum / n_train_batches
            writer.add_scalar("loss/train", train_loss_mean, epoch)
            # validation
            if val_loader is not None:
                n_val_batches = len(val_loader)
                self.model_.eval()
                with torch.no_grad():  # less memory usage.
                    val_loss_sum = 0.0
                    for val_batch in val_loader:
                        val_batch_loss = self.model_.get_batch_loss(val_batch)
                        val_loss_sum += val_batch_loss.item()
                val_loss_mean = val_loss_sum / n_val_batches
                pbar.update(n_train_batches, values=[("val_loss", val_loss_mean)])
                writer.add_scalar("loss/val", val_loss_mean, epoch)
                if self.callbacks_hps_["early_stopping_target"] == "val_loss":
                    early_stopped_loss = val_loss_mean
                    checkpointed_loss = val_loss_mean
                else:
                    early_stopped_loss = train_loss_mean
                    checkpointed_loss = train_loss_mean
            else:
                early_stopped_loss = train_loss_mean
                checkpointed_loss = train_loss_mean
            # epoch callbacks
            checkpointer.checkpoint(epoch, checkpointed_loss)
            if early_stopper.early_stop(early_stopped_loss):
                break
        writer.flush()
        writer.close()

    def _predict_window_model(self, X):
        pass

    def _predict_window_scorer(self, X):
        """`X` should correspond to sequential windows in a given sequence."""
        dataloader = get_mscred_loader(
            X,
            **self.data_hps_,
            batch_size=self.train_hps_["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        window_scores = self.model_.get_window_scores(dataloader)
        return window_scores

    def __getstate__(self):
        removed = ["model_", "optimizer_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        self.model_ = TorchMSCRED(n_features=self.n_features_, **self.arch_hps_)
        self.optimizer_ = get_optimizer(self.model_.parameters(), **self.opt_hps_)
        try:
            checkpoint = torch.load(
                os.path.join(self.window_model_path, self.model_file_name),
                map_location=torch.device("cpu"),
            )
        except OSError:  # works both if not found and permission denied
            checkpoint = torch.load(
                os.path.join(os.curdir, self.model_file_name),
                map_location=torch.device("cpu"),
            )
            self.window_model_path = os.curdir

        self.model_.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_.load_state_dict(checkpoint["optimizer_state_dict"])
        self.last_epoch_ = checkpoint["epoch"]
        self.last_checkpointed_loss_ = checkpoint["loss"]
