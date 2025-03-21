import os
import time
import logging

import torch
import tensorflow as tf
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.guarding import check_value_in_choices
from data.helpers import save_files
from detection.detectors.helpers.general import get_normal_windows, log_windows_memory
from detection.detectors.base import BaseDetector
from detection.detectors.helpers.torch_helpers import (
    get_optimizer,
    Checkpointer,
    EarlyStopper,
)
from detection.detectors.helpers.torch_tranad import TorchTranad, get_tranad_loader


class Tranad(BaseDetector):
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
        "tranad_",
        # architecture, optimization, training and callbacks hyperparameters
        "arch_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        # "needed" to properly re-set callbacks after loading
        "n_train_samples_",
        "n_features_",
    ]
    window_scorer_param_names = []
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_file_name = "tranad.pt"
        # model and training components
        self.tranad_ = None
        self.optimizer_ = None
        self.scheduler_ = None
        self.last_epoch_ = None
        self.last_checkpointed_loss_ = None
        self.arch_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        # needed to load the model
        self.n_features_ = None

    def set_window_model_params(
        self,
        dim_feedforward: int = 64,
        last_activation: str = "sigmoid",
        optimizer: str = "adam",
        adamw_weight_decay: float = 0.0,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        n_epochs: int = 400,
        early_stopping_target: str = "val_loss",
        early_stopping_patience: int = 20,
    ):
        """Sets hyperparameters relevant to the window model.

        Args:
            dim_feedforward: dimension of feed-forward layers.
            last_activation: activation function of the last FC layer (either "sigmoid" or "linear").
            optimizer: optimization algorithm used for training the network.
            adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
            learning_rate: learning rate used by the optimization algorithm.
            batch_size: mini-batch size.
            n_epochs: number of epochs to train the model for.
            early_stopping_target: early stopping target (either "loss" or "val_loss").
            early_stopping_patience: early stopping patience (in epochs).
        """
        check_value_in_choices(
            last_activation, "last_activation", ["sigmoid", "linear"]
        )
        check_value_in_choices(
            optimizer,
            "optimizer",
            ["nag", "rmsprop", "adam", "nadam", "adadelta", "adamw"],
        )
        check_value_in_choices(
            early_stopping_target, "early_stopping_target", ["loss", "val_loss"]
        )
        self.arch_hps_ = {
            "dim_feedforward": dim_feedforward,
            "last_activation": last_activation,
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
        self.n_train_samples_ = n_train_samples
        self.n_features_ = n_features
        self.tranad_ = TorchTranad(
            window_size=window_size, n_features=n_features, **self.arch_hps_
        )
        # every batch should be contiguous windows from a same sequence
        train_loader = get_tranad_loader(
            X_train,
            train_info,
            batch_size=self.train_hps_["batch_size"],
        )
        if X_val is not None:
            # the batch size should be the same for validation (as it is considered by sequence)
            val_loader = get_tranad_loader(
                X_val, val_info, batch_size=self.train_hps_["batch_size"]
            )
        else:
            val_loader = None
        self.optimizer_ = get_optimizer(
            self.tranad_.model.parameters(), **self.opt_hps_
        )
        self.scheduler_ = torch.optim.lr_scheduler.StepLR(self.optimizer_, 5, 0.9)
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")
        early_stopper = EarlyStopper(
            patience=self.callbacks_hps_["early_stopping_patience"], min_delta=0
        )
        checkpointer = Checkpointer(
            self.tranad_.model,
            self.optimizer_,
            self.scheduler_,
            os.path.join(self.window_model_path, self.model_file_name),
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
            self.tranad_.model.train()
            for train_idx, train_batch in enumerate(train_loader):
                self.optimizer_.zero_grad()
                train_batch_loss = self.tranad_.get_batch_loss(train_batch, epoch)
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
                self.tranad_.model.eval()
                with torch.no_grad():  # less memory usage.
                    val_loss_sum = 0.0
                    for val_batch in val_loader:
                        val_batch_loss = self.tranad_.get_batch_loss(val_batch, epoch)
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
            self.scheduler_.step()
        writer.flush()
        writer.close()

    def _predict_window_model(self, X):
        pass

    def _predict_window_scorer(self, X):
        """`X` should correspond to sequential windows in a given sequence."""
        dataloader = DataLoader(X, batch_size=len(X), shuffle=False, drop_last=False)
        window_scores = self.tranad_.get_window_scores(dataloader)
        return window_scores

    def __getstate__(self):
        removed = ["tranad_", "optimizer_", "scheduler_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        self.tranad_ = TorchTranad(
            window_size=self.window_size_, n_features=self.n_features_, **self.arch_hps_
        )
        self.optimizer_ = get_optimizer(
            self.tranad_.model.parameters(), **self.opt_hps_
        )
        self.scheduler_ = torch.optim.lr_scheduler.StepLR(self.optimizer_, 5, 0.9)
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

        self.tranad_.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler_.load_state_dict(checkpoint["scheduler"])
        self.last_epoch_ = checkpoint["epoch"]
        self.last_checkpointed_loss_ = checkpoint["loss"]
