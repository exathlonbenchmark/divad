import os
import time
import random
import logging
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.guarding import check_value_in_choices, check_is_not_none
from data.helpers import save_files
from detection.detectors.helpers.general import (
    log_windows_memory,
    get_parsed_list_argument,
    get_balanced_samples,
    get_normal_windows,
)
from detection.detectors.helpers.tf_contrastive_autoencoder import (
    get_balanced_ano_types,
)
from detection.detectors.base import BaseDetector
from detection.detectors.helpers.torch_helpers import (
    get_optimizer,
    Checkpointer,
    EarlyStopper,
)
from detection.detectors.helpers.torch_deep_sad import DeepSADDataset, TorchDeepSAD


def train_model(
    train_loader,
    val_loader,
    model,
    get_batch_loss_func,
    optimizer,
    scheduler,
    early_stopper,
    checkpointer,
    writer,
    n_epochs,
    early_stopping_target,
    model_name="",
):
    # TODO: move to torch helpers and use in TranAD as well.
    model_name_str = f"{model_name}/" if len(model_name) > 0 else ""
    for epoch in range(n_epochs):
        n_train_batches = len(train_loader)
        pbar = tf.keras.utils.Progbar(target=n_train_batches)
        print(f"Epoch {epoch + 1}/{n_epochs}")
        # training
        train_loss_sum = 0.0
        model.train()
        for train_idx, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            train_batch_loss = get_batch_loss_func(train_batch, epoch)
            train_batch_loss.backward(retain_graph=True)
            optimizer.step()
            train_loss_sum += train_batch_loss.item()
            # loss metrics are the average training losses from the start of the epoch
            pbar.update(train_idx, values=[("loss", train_loss_sum / (train_idx + 1))])
        train_loss_mean = train_loss_sum / n_train_batches
        writer.add_scalar(f"{model_name_str}train_loss", train_loss_mean, epoch)
        # validation
        if val_loader is not None:
            n_val_batches = len(val_loader)
            model.eval()
            with torch.no_grad():  # less memory usage.
                val_loss_sum = 0.0
                for val_batch in val_loader:
                    val_batch_loss = get_batch_loss_func(val_batch, epoch)
                    val_loss_sum += val_batch_loss.item()
            val_loss_mean = val_loss_sum / n_val_batches
            pbar.update(n_train_batches, values=[("val_loss", val_loss_mean)])
            writer.add_scalar(f"{model_name_str}val_loss", val_loss_mean, epoch)
            if early_stopping_target == "val_loss":
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
        scheduler.step()
    writer.flush()
    writer.close()


class DeepSad(BaseDetector):
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
        "deep_sad_",
        "remove_anomalies_",
        "normal_as_unlabeled_",
        "oversample_anomalies_",
        "n_ano_per_normal_",
        "arch_hps_",
        "pretrain_opt_hps_",
        "pretrain_train_hps_",
        "pretrain_callbacks_hps_",
        "opt_hps_",
        "train_hps_",
        "callbacks_hps_",
        "n_features_",
        "n_train_samples_",
        "random_state_",
    ]
    window_scorer_param_names = []
    online_scorer_param_names = []
    online_detector_param_names = []

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.pretrain_model_file_name = "autoencoder"
        self.model_file_name = "deep_sad"
        self.deep_sad_ = None
        self.normal_as_unlabeled_ = None
        self.remove_anomalies_ = None
        self.oversample_anomalies_ = None
        self.n_ano_per_normal_ = None
        self.arch_hps_ = None
        self.pretrain_opt_hps_ = None
        self.pretrain_train_hps_ = None
        self.pretrain_callbacks_hps_ = None
        self.opt_hps_ = None
        self.train_hps_ = None
        self.callbacks_hps_ = None
        self.n_features_ = None
        self.n_train_samples_ = None
        self.random_state_ = None

    def set_window_model_params(
        self,
        normal_as_unlabeled: bool = True,
        remove_anomalies: bool = False,
        oversample_anomalies: bool = True,
        n_ano_per_normal: float = 1.0,
        network: str = "mlp",
        enc_conv1d_filters: Union[int, str] = "",
        enc_conv1d_kernel_sizes: Union[int, str] = "",
        enc_conv1d_strides: Union[int, str] = "",
        conv1d_batch_norm: bool = True,
        hidden_dims: Union[int, str] = "100 50",
        rep_dim: int = 128,
        eta: float = 1.0,
        ae_out_act: str = "sigmoid",
        pretrain_optimizer: str = "adamw",
        pretrain_adamw_weight_decay: float = 1e-6,
        pretrain_learning_rate: float = 1e-4,
        pretrain_lr_milestones: Union[int, str] = "50",
        pretrain_batch_size: int = 200,
        pretrain_n_epochs: int = 150,
        pretrain_early_stopping_target: str = "val_loss",
        pretrain_early_stopping_patience: int = 100,
        optimizer: str = "adamw",
        adamw_weight_decay: float = 1e-6,
        learning_rate: float = 1e-4,
        lr_milestones: Union[int, str] = "50",
        batch_size: int = 200,
        n_epochs: int = 150,
        early_stopping_target: str = "val_loss",
        early_stopping_patience: int = 100,
        random_state: int = 0,
        fix_weights_init: bool = True,
    ):
        """Sets hyperparameters relevant to the DeepSAD model."""
        check_value_in_choices(network, "network", ["mlp", "rec"])
        check_value_in_choices(ae_out_act, "ae_out_act", ["linear", "sigmoid"])
        if ae_out_act == "sigmoid" and network == "rec":
            raise NotImplementedError(
                "Sigmoid output activation is not supported yet for recurrent networks."
            )
        # turn list parameters to actual lists
        enc_conv1d_filters = get_parsed_list_argument(enc_conv1d_filters)
        enc_conv1d_kernel_sizes = get_parsed_list_argument(enc_conv1d_kernel_sizes)
        enc_conv1d_strides = get_parsed_list_argument(enc_conv1d_strides)
        hidden_dims = get_parsed_list_argument(hidden_dims)
        pretrain_lr_milestones = get_parsed_list_argument(pretrain_lr_milestones)
        lr_milestones = get_parsed_list_argument(lr_milestones)
        opt_choices = ["nag", "rmsprop", "adam", "nadam", "adadelta", "adamw"]
        for k, v in zip(
            ["optimizer", "pretrain_optimizer"], [optimizer, pretrain_optimizer]
        ):
            check_value_in_choices(v, k, opt_choices)
        for k, v in zip(
            ["early_stopping_target", "pretrain_early_stopping_target"],
            [early_stopping_target, pretrain_early_stopping_target],
        ):
            check_value_in_choices(v, k, ["loss", "val_loss"])
        self.normal_as_unlabeled_ = normal_as_unlabeled
        self.remove_anomalies_ = remove_anomalies
        self.oversample_anomalies_ = oversample_anomalies
        self.n_ano_per_normal_ = n_ano_per_normal
        self.arch_hps_ = {
            "network": network,
            "enc_conv1d_filters": enc_conv1d_filters,
            "enc_conv1d_kernel_sizes": enc_conv1d_kernel_sizes,
            "enc_conv1d_strides": enc_conv1d_strides,
            "conv1d_batch_norm": conv1d_batch_norm,
            "hidden_dims": hidden_dims,
            "rep_dim": rep_dim,
            "ae_out_act": ae_out_act,
            "eta": eta,
            "fix_weights_init": fix_weights_init,
        }
        self.pretrain_opt_hps_ = {
            "optimizer": pretrain_optimizer,
            "learning_rate": pretrain_learning_rate,
        }
        if pretrain_optimizer == "adamw":
            self.pretrain_opt_hps_["adamw_weight_decay"] = pretrain_adamw_weight_decay
        self.opt_hps_ = {"optimizer": optimizer, "learning_rate": learning_rate}
        if optimizer == "adamw":
            self.opt_hps_["adamw_weight_decay"] = adamw_weight_decay
        self.pretrain_train_hps_ = {
            "lr_milestones": pretrain_lr_milestones,
            "batch_size": pretrain_batch_size,
            "n_epochs": pretrain_n_epochs,
        }
        self.train_hps_ = {
            "lr_milestones": lr_milestones,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        }
        self.pretrain_callbacks_hps_ = {
            "early_stopping_target": pretrain_early_stopping_target,
            "early_stopping_patience": pretrain_early_stopping_patience,
        }
        self.callbacks_hps_ = {
            "early_stopping_target": early_stopping_target,
            "early_stopping_patience": early_stopping_patience,
        }
        self.random_state_ = random_state

    def _fit_window_model(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        train_info=None,
        val_info=None,
    ) -> None:
        check_is_not_none(y_train, "y_train")
        if X_val is not None:
            check_is_not_none(y_val, "y_val")
        if self.callbacks_hps_["early_stopping_target"] == "val_loss" and X_val is None:
            raise ValueError(
                "Validation data must be provided when specifying early stopping on validation loss."
            )
        if self.remove_anomalies_:
            logging.info("Memory used before removing anomalies:")
            log_windows_memory(X_train, X_val)
            X_train, y_train, X_val, y_val, train_info, val_info = get_normal_windows(
                X_train, y_train, X_val, y_val, train_info, val_info
            )
            logging.info("Memory used after removing anomalies:")
            log_windows_memory(X_train, X_val)
        else:
            logging.info("Memory used before balancing:")
            log_windows_memory(X_train, X_val)
            if self.oversample_anomalies_:
                X_train, y_train = get_balanced_samples(
                    X_train,
                    y_train,
                    n_ano_per_normal=self.n_ano_per_normal_,
                    random_seed=self.random_state_,
                )
                if X_val is not None:
                    X_val, y_val = get_balanced_samples(
                        X_val,
                        y_val,
                        n_ano_per_normal=self.n_ano_per_normal_,
                        random_seed=self.random_state_,
                    )
            else:
                # always balance anomaly types
                X_train, y_train = get_balanced_ano_types(
                    X_train, y_train.astype(np.float32), self.random_state_
                )
                logging.info(
                    f"Balanced train labels:\n{pd.Series(y_train).value_counts()}"
                )
                if X_val is not None:
                    X_val, y_val = get_balanced_ano_types(
                        X_val, y_val.astype(np.float32), self.random_state_
                    )
                    logging.info(
                        f"Balanced val labels:\n{pd.Series(y_val).value_counts()}"
                    )
            logging.info("Memory used after balancing:")
            log_windows_memory(X_train, X_val)
        n_train_samples, window_size, n_features = X_train.shape
        self.n_train_samples_ = n_train_samples
        self.n_features_ = n_features
        # we can already save the detector, as the model will be saved/loaded separately
        save_files(self.window_model_path, {"detector": self}, "pickle")

        # format data for DeepSAD
        if self.arch_hps_["network"] == "mlp":
            # flatten windows here for MLP architectures
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)
        # 1 is for known normal, 0 for unlabeled and -1 for known anomalies
        normal_label = 0.0 if self.normal_as_unlabeled_ else 1.0
        y_train[y_train == 0.0] = normal_label
        y_train[y_train > 0.0] = -1.0
        if y_val is not None:
            y_val[y_val == 0.0] = normal_label
            y_val[y_val > 0.0] = -1.0

        # set up seed
        if self.random_state_ != -1:
            random.seed(self.random_state_)
            np.random.seed(self.random_state_)
            torch.manual_seed(self.random_state_)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state_)
                torch.backends.cudnn.deterministic = True

        # set up data
        train_dataset = DeepSADDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_hps_["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        if X_val is not None:
            val_dataset = DeepSADDataset(X_val, y_val)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.train_hps_["batch_size"],
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = None

        # set up model
        self.deep_sad_ = TorchDeepSAD(
            self.window_size_, self.n_features_, **self.arch_hps_
        )

        # pretraining
        self.pretrain_optimizer_ = get_optimizer(
            self.deep_sad_.pretrain_model.parameters(), **self.pretrain_opt_hps_
        )
        self.pretrain_scheduler_ = torch.optim.lr_scheduler.MultiStepLR(
            self.pretrain_optimizer_,
            milestones=self.pretrain_train_hps_["lr_milestones"],
            gamma=0.1,
        )
        pretrain_early_stopper = EarlyStopper(
            patience=self.pretrain_callbacks_hps_["early_stopping_patience"],
            min_delta=0,
        )
        pretrain_checkpointer = Checkpointer(
            self.deep_sad_.pretrain_model,
            self.pretrain_optimizer_,
            self.pretrain_scheduler_,
            os.path.join(self.window_model_path, self.pretrain_model_file_name),
        )
        logging_path = os.path.join(
            self.window_model_path, time.strftime("%Y_%m_%d-%H_%M_%S")
        )
        pretrain_writer = SummaryWriter(log_dir=logging_path)
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=self.deep_sad_.pretrain_model,
            get_batch_loss_func=self.deep_sad_.get_pretrain_batch_loss,
            optimizer=self.pretrain_optimizer_,
            scheduler=self.pretrain_scheduler_,
            early_stopper=pretrain_early_stopper,
            checkpointer=pretrain_checkpointer,
            writer=pretrain_writer,
            n_epochs=self.pretrain_train_hps_["n_epochs"],
            early_stopping_target=self.pretrain_callbacks_hps_["early_stopping_target"],
            model_name=self.pretrain_model_file_name,
        )
        self.deep_sad_.init_network_weights_from_pretraining()

        # training
        self.optimizer_ = get_optimizer(
            self.deep_sad_.model.parameters(), **self.opt_hps_
        )
        self.scheduler_ = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_,
            milestones=self.train_hps_["lr_milestones"],
            gamma=0.1,
        )
        early_stopper = EarlyStopper(
            patience=self.callbacks_hps_["early_stopping_patience"],
            min_delta=0,
        )
        checkpointer = Checkpointer(
            self.deep_sad_.model,
            self.optimizer_,
            self.scheduler_,
            os.path.join(self.window_model_path, self.model_file_name),
        )
        writer = SummaryWriter(log_dir=logging_path)
        c = self.deep_sad_.init_center_c(train_loader, eps=0.1)
        torch.save(c, os.path.join(self.window_model_path, "c.pt"))
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=self.deep_sad_.model,
            get_batch_loss_func=self.deep_sad_.get_batch_loss,
            optimizer=self.optimizer_,
            scheduler=self.scheduler_,
            early_stopper=early_stopper,
            checkpointer=checkpointer,
            writer=writer,
            n_epochs=self.train_hps_["n_epochs"],
            early_stopping_target=self.callbacks_hps_["early_stopping_target"],
            model_name=self.model_file_name,
        )

    def _predict_window_model(self, X):
        pass

    def _predict_window_scorer(self, X):
        """`X` should correspond to sequential windows in a given sequence."""
        if self.arch_hps_["network"] == "mlp":
            # flatten here for MLP architectures
            X = X.reshape(X.shape[0], -1)
        dataloader = DataLoader(X, batch_size=256, shuffle=False, drop_last=False)
        window_scores = self.deep_sad_.get_window_scores(dataloader)
        return window_scores

    def __getstate__(self):
        removed = ["deep_sad_"]
        return {k: v for k, v in self.__dict__.items() if k not in removed}

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            c = torch.load(os.path.join(self.window_model_path, "c.pt"))
        except OSError:  # works both if not found and permission denied
            c = torch.load(os.path.join(os.curdir, "c.pt"))
            self.window_model_path = os.curdir
        self.deep_sad_ = TorchDeepSAD(
            window_size=self.window_size_,
            n_features=self.n_features_,
            c=c,
            **self.arch_hps_,
        )
        self.optimizer_ = get_optimizer(
            self.deep_sad_.model.parameters(), **self.opt_hps_
        )
        self.scheduler_ = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_,
            milestones=self.train_hps_["lr_milestones"],
            gamma=0.1,
        )
        checkpoint = torch.load(
            os.path.join(self.window_model_path, self.model_file_name)
        )
        self.deep_sad_.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler_.load_state_dict(checkpoint["scheduler"])
        self.last_epoch_ = checkpoint["epoch"]
        self.last_checkpointed_loss_ = checkpoint["loss"]
