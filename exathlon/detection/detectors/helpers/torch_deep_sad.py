"""
MIT License

Copyright (c) 2019 lukasruff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from numpy.typing import NDArray

from utils.guarding import check_value_in_choices
from detection.detectors.helpers.torch_helpers import get_and_set_device
from detection.detectors.helpers.torch_deep_sad_helpers import (
    MLP,
    MLP_Autoencoder,
    RecurrentEncoder,
    RecurrentAutoencoder,
)


class TorchDeepSAD:
    """`DeepSAD` wrapper with adapted training and inference steps."""

    def __init__(
        self,
        window_size: int = 20,
        n_features: int = 12,
        network: str = "mlp",
        enc_conv1d_filters: Optional[Sequence] = None,
        enc_conv1d_kernel_sizes: Optional[Sequence] = None,
        enc_conv1d_strides: Optional[Sequence] = None,
        conv1d_batch_norm: bool = True,
        hidden_dims: Optional[Sequence] = None,
        rep_dim: int = 128,
        ae_out_act: str = "sigmoid",
        eta: float = 1.0,
        fix_weights_init: bool = True,
        c: Optional[torch.Tensor] = None,
    ):
        check_value_in_choices(network, "network", ["mlp", "rec"])
        check_value_in_choices(ae_out_act, "ae_out_act", ["linear", "sigmoid"])
        if enc_conv1d_filters is None:
            enc_conv1d_filters = []
        if enc_conv1d_kernel_sizes is None:
            enc_conv1d_kernel_sizes = []
        if enc_conv1d_strides is None:
            enc_conv1d_strides = []
        if hidden_dims is None:
            hidden_dims = []
        self.window_size = window_size
        self.n_features = n_features
        if network == "mlp":
            self.pretrain_model = MLP_Autoencoder(
                x_dim=window_size * n_features,
                h_dims=hidden_dims,
                rep_dim=rep_dim,
                out_act=ae_out_act,
                bias=False,
            )
            self.model = MLP(
                x_dim=window_size * n_features,
                h_dims=hidden_dims,
                rep_dim=rep_dim,
                bias=False,
            )
        elif network == "rec":
            self.pretrain_model = RecurrentAutoencoder(
                window_size,
                n_features,
                enc_conv1d_filters=enc_conv1d_filters,
                enc_conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
                enc_conv1d_strides=enc_conv1d_strides,
                conv1d_batch_norm=conv1d_batch_norm,
                enc_h_dims=hidden_dims,
                rep_dim=rep_dim,
            )
            self.model = RecurrentEncoder(
                window_size,
                n_features,
                conv1d_filters=enc_conv1d_filters,
                conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
                conv1d_strides=enc_conv1d_strides,
                conv1d_batch_norm=conv1d_batch_norm,
                h_dims=hidden_dims,
                rep_dim=rep_dim,
            )
        self.pretrain_criterion = nn.MSELoss(reduction="none")
        self.device = get_and_set_device(self.pretrain_model)
        self.device = get_and_set_device(self.model)
        logging.info(f"Device: {self.device}")
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        self.eps = 1e-6
        self.fix_weights_init = fix_weights_init

    def get_pretrain_batch_loss(self, batch: Tensor, epoch: int) -> Tensor:
        inputs, _ = batch
        inputs = inputs.to(self.device)
        rec = self.pretrain_model(inputs)
        rec_loss = self.pretrain_criterion(rec, inputs)
        loss = torch.mean(rec_loss)
        return loss

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""
        if not self.fix_weights_init:
            # original implementation of the repository
            # Issue https://github.com/lukasruff/Deep-SAD-PyTorch/issues/9: ae_net_dict is empty
            net_dict = self.model.state_dict()
            ae_net_dict = self.pretrain_model.state_dict()
            # Filter out decoder network keys
            ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
            # Overwrite values in the existing state_dict
            net_dict.update(ae_net_dict)
            # Load the new state_dict
            self.model.load_state_dict(net_dict)
        else:
            # fixed by PR #11: https://github.com/lukasruff/Deep-SAD-PyTorch/pull/11
            ae_net_dict = self.pretrain_model.encoder.state_dict()
            self.model.load_state_dict(ae_net_dict)

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c
        return c

    def get_batch_loss(self, batch: Tensor, epoch: int) -> Tensor:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Update network parameters via backpropagation: forward + backward + optimize
        outputs = self.model(inputs)
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        losses = torch.where(
            targets == 0, dist, self.eta * ((dist + self.eps) ** targets.float())
        )
        loss = torch.mean(losses)
        return loss

    def get_window_scores(self, loader: DataLoader) -> NDArray[np.float32]:
        """Returns the anomaly scores for the windows in `loader` (which should contain only one batch).

        Args:
            loader: data loader providing the batch of windows to return anomaly scores for.

        Returns:
            The window anomaly scores.
        """
        window_scores = []
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.float().to(self.device)
                outputs = self.model(data)
                scores = torch.sum((outputs - self.c) ** 2, dim=1)
                window_scores += scores.cpu().data.numpy().tolist()
        window_scores = np.array(window_scores)
        return window_scores


class DeepSADDataset(Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
        """
        sample = self.data[index]
        target = int(self.targets[index])
        return (
            sample,
            target,
        )

    def __len__(self):
        return len(self.data)
