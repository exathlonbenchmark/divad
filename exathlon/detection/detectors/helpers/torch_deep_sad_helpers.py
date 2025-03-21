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

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the code layer or last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info("Trainable parameters: {}".format(params))
        self.logger.info(self)


class MLP(BaseNet):
    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [
            Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias)
            for i in range(1, len(neurons))
        ]
        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class MLP_Decoder(BaseNet):
    def __init__(
        self, x_dim, h_dims=[64, 128], rep_dim=32, out_act="sigmoid", bias=False
    ):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [
            Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias)
            for i in range(1, len(neurons))
        ]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.out_act = out_act
        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        if self.out_act == "sigmoid":
            x = self.sigmoid_activation(x)
        return x


class MLP_Autoencoder(BaseNet):
    def __init__(
        self, x_dim, h_dims=[128, 64], rep_dim=32, out_act="sigmoid", bias=False
    ):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(
            x_dim, list(reversed(h_dims)), rep_dim, out_act, bias
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))


class RecurrentEncoder(BaseNet):
    """Adapted from `CIFAR10_LeNet`."""

    def __init__(
        self,
        window_size,
        n_features,
        conv1d_filters=[32],
        conv1d_kernel_sizes=[5],
        conv1d_strides=[2],
        conv1d_batch_norm=True,
        h_dims=[],
        rep_dim=128,
    ):
        super().__init__()
        conv_block = []
        downsampled_size = window_size
        for i, (filters, kernel_size, strides) in enumerate(
            zip(conv1d_filters, conv1d_kernel_sizes, conv1d_strides)
        ):
            in_channels = n_features if i == 0 else conv1d_filters[i - 1]
            conv_block.append(
                nn.Conv1d(
                    in_channels,
                    filters,
                    kernel_size,
                    stride=1,
                    bias=False,
                    padding="same",
                )
            )
            if conv1d_batch_norm:
                conv_block.append(nn.BatchNorm1d(filters, eps=1e-04, affine=False))
            conv_block.append(nn.LeakyReLU())
            if strides > 1:
                conv_block.append(nn.MaxPool1d(strides, stride=strides))
            downsampled_size //= strides
        hidden_block = []
        hidden_input_size = (
            conv1d_filters[-1] if len(conv1d_filters) > 0 else n_features
        )
        for i, n in enumerate(h_dims):
            input_size = hidden_input_size if i == 0 else h_dims[i - 1]
            hidden_block.append(
                nn.GRU(input_size, n, 1, batch_first=True, bias=False, dropout=0.0)
            )
        latent_block = []
        if len(h_dims) > 0:
            latent_input_size = h_dims[-1]
        else:
            latent_block.append(nn.Flatten(start_dim=1, end_dim=-1))
            latent_input_size = downsampled_size * hidden_input_size
        latent_block.append(nn.Linear(latent_input_size, rep_dim, bias=False))
        self.window_size = window_size
        self.n_features = n_features
        self.downsampled_size = downsampled_size
        self.conv_block = nn.Sequential(*conv_block) if len(conv_block) > 0 else None
        self.hidden_block = (
            nn.Sequential(*hidden_block) if len(hidden_block) > 0 else None
        )
        self.latent_block = nn.Sequential(*latent_block)
        self.rep_dim = rep_dim

    def forward(self, x):
        if self.conv_block is not None:
            x = x.permute(0, 2, 1)
            x = self.conv_block(x)
            x = x.permute(0, 2, 1)
        if self.hidden_block is not None:
            x, _ = self.hidden_block(x)
            x = x[:, -1, :]  # return_sequence=False in keras
        # x = x.view(int(x.size(0)), -1)  # flatten already in the latent block
        x = self.latent_block(x)
        return x


class RecurrentDecoder(BaseNet):
    """Adapted from `CIFAR10_LeNet_Decoder`.

    TODO: add support for "sigmoid" last activation.
    """

    def __init__(
        self,
        window_size,
        downsampled_size,
        rep_dim=128,
        h_dims=[],
        conv1d_filters=[32],
        conv1d_kernel_sizes=[5],
        conv1d_strides=[2],
        conv1d_batch_norm=True,
    ):
        super().__init__()
        hidden_block = []
        for i, n in enumerate(h_dims):
            input_size = rep_dim if i == 0 else h_dims[i - 1]
            hidden_block.append(
                nn.GRU(input_size, n, 1, batch_first=True, bias=False, dropout=0.0)
            )
        conv_blocks = []
        conv_strides = []
        conv_input_channels = h_dims[-1] if len(h_dims) > 0 else rep_dim
        for i, (filters, kernel_size, strides) in enumerate(
            zip(conv1d_filters, conv1d_kernel_sizes, conv1d_strides)
        ):
            last_layer = i == len(conv1d_filters) - 1
            if last_layer:
                activation = "linear"
                batch_norm = False
            else:
                activation = "leaky_relu"
                batch_norm = conv1d_batch_norm
            in_channels = conv_input_channels if i == 0 else conv1d_filters[i - 1]
            conv_block = [
                nn.Conv1d(
                    in_channels,
                    filters,
                    kernel_size,
                    stride=1,
                    bias=False,
                    padding="same",
                )
            ]
            # only used in `CIFAR10_LeNet_Decoder`, so not added here.
            # nn.init.xavier_uniform_(
            #     conv_block[-1].weight,
            #     gain=nn.init.calculate_gain(activation),
            # )
            if batch_norm:
                conv_block.append(nn.BatchNorm1d(filters, eps=1e-04, affine=False))
            if activation == "leaky_relu":
                conv_block.append(nn.LeakyReLU())
            conv_blocks.append(nn.Sequential(*conv_block))
            conv_strides.append(strides)
        self.window_size = window_size
        self.downsampled_size = downsampled_size
        self.hidden_block = (
            nn.Sequential(*hidden_block) if len(hidden_block) > 0 else None
        )
        # `ModuleList` is needed to register a list of layers
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.conv_strides = conv_strides
        self.rep_dim = rep_dim

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.downsampled_size, 1)  # RepeatVector in keras
        if self.hidden_block is not None:
            x, _ = self.hidden_block(x)
        if len(self.conv_blocks) > 0:
            x = x.permute(0, 2, 1)
            for block, strides in zip(self.conv_blocks, self.conv_strides):
                x = block(x)
                if strides > 1:
                    x = F.interpolate(x, scale_factor=strides)
            x = x.permute(0, 2, 1)
        size_diff = x.shape[1] - self.window_size
        if size_diff > 0:
            x = x[:, :-size_diff, :]  # Cropping1D in keras
        elif size_diff < 0:
            x = nn.ConstantPad1d(-size_diff, 0.0)(x)  # Padding1D in keras
        return x


class RecurrentAutoencoder(BaseNet):
    """Adapted from `CIFAR10_LeNet_Autoencoder`."""

    def __init__(
        self,
        window_size,
        n_features,
        enc_conv1d_filters=[32],
        enc_conv1d_kernel_sizes=[5],
        enc_conv1d_strides=[2],
        conv1d_batch_norm=True,
        enc_h_dims=[],
        rep_dim=128,
    ):
        super().__init__()
        dec_conv1d_filters = list(reversed(enc_conv1d_filters))
        if len(dec_conv1d_filters) > 0:
            dec_conv1d_filters[-1] = n_features
        dec_conv1d_kernel_sizes = list(reversed(enc_conv1d_kernel_sizes))
        dec_conv1d_strides = list(reversed(enc_conv1d_strides))
        dec_h_dims = list(reversed(enc_h_dims))
        self.encoder = RecurrentEncoder(
            window_size,
            n_features,
            conv1d_filters=enc_conv1d_filters,
            conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
            conv1d_strides=enc_conv1d_strides,
            conv1d_batch_norm=conv1d_batch_norm,
            h_dims=enc_h_dims,
            rep_dim=rep_dim,
        )
        self.decoder = RecurrentDecoder(
            window_size,
            downsampled_size=self.encoder.downsampled_size,
            rep_dim=rep_dim,
            h_dims=dec_h_dims,
            conv1d_filters=dec_conv1d_filters,
            conv1d_kernel_sizes=dec_conv1d_kernel_sizes,
            conv1d_strides=dec_conv1d_strides,
            conv1d_batch_norm=conv1d_batch_norm,
        )
        self.rep_dim = rep_dim

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
