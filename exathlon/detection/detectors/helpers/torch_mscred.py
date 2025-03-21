"""Adapted from https://github.com/TimeEval/TimeEval-algorithms/blob/main/mscred/mscred/model.py.

MIT License

Copyright (c) 2020-2021 Phillip Wenig and Sebastian Schmidl

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

from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


from utils.guarding import check_is_not_empty
from detection.detectors.helpers.torch_helpers import get_and_set_device
from detection.detectors.helpers.torch_mscred_helpers.convlstm import (
    ConvLSTM,
    ConvLSTMAttention,
)
from detection.detectors.helpers.torch_mscred_helpers.correlation_matrix import (
    CorrelationMatrices,
)


def check_layer_params(block_name: str, layer_idx: int, w_in: int, k: int, s: int):
    """Checks layer parameters to ensure w_out = w_in // s after each Conv2D layer."""
    if w_in % s > 0:
        raise ValueError(
            f"Stride lengths should be multiples of corresponding input lengths: "
            f"found s={s} for w_in={w_in} in layer {layer_idx} of {block_name}."
        )
    if k < s:
        raise ValueError(
            f"Kernel sizes should be >= to corresponding stride lengths: "
            f"found k={k} for s={s} in layer {layer_idx} of {block_name}."
        )
    if (k + s) % 2 > 0:
        raise ValueError(
            f"Using symmetrical padding: kernel and stride lengths should have the same parity:"
            f"found k={k} and s={s} in layer {layer_idx} of {block_name}."
        )


def same_size_padding(w: int, k: int, s: int) -> int:
    return padding_from_size(w, w, k, s)


def padding_from_size(w_in: int, w_out: int, k: int, s: int) -> int:
    p = (w_out * s - s - w_in + k) / 2
    return int(np.ceil(p))


def output_size(w: int, k: int, s: int, p: int) -> int:
    return int(np.floor((w - k + 2 * p) / s + 1))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_dim: int,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        w_out = image_dim
        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            w_in = w_out
            w_out = w_out // s
            check_layer_params("encoder", i, w_in, k, s)
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else filters[i - 1],
                    out_channels=f,
                    kernel_size=k,
                    stride=s,
                    padding=padding_from_size(w_in, w_out, k, s),
                )
            )
        self.w_out = w_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        xs: List[torch.Tensor] = [x]
        for conv in self.convs:
            x = conv(x)
            xs.append(x)
        return tuple(xs)


class Temporal(nn.Module):
    def __init__(
        self,
        image_dim: int,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()
        self.lstms = nn.ModuleList([])
        w_out = image_dim
        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            w_out = w_out // s
            self.lstms.append(
                ConvLSTM(
                    in_channels=f,
                    out_channels=f,
                    kernel_size=k,
                    stride=s,
                    padding=same_size_padding(w_out, k, s),
                )
            )
        self.attention = ConvLSTMAttention()

    def forward(self, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output: List[torch.Tensor] = []
        for x, lstm in zip(xs, self.lstms):
            lstm(x)
            x = self.attention(lstm.outputs)
            output.append(x)

        return tuple(output)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        w_in: int,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()
        self.deconvs = nn.ModuleList([])
        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            w_out = s * w_in
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels if i == 0 else 2 * filters[i - 1],  # because of concat
                    f,
                    kernel_size=k,
                    stride=s,
                    padding=padding_from_size(w_out, w_in, k, s),
                )
            )
            w_in = w_out

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x_hat = self.deconvs[0](xs[-1])
        for deconv, x in zip(self.deconvs[1:], xs[-2::-1]):
            x_hat = deconv(torch.cat([x_hat, x], dim=1))
        return x_hat


class TorchMSCRED(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_signatures: int,
        filters: List[int] = None,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 2, 2, 2]
        if strides is None:
            strides = [1, 2, 2, 2]
        for v, s in zip(
            [filters, kernel_sizes, strides], ["filters", "kernel_sizes", "strides"]
        ):
            check_is_not_empty(v, s)

        self.n_features = n_features
        self.encoder = Encoder(
            in_channels=n_signatures,
            image_dim=self.n_features,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
        )
        self.temporal = Temporal(
            image_dim=self.n_features,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
        )
        self.decoder = Decoder(
            in_channels=filters[-1],
            w_in=self.encoder.w_out,
            filters=list(reversed(filters))[1:] + [n_signatures],
            kernel_sizes=list(reversed(kernel_sizes)),
            strides=list(reversed(strides)),
        )
        self.criterion = nn.MSELoss()
        for model in [self.encoder, self.temporal, self.decoder]:
            self.device = get_and_set_device(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, window_size, n_signatures, n_features, n_features)."""
        xs = []
        for t in range(x.shape[1]):
            xs.append([x_i.unsqueeze(1) for x_i in self.encoder(x[:, t])])
        xs = [torch.cat(x_i, dim=1).float().to(self.device) for x_i in zip(*xs)]
        xs = xs[1:] if len(xs) > 1 else xs

        xs = self.temporal(*xs)
        xs = [x_i.float().to(self.device) for x_i in xs]
        x = self.decoder(*xs)

        return x

    def get_batch_loss(self, batch: Tensor) -> Tensor:
        """Returns the average loss for the provided `batch`.

        Args:
            batch: the current batch.

        Returns:
            The average loss for the batch.
        """
        batch = batch.float().to(self.device)
        x_hat = self(batch)
        loss = self.criterion(
            torch.flatten(x_hat, start_dim=1),
            torch.flatten(batch[:, -1], start_dim=1),
        )
        return loss

    def get_window_scores(self, loader: DataLoader) -> NDArray[np.float32]:
        """Returns the anomaly scores for the windows in `loader`.

        Args:
            loader: data loader providing the batch of windows to return anomaly scores for.

        Returns:
            The window anomaly scores.
        """
        self.eval()
        losses = []
        for x in loader:
            x = x.float().to(self.device)
            x_hat = self(x)
            loss = F.mse_loss(
                torch.flatten(x_hat, start_dim=1),
                torch.flatten(x[:, -1], start_dim=1),
                reduction="none",
            )
            losses.append(loss)
        return torch.cat(losses, dim=0).mean(dim=1).detach().cpu().numpy()


def get_mscred_loader(
    X: NDArray[np.float32],
    signature_lengths: List[int],
    input_reduction_factor: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Returns the data loader to use for training the MSCRED model.

    Args:
        X: windows of shape `(n_windows, window_size, n_features)`.
        signature_lengths: signature matrix lengths (must all be below the window size used).
        input_reduction_factor: total reduction factor of the input after the encoder
          (e.g., 8 means both dimensions have been divided by 8).
        batch_size: batch size.
        shuffle: whether to shuffle the data at the end of every iteration (epoch).
        drop_last: whether to drop the last data elements or include them as a smaller batch.

    Returns:
        The data loader.
    """
    if max(signature_lengths) > X.shape[1]:
        raise ValueError(
            f"Found signature length {max(signature_lengths)} larger than window size {X.shape[1]}."
        )
    dataset = CorrelationMatrices(X, signature_lengths, input_reduction_factor)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return loader
