"""Adapted from https://github.com/TimeEval/TimeEval-algorithms/blob/main/mscred/mscred/correlation_matrix.py.

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

from typing import List

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset


def get_nearest_power_of_two(n_features: int, input_reduction_factor: int) -> int:
    next_bigger_power = int(2 ** np.ceil(np.log2(n_features)))
    if next_bigger_power < input_reduction_factor:
        next_bigger_power = input_reduction_factor
    return next_bigger_power


class CorrelationMatrices(Dataset):
    def __init__(
        self, X: NDArray, signature_lengths: List[int], input_reduction_factor: int
    ):
        """MSCRED dataset of shape `(n_samples, new_window_size, n_signatures, n_features, n_features)`.

        Args:
            X: input windows of shape `(n_windows, window_size, n_features)`.
            signature_lengths: signature lengths.
            input_reduction_factor: total reduction factor of the input after the encoder
              (e.g., 8 means both dimensions have been divided by 8).
        """
        self.X = torch.from_numpy(X)
        self.signature_lengths = signature_lengths
        self.input_reduction_factor = input_reduction_factor

    def _get_padded_matrix(self, cm: torch.Tensor) -> torch.Tensor:
        """Add padding so that the number of features is a power of 2."""
        n_dim = cm.shape[2]
        next_bigger_power = get_nearest_power_of_two(n_dim, self.input_reduction_factor)
        padded = torch.zeros(
            cm.shape[0], cm.shape[1], next_bigger_power, next_bigger_power
        )
        padded[:, :, :n_dim, :n_dim] = cm
        return padded

    def _get_correlation_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Turns sample into correlation matrix.

        `(window_size, n_feature)` -> `(new_window_size, n_signatures, n_features, n_features)`.

        The new window size is `(window_size - max_signature_length + 1)`: auto-correlations looking
        at `window_size` past data records at most: data coming from smaller signature lengths are pruned
        to have to same window size for all.
        """
        window_size, n_features = X.shape
        max_signature_length = max(self.signature_lengths)
        new_window_size = window_size - max_signature_length + 1
        correlation_matrix = torch.zeros(
            new_window_size, len(self.signature_lengths), n_features, n_features
        )
        for s, w in enumerate(self.signature_lengths):
            for t_, t in enumerate(range(max_signature_length, window_size)):
                correlation_matrix[t_, s] = (
                    torch.matmul(torch.t(X[t - w : t]), X[t - w : t]) / w
                )
        return self._get_padded_matrix(correlation_matrix)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item: int):
        return self._get_correlation_matrix(self.X[item])
