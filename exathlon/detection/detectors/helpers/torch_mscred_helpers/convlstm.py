"""https://github.com/TimeEval/TimeEval-algorithms/blob/main/mscred/mscred/convlstm.py.

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

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            3 * in_channels,
            4 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def _initial_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), torch.zeros_like(x)

    def forward(
        self, x: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = hx if hx else self._initial_hidden(x)
        A = self.conv(torch.cat([x, h, c], dim=1))
        (ai, af, ao, ag) = torch.split(A, self.out_channels, dim=1)

        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return o, (h, c)


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        num_layers: int = 1,
    ):
        super().__init__()

        self.out_channels = out_channels

        self.convlstms = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                for _ in range(num_layers)
            ]
        )
        self.outputs: Optional[torch.Tensor] = None

    def forward(
        self, X: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.outputs = torch.zeros(
            X.shape[0], X.shape[1], self.out_channels, X.shape[3], X.shape[4]
        )
        x = torch.zeros(X.shape[0], self.out_channels, X.shape[3], X.shape[4])
        for t in range(X.shape[1]):
            for cell in self.convlstms:
                x, hx = cell(X[:, t], hx)
                self.outputs[:, t] = x
        return x, hx


class ConvLSTMAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chi = x.shape[1]
        last = x[:, -1].reshape(x.shape[0], 1, -1)
        alphas = [
            (last.bmm(x[:, i].reshape(x.shape[0], -1, 1)) / chi).reshape(-1, 1)
            for i in range(chi)
        ]
        alpha = torch.cat(alphas, dim=1)
        alpha = torch.softmax(alpha, dim=1)
        result = (
            x.reshape(x.shape[0], -1, chi)
            .bmm(alpha.unsqueeze(2))
            .reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        )
        return result
