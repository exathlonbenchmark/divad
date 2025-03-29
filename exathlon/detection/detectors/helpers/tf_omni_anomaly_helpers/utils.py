from typing import Generator

import numpy as np

from data.helpers import get_aligned_shuffle, get_sliding_windows


def get_batch_generator(
    X: np.array, batch_size: int, shuffle: bool, include_remainder: bool
) -> Generator[np.array, None, None]:
    if shuffle:
        X = get_aligned_shuffle(X)
    batches = list(
        get_sliding_windows(
            X,
            window_size=batch_size,
            window_step=batch_size,
            include_remainder=False,
            dtype=np.float32,
            ranges_only=False,
        )
    )
    remainder_size = X.shape[0] % batch_size
    if remainder_size > 0 and include_remainder:
        batches.append(X[-remainder_size:])
    yield from batches


def save_z(z, filename="z"):
    """
    https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/utils.py.

    MIT License

    Copyright (c) 2021 NetManAIOps-OmniAnomaly

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

    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + "_" + str(i) + ".txt", "w") as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write("%f " % (z[j][i][k]))
                file.write("\n")
    i = z.shape[1] - 1
    with open(filename + "_" + str(i) + ".txt", "w") as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write("%f " % (z[j][i][k]))
            file.write("\n")
