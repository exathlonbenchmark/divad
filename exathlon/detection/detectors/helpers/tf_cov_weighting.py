"""Adapted from:
- https://github.com/rickgroen/cov-weighting/blob/main/losses/covweighting_loss.py.
- https://github.com/rickgroen/cov-weighting/blob/main/losses/base_loss.py.

MIT License

Copyright (c) 2020 Rick

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
import tensorflow as tf

from utils.guarding import check_value_in_choices


class TensorFlowCoVWeighting:
    """CoVWeightingLoss.

    Args:
        num_losses: number of losses considered.
        mean_sort: either "full" or "decay".
        mean_decay_param: what decay to use with mean decay.
    """

    def __init__(
        self,
        num_losses: int = 2,
        mean_sort: str = "full",
        mean_decay_param: float = 1.0,
    ):
        check_value_in_choices(mean_sort, "mean_sort", ["full", "decay"])
        self.num_losses = num_losses
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if mean_sort == "decay" else False
        self.mean_decay_param = mean_decay_param
        self.current_iter = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        # Initialize weights.
        self.alphas = tf.Variable(
            tf.zeros((self.num_losses,), dtype=tf.float32), trainable=False
        )
        # Initialize all running statistics.
        self.running_mean_L = tf.Variable(
            tf.zeros((self.num_losses,), dtype=tf.float32), trainable=False
        )
        self.running_mean_l = tf.Variable(
            tf.zeros((self.num_losses,), dtype=tf.float32), trainable=False
        )
        self.running_S_l = tf.Variable(
            tf.zeros((self.num_losses,), dtype=tf.float32), trainable=False
        )
        self.running_std_l = tf.Variable(
            tf.zeros((self.num_losses,), dtype=tf.float32), trainable=False
        )

    @tf.function
    def get_combined_loss(
        self, unweighted_losses: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Returns the combined `unweighted_losses` using the Cov-Weighting method.

        Args:
            unweighted_losses: raw loss values of shape `(self.num_losses,)`.
            training: whether the model is in training or inference mode.

        Returns:
            The combined loss.
        """
        L = tf.identity(unweighted_losses)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not training:
            return tf.math.reduce_sum(L)
        # Increase the current iteration parameter.
        self.current_iter.assign_add(tf.cast(1.0, dtype=tf.float32))
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = tf.identity(L) if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/num_losses
        if self.current_iter <= 1.0:
            self.alphas.assign(
                tf.ones((self.num_losses,), dtype=tf.float32) / self.num_losses
            )
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas.assign(ls / tf.math.reduce_sum(ls))

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0.0:
            mean_param = 0.0
        elif self.current_iter > 0.0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = 1.0 - 1.0 / (self.current_iter + 1.0)

        # 2. Update the statistics for l
        x_l = tf.identity(l)
        new_mean_l = mean_param * self.running_mean_l + (1.0 - mean_param) * x_l
        self.running_S_l.assign_add((x_l - self.running_mean_l) * (x_l - new_mean_l))
        self.running_mean_l.assign(new_mean_l)

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1.0)
        self.running_std_l.assign(tf.math.sqrt(running_variance_l + 1e-8))

        # 3. Update the statistics for L
        x_L = tf.identity(L)
        self.running_mean_L.assign(
            mean_param * self.running_mean_L + (1.0 - mean_param) * x_L
        )

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [
            self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))
        ]
        loss = tf.math.reduce_sum(weighted_losses)
        return loss
