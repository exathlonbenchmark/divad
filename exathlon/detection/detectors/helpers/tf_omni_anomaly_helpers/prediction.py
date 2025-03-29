"""https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/recurrent_distribution.py.

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
"""

import time

import numpy as np
import six
import tensorflow as tf
from tfsnippet.utils import (
    VarScopeObject,
    get_default_session_or_error,
    reopen_variable_scope,
)

from detection.detectors.helpers.tf_omni_anomaly_helpers.utils import (
    get_batch_generator,
)

__all__ = ["Predictor"]


class Predictor(VarScopeObject):
    """
    OmniAnomaly predictor.

    Args:
        model (OmniAnomaly): The :class:`OmniAnomaly` model instance.
        n_z (int or None): Number of `z` samples to take for each `x`.
            If :obj:`None`, one sample without explicit sampling dimension.
            (default 1024)
        batch_size (int): Size of each mini-batch for prediction.
            (default 32)
        feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            prediction. (default :obj:`None`)
        last_point_only (bool): Whether to obtain the reconstruction
            probability of only the last point in each window?
            (default :obj:`True`)
        name (str): Optional name of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(
        self,
        model,
        n_z=1024,
        batch_size=32,
        feed_dict=None,
        last_point_only=True,
        name=None,
        scope=None,
    ):
        super(Predictor, self).__init__(name=name, scope=scope)
        self._model = model
        self._n_z = n_z
        self._batch_size = batch_size
        if feed_dict is not None:
            self._feed_dict = dict(six.iteritems(feed_dict))
        else:
            self._feed_dict = {}
        self._last_point_only = last_point_only

        with reopen_variable_scope(self.variable_scope):
            # input placeholders
            self._input_x = tf.placeholder(
                dtype=tf.float32,
                shape=[None, model.window_length, model.x_dims],
                name="input_x",
            )
            self._input_y = tf.placeholder(
                dtype=tf.int32, shape=[None, model.window_length], name="input_y"
            )

            # outputs of interest
            self._score = self._score_without_y = None

    def _get_score_without_y(self):
        if self._score_without_y is None:
            with reopen_variable_scope(self.variable_scope), tf.name_scope(
                "score_without_y"
            ):
                self._score_without_y, self._q_net_z = self.model.get_score(
                    x=self._input_x,
                    n_z=self._n_z,
                    last_point_only=self._last_point_only,
                )
                # print ('\t_get_score_without_y ',type(self._q_net_z))
        return self._score_without_y, self._q_net_z

    @property
    def model(self):
        """
        Get the :class:`OmniAnomaly` model instance.

        Returns:
            OmniAnomaly: The :class:`OmniAnomaly` model instance.
        """
        return self._model

    def get_window_scores(self, X):
        """
        Get the `reconstruction probability` of specified KPI observations.

        The larger `reconstruction probability`, the less likely a point
        is anomaly.  You may take the negative of the score, if you want
        something to directly indicate the severity of anomaly.

        Args:
            X (np.ndarray): 2-D float32 windows.

        Returns:
            np.ndarray: The `reconstruction probability`,
                1-D array if `last_point_only` is :obj:`True`,
                or 2-D array if `last_point_only` is :obj:`False`.
        """
        with tf.name_scope("Predictor.get_window_scores"):
            sess = get_default_session_or_error()
            collector = []
            collector_z = []

            # run the prediction in mini-batches
            batch_gen = get_batch_generator(
                X,
                batch_size=self._batch_size,
                shuffle=False,
                include_remainder=True,
            )
            pred_time = []
            for b_x in batch_gen:
                start_iter_time = time.time()
                feed_dict = dict(six.iteritems(self._feed_dict))
                feed_dict[self._input_x] = b_x
                b_r, q_net_z = sess.run(
                    self._get_score_without_y(), feed_dict=feed_dict
                )
                collector.append(b_r)
                pred_time.append(time.time() - start_iter_time)
                collector_z.append(q_net_z)

            # merge the results of mini-batches
            result = np.concatenate(collector, axis=0)
            result_z = np.concatenate(collector_z, axis=0)
            return result, result_z, np.mean(pred_time)
