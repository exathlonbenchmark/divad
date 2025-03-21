"""Model building helpers module.

Gathers elements used for building and compiling keras models.
"""
import os
import six
import math
import time
from argparse import Namespace

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.initializers import GlorotUniform, HeUniform, LecunNormal
from tensorflow.keras.layers import (
    Layer,
    Concatenate,
    BatchNormalization,
    Activation,
    Dropout,
    Dense,
    SimpleRNN,
    LSTM,
    GRU,
    Conv1D,
    Conv1DTranspose,
    Cropping1D,
)
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop, Adadelta
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import (
    Callback,
    LambdaCallback,
    TensorBoard,
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN,
)
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable

from utils.guarding import check_value_in_choices, check_is_not_none
from data.helpers import get_aligned_list_values, get_list_value


# keras classes corresponding to string parameters (i.e. "parameter classes")
PC = {
    "layers": {
        "dense": Dense,
        "conv1d": Conv1D,
        "conv1d_transpose": Conv1DTranspose,
        "rnn": SimpleRNN,
        "lstm": LSTM,
        "gru": GRU,
    },
    # class and keyword arguments to overwrite
    "opt": {
        "nag": (SGD, {"momentum": 0.9, "nesterov": True, "name": "NAG"}),
        "adam": (Adam, dict()),
        "nadam": (Nadam, dict()),
        "adamw": (AdamW, {"beta_1": 0.9}),  # TODO: fix (cannot be modified)
        "rmsprop": (RMSprop, dict()),
        "adadelta": (Adadelta, dict()),
    },
    "loss": {"bce": BinaryCrossentropy, "mse": MeanSquaredError},
}
PC = Namespace(**PC)

# kernel initializers (and corresponding classes) to use for each activation function
INIT_FOR_ACT = {
    **{k: "glorot_uniform" for k in ["linear", "tanh", "sigmoid", "softmax"]},
    **{k: "he_uniform" for k in ["relu", "leaky_relu", "elu", "rrelu", "prelu"]},
    **{k: "lecun_normal" for k in ["selu"]},
}
INIT_CLASS_FOR_ACT = {
    **{k: GlorotUniform for k in ["linear", "tanh", "sigmoid", "softmax"]},
    **{k: HeUniform for k in ["relu", "leaky_relu", "elu", "rrelu", "prelu"]},
    **{k: LecunNormal for k in ["selu"]},
}


def sample_window_mse(y_true, y_pred):
    """Returns sample-wise MSE for tensors of shape `(batch_size, window_size, n_features)`."""
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])


def sample_euclidean_distance(x, y):
    """Returns sample-wise Euclidean distance between tensors of shape `(batch_size, n_features)`."""
    return tf.sqrt(
        tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=False), K.epsilon())
    )


def sample_squared_euclidean_distance(x, y):
    """Returns sample-wise squared Euclidean distance between tensors of shape `(batch_size, n_features)`."""
    return tf.reduce_sum(tf.square(x - y), axis=1, keepdims=False)


class PiecewiseConstantSchedule:
    """Piecewise-constant schedule.

    In this implementation, the learning rate is reduced by a constant factor at a given epoch frequency.

    Args:
        reduction_factor (float): learning rate reduction factor.
        reduction_freq (int): learning rate reduction frequency in epochs.
    """

    def __init__(self, reduction_factor=2.0, reduction_freq=10):
        self.reduction_factor = reduction_factor
        self.reduction_freq = reduction_freq

    def fn(self, epoch, lr):
        n_reductions = (epoch + 1) // self.reduction_freq
        return lr / (self.reduction_factor**n_reductions)


def get_optimizer(optimizer, learning_rate, adamw_weight_decay=None, clipnorm=None):
    """Returns optimizer object for the provided string and parameters.

    Args:
        optimizer (str): optimizer string (must be a key of `PC.opt`).
        learning_rate (float): learning rate used by the optimization algorithm.
        adamw_weight_decay (float): weight decay used for the AdamW optimizer if relevant.
        clipnorm (float): gradient norm clipping value.

    Returns:
        tf.keras.optimizers.Optimizer: the optimizer object.
    """
    opt_class, opt_kwargs = PC.opt[optimizer]
    if optimizer == "adamw":
        check_is_not_none(
            adamw_weight_decay, "weight decay should be provided for AdamW optimizer."
        )
        opt_kwargs = dict(opt_kwargs, **{"weight_decay": adamw_weight_decay})
    return opt_class(learning_rate=learning_rate, clipnorm=clipnorm, **opt_kwargs)


def set_tf_config(enable_mixed_precision=True, enable_xla=True):
    """Sets precision policy and xla compilation for all tensorflow models."""
    if enable_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(enable_xla)


def add_batch_loss_summary(batch, logs):
    """Adds batch loss to the default summary writer used by TensorBoard."""
    tf.summary.scalar("batch_loss", data=logs["loss"], step=batch)
    return batch


batch_loss_callback = LambdaCallback(on_batch_end=add_batch_loss_summary)


def get_callbacks(
    callbacks_type: str = "train",
    output_path: str = ".",
    model_file_name: str = "model",
    save_weights_only: bool = True,
    n_train_samples: int = None,
    batch_size: int = 32,
    n_epochs: int = 400,
    lr_scheduling: str = "none",
    lrs_pwc_red_factor: float = 2,
    lrs_pwc_red_freq: int = 10,
    lrs_oc_start_lr: float = 1e-4,
    lrs_oc_max_lr: float = 1e-3,
    lrs_oc_min_mom: float = 0.85,
    lrs_oc_max_mom: float = 0.95,
    early_stopping_target: str = "val_loss",
    early_stopping_patience: int = 20,
    include_tensorboard: bool = True,
    logging_path: str = None,
):
    check_value_in_choices(
        callbacks_type, "callbacks_type", ["train", "search", "searched"]
    )
    if logging_path is None:
        logging_path = os.path.join(output_path, time.strftime("%Y_%m_%d-%H_%M_%S"))
    if lr_scheduling == "one_cycle":
        check_is_not_none(n_train_samples, "n_train_samples")
        n_iterations = n_epochs * (n_train_samples // batch_size)
        start_lr = lrs_oc_start_lr
        max_lr = lrs_oc_max_lr
        min_mom = lrs_oc_min_mom
        max_mom = lrs_oc_max_mom
        scheduling_cbs = [
            ThreePhaseOneCycleScheduler(
                n_iterations, start_lr, max_lr, max_mom=max_mom, min_mom=min_mom
            )
        ]
        logging_freq = "batch"
        batch_cbs = [batch_loss_callback]
    else:
        if lr_scheduling == "pw_constant":
            scheduling_cbs = [
                LearningRateScheduler(
                    PiecewiseConstantSchedule(
                        lrs_pwc_red_factor,
                        lrs_pwc_red_freq,
                    ).fn
                )
            ]
        else:
            scheduling_cbs = []
        logging_freq = "epoch"
        batch_cbs = []
    if callbacks_type == "train":
        # regular training callbacks
        callbacks = [
            LearningRateLogger(logging_freq=logging_freq),
            MomentumLogger(logging_freq=logging_freq),
            *batch_cbs,  # `update_freq='batch'` of TensorBoard alone does not work
            *scheduling_cbs,
            EarlyStopping(
                monitor=early_stopping_target,
                patience=early_stopping_patience,
                mode="min",
                restore_best_weights=True,
            ),
            # main checkpoint (one per hyperparameters set)
            ModelCheckpoint(
                os.path.join(output_path, model_file_name),
                monitor=early_stopping_target,
                save_best_only=True,
                save_weights_only=save_weights_only,
            ),
            # backup checkpoint (one per run instance)
            ModelCheckpoint(
                os.path.join(logging_path, model_file_name),
                monitor=early_stopping_target,
                save_best_only=True,
                save_weights_only=save_weights_only,
            ),
        ]
        if include_tensorboard:
            callbacks.append(
                TensorBoard(
                    logging_path,
                    histogram_freq=1,
                    write_graph=True,
                    profile_batch="100, 120",
                )
            )
    elif callbacks_type == "search":
        # fixed callbacks used when searching hyperparameters
        callbacks = [
            # add a `TerminateOnNaN` callback to skip bad hyperparameter configurations
            TerminateOnNaN()
        ]
        if include_tensorboard:
            callbacks.append(
                TensorBoard(
                    output_path,  # TODO: was self.search_output_path before.
                    histogram_freq=1,
                    write_graph=True,
                    profile_batch="100, 120",
                )
            )
    else:
        # "searched" callbacks (updated at each hyperparameter search trial)
        callbacks = [
            LearningRateLogger(logging_freq=logging_freq),
            MomentumLogger(logging_freq=logging_freq),
            *batch_cbs,  # `update_freq='batch'` of TensorBoard alone does not work
            *scheduling_cbs,
            EarlyStopping(
                monitor=early_stopping_target,
                patience=early_stopping_patience,
                mode="min",
                restore_best_weights=True,
            ),
        ]
    return callbacks


class LearningRateLogger(Callback):
    """Learning logging class.

    Adds the current learning rate to the logs and current default summary writer used by TensorBoard.

    Args:
        logging_freq (str): learning rate logging frequency, either "epoch" or "batch".
    """

    def __init__(self, logging_freq="epoch"):
        super().__init__()
        check_value_in_choices(logging_freq, "logging_freq", ["epoch", "batch"])
        self.logging_freq = logging_freq
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        if self.logging_freq == "batch":
            logs = logs or dict()
            lr = K.eval(self.model.optimizer.lr)
            logs.update({"batch_lr": lr})
            tf.summary.scalar("batch_lr", data=lr, step=batch)

    def on_epoch_end(self, epoch, logs=None):
        if self.logging_freq == "epoch":
            logs = logs or dict()
            lr = K.eval(self.model.optimizer.lr)
            # "lr" to be consistent with potential learning rate schedulers
            logs.update({"lr": lr})
            tf.summary.scalar("lr", data=lr, step=epoch)


class MomentumLogger(Callback):
    """Momentum logging class.

    Adds the current momentum ("beta_1" for Adam-based optimizers) to the logs and current
    default summary writer used by TensorBoard.

    Args:
        logging_freq (str): momentum logging frequency, either "epoch" or "batch".
    """

    def __init__(self, logging_freq="epoch"):
        super().__init__()
        check_value_in_choices(logging_freq, "logging_freq", ["epoch", "batch"])
        self.logging_freq = logging_freq
        self._supports_tf_logs = True

    def _get_momentum(self):
        """Returns current momentum value and corresponding attribute name ((None, None) if not found)."""
        for attr_name in ["beta_1", "momentum"]:
            try:
                return K.eval(getattr(self.model.optimizer, attr_name)), attr_name
            except AttributeError:
                pass
        return None, None

    def on_batch_end(self, batch, logs=None):
        if self.logging_freq == "batch":
            logs = logs or dict()
            momentum, attr_name = self._get_momentum()
            if momentum is not None:
                logs.update({f"batch_{attr_name}": momentum})
                tf.summary.scalar(f"batch_{attr_name}", data=momentum, step=batch)

    def on_epoch_end(self, epoch, logs=None):
        if self.logging_freq == "epoch":
            logs = logs or dict()
            momentum, attr_name = self._get_momentum()
            if momentum is not None:
                logs.update({attr_name: momentum})
                tf.summary.scalar(attr_name, data=momentum, step=epoch)


class LRRangeTestCallback(Callback):
    """Learning rate range test callback.

    Sets the initial learning rate to a small value and multiply it by a constant factor at
    every iteration, up to a large value.

    Args:
        start_lr (float): start learning rate.
        end_lr (float): end learning rate.
        n_iterations (int): number of iterations to go from the start to the end learning rate.
    """

    def __init__(self, start_lr=1e-6, end_lr=1, n_iterations=100):
        super().__init__()
        self.start_lr = start_lr
        self.lr_multiplier = np.exp(np.log(end_lr / start_lr) / n_iterations)

    def on_batch_end(self, batch, logs=None):
        lr = (
            self.start_lr
            if batch == 0
            else self.lr_multiplier * K.get_value(self.model.optimizer.lr)
        )
        K.set_value(self.model.optimizer.lr, lr)


class ThreePhaseOneCycleScheduler(Callback):
    """1Cycle scheduling, as described in Leslie Smith's paper: https://arxiv.org/pdf/1803.09820.pdf.

    This is the original "three-phase" version. Another "two-phase" version exists and is described
    at https://fastai1.fast.ai/callbacks.one_cycle.html#OneCycleScheduler.

    Args:
        n_iterations (int): number of iterations to consider the cycle for.
        start_lr (float): starting learning rate.
        max_lr (float): maximum learning rate.
        last_lr_red_factor (int): in the third phase, the learning rate will decrease linearly
            to a last value that is `last_lr_red_factor` times less than the starting learning rate.
        third_phase_prop (float): proportion of `n_iterations` taken by the third phase.
        max_mom (float): maximum momentum ("beta_1" for Adam-based optimizers).
        min_mom (float): minimum momentum ("beta_1" for Adam-based optimizers).
    """

    def __init__(
        self,
        n_iterations,
        start_lr=1e-4,
        max_lr=1e-3,
        last_lr_red_factor=1000,
        third_phase_prop=0.1,
        max_mom=0.95,
        min_mom=0.85,
    ):
        self.n_iterations = n_iterations
        self.start_lr, self.max_lr = start_lr, max_lr
        self.n_third_phase_iterations = int(third_phase_prop * n_iterations) + 1
        # start iterations of the second and third phases
        self.second_start = (n_iterations - self.n_third_phase_iterations) // 2
        self.third_start = 2 * self.second_start
        self.last_lr = self.start_lr / last_lr_red_factor
        self.min_mom, self.max_mom = min_mom, max_mom
        self.cur_iteration = 0

    def _get_linear_interp(self, iter1, iter2, v1, v2):
        return (v2 - v1) * (self.cur_iteration - iter1) / (iter2 - iter1) + v1

    def _get_momentum_attr_name(self):
        for attr_name in ["beta_1", "momentum"]:
            try:
                getattr(self.model.optimizer, attr_name)
                return attr_name
            except AttributeError:
                pass

    def on_batch_begin(self, batch, logs):
        mom_attr, mom = self._get_momentum_attr_name(), None
        if self.cur_iteration < self.second_start:
            lr = self._get_linear_interp(
                0, self.second_start, self.start_lr, self.max_lr
            )
            if mom_attr is not None:
                mom = self._get_linear_interp(
                    0, self.second_start, self.max_mom, self.min_mom
                )
        elif self.cur_iteration < self.third_start:
            lr = self._get_linear_interp(
                self.second_start, self.third_start, self.max_lr, self.start_lr
            )
            if mom_attr is not None:
                mom = self._get_linear_interp(
                    self.second_start, self.third_start, self.min_mom, self.max_mom
                )
        else:
            lr = self._get_linear_interp(
                self.third_start, self.n_iterations, self.start_lr, self.last_lr
            )
            if mom_attr is not None:
                mom = self.max_mom
        K.set_value(self.model.optimizer.learning_rate, lr)
        if mom is not None:
            # TODO: throws error for AdamW: `beta_1` cannot be modified, and if set as
            #  `tf.Variable`, then error when serialized (i.e., when saved at checkpointing).
            try:
                K.set_value(getattr(self.model.optimizer, mom_attr), mom)
            except AttributeError:
                pass
        self.cur_iteration += 1


class EpochTimesCallback(Callback):
    """Keras callback recording each training epoch time.

    Args:
        epoch_times (list): reference to the list of epoch times.
            The times will be appended to this list inplace (emptying it first).
    """

    def __init__(self, epoch_times):
        super().__init__()
        self.epoch_times = epoch_times
        self.epoch_start = None

    def on_train_begin(self, logs=None):
        del self.epoch_times[:]

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start)


class MeanMetricWrapper(Mean):
    """MeanMetricWrapper class.

    When using tensorflow probability, metric wrappers automatically cast Distribution objects to
    Tensor objects, preventing us from using Distribution-specific methods like `log_prob`.

    This wrapper computes the mean value of the metric defined by `fn`, bypassing the troublesome casting.

    Source: https://github.com/tensorflow/probability/issues/742.
    """

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if is_tensor_or_variable(v) else v
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_shape_and_cropping(window_size, n_features, reduction_factor):
    """Returns the reduced window shape and cropping corresponding to `reduction_factor`.

    => Convolutional generators typically multiply spatial dimensions by a constant
    factor at specific layers. In case the original window dimensions are not multiples
    of their total reduction factor, windows will be generated as slightly larger and then
    cropped to the right dimensions.

    This function returns the shape to start from when generating windows from a
    `reduction_factor`-fold reduction of its spatial dimensions, as well as the final
    dimension-wise cropping to apply at the end of the generation process.

    Args:
        window_size (int): size of input samples in number of records.
        n_features (int): number of input features.
        reduction_factor (float): total reduction factor of spatial dimensions (e.g., 4.0).

    Returns:
        tuple, [[int, int], [int, int]]: reduced window shape and cropping to apply.
    """
    reduced_shape = math.ceil(window_size / reduction_factor), math.ceil(
        n_features / reduction_factor
    )
    cropping = [[0, 0], [0, 0]]
    for i, dim in enumerate([window_size, n_features]):
        # crop the excess in shape when multiplying back by the factor
        dim_crop = reduction_factor - (dim % reduction_factor)
        if dim_crop != reduction_factor:
            # crop more at the end if uneven (arbitrary)
            half_crop = dim_crop / 2.0
            cropping[i] = [math.floor(half_crop), math.ceil(half_crop)]
    return reduced_shape, cropping


class LayerBlock(Layer):
    """Utility layer defining a block of `layer_type` layers that will be called sequentially.

    All relevant keyword arguments for the layers can be passed through the `layers_kwargs` dict.
    All `layers_kwargs` values must be passed as lists (or sequences) whose lengths are either
    one or the maximum provided (i.e., `max_length`).

    `max_length` will define the number of layers in the block, and values of unit-length sequences
    will be repeated `max_length` times (i.e., for every layer in the block).

    If a Conv1DTranspose block is specified with a (total) reduction factor and a window size to match,
    its output will be cropped so as to match the window size, if this window size was not a multiple of
    the reduction factor.

    Args:
        layer_type (str): layer type to use in the block (as a key in `modeling.helpers.PC.layers`).
        layers_kwargs (dict|None): optional keyword arguments to pass to each layer.
        batch_normalization (list|bool): whether to apply batch normalization before each
            or all layer activation(s).
        dropout (float|bool): optional dropout rate(s) to apply to each or all layer(s).
        conv1d_transpose_red_factor (int|None): optional total window size reduction factor for
            a Conv1DTranspose block.
        conv1d_transpose_target_window_size (int|None): window size to match by cropping for
            a Conv1DTranspose block in case it was not a multiple of the total reduction factor
            (must be provided with `conv1d_transpose_red_factor`).
        **kwargs: optional keyword arguments of the Layer object.
    """

    def __init__(
        self,
        layer_type,
        layers_kwargs=None,
        batch_normalization=False,
        dropout=None,
        conv1d_transpose_red_factor=None,
        conv1d_transpose_target_window_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # set arguments as attributes to recover the layer config
        self.f_args_dict = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "kwargs"]
        }
        assert (
            layer_type in PC.layers
        ), "the provided layer type must be a key in `modeling.helpers.PC.layers`"
        layer_class = PC.layers[layer_type]
        aligned_layers_kwargs = (
            dict() if layers_kwargs is None else get_aligned_list_values(layers_kwargs)
        )
        n_layers = len(list(aligned_layers_kwargs.values())[0])
        # always set `return_sequences=True` for intermediate recurrent layers if relevant
        if "return_sequences" in aligned_layers_kwargs and n_layers > 1:
            aligned_layers_kwargs["return_sequences"] = [
                *[True for _ in range(n_layers - 1)],
                aligned_layers_kwargs["return_sequences"][-1],
            ]
        batch_normalization = get_list_value(batch_normalization, n_layers)
        dropout = get_list_value(dropout, n_layers)
        layerwise_kwargs = [
            {k: [] for k in aligned_layers_kwargs} for _ in range(n_layers)
        ]
        for i in range(n_layers):
            for k, v in aligned_layers_kwargs.items():
                layerwise_kwargs[i][k] = v[i]
        # create layers with layerwise arguments
        self.layers = []
        for layer_kwargs, bn, d in zip(layerwise_kwargs, batch_normalization, dropout):
            self.layers.append(layer_class(**layer_kwargs))
            if bn:
                # apply batch normalization before the layer activation
                prev_activation = self.layers[-1].activation
                self.layers[-1].activation = "linear"
                self.layers.append(BatchNormalization())
                self.layers.append(Activation(prev_activation))
            if d is not None:
                self.layers.append(Dropout(d))
        # window size cropping for Conv1DTranspose blocks
        if layer_type == "conv1d_transpose" and conv1d_transpose_red_factor is not None:
            a_t = "`conv1d_transpose_target_window_size` must be provided with `conv1d_transpose_red_factor`"
            assert conv1d_transpose_target_window_size is not None, a_t
            # crop Conv1DTranspose output to match the target size if not a multiple of the reduction factor
            _, cropping = get_shape_and_cropping(
                conv1d_transpose_target_window_size,
                layerwise_kwargs[-1]["filters"],
                conv1d_transpose_red_factor,
            )
            self.layers.append(Cropping1D(cropping[0]))

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.compute_output_shape(output_shape)
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(self.f_args_dict)
        return config


def add_batch_norm_layer(model, batch_normalization):
    """Adds a batch normalization layer to the provided Sequential model if `batch_normalization` is True."""
    if batch_normalization:
        model.add(BatchNormalization())


def add_dropout_layer(model: tf.keras.Model, rate: float):
    """Adds a dropout layer with the provided rate to the model if the rate is greater than zero."""
    if rate > 0:
        model.add(Dropout(rate))


def get_test_batch_size():
    """Returns the batch size to use at test time to prevent memory overflow.

    This is especially relevant when using the memory of a GPU, which is usually more
    limited.

    Returns:
        int: the batch size to use at test time.
    """
    n_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    return 128 if n_gpus > 0 else None
