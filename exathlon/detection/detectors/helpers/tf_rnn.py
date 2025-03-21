"""RNN building and compilation module.

Gathers functions for building and compiling recurrent neural networks.
"""
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dense,
    Flatten,
    Reshape,
    Activation,
)

from detection.detectors.helpers.tf_helpers import (
    INIT_FOR_ACT,
    LayerBlock,
    get_optimizer,
)


def get_rnn(
    window_size: int = 40,
    n_features: int = 20,
    n_forward: int = 20,
    conv1d_filters: Optional[Sequence] = None,
    conv1d_kernel_sizes: Optional[Sequence] = None,
    conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = True,
    conv1d_batch_norm: bool = True,
    unit_type: str = "lstm",
    n_hidden_neurons: Optional[Sequence] = None,
    dropout: float = 0.0,
    rec_dropout: float = 0.0,
) -> tf.keras.Model:
    """Returns the RNN model matching the specified architecture hyperparameters.

    Args:
        window_size: input sequence length.
        n_forward: number of time steps to forecast ahead.
        n_features: number of features for each record in the sequence.
        conv1d_filters: number of filters for each Conv1D layer before the recurrent layers.
        conv1d_kernel_sizes: kernel sizes for each Conv1D layer before the recurrent layers.
        conv1d_strides: strides for each Conv1D layer before the recurrent layers.
        conv1d_pooling: whether to perform downsampling pooling rather than strided convolutions.
        conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
        unit_type: hidden unit type (simple RNN, LSTM or GRU).
        n_hidden_neurons: number of units for each recurrent layer before regression.
        dropout: dropout rate for feed-forward layers.
        rec_dropout: dropout rate for recurrent layers.

    Returns:
        The RNN keras model.
    """
    model = Sequential([InputLayer(input_shape=[window_size, n_features])], name="rnn")
    for i, (filters, kernel_size, strides) in enumerate(
        zip(conv1d_filters, conv1d_kernel_sizes, conv1d_strides)
    ):
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                strides=1 if conv1d_pooling else strides,
                kernel_initializer=INIT_FOR_ACT["relu"],
                use_bias=not conv1d_batch_norm,
            )
        )
        # normalize on the filters axis
        if conv1d_batch_norm:
            model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        if conv1d_pooling and strides > 1:  # typically i % 2 == 0
            model.add(MaxPooling1D(pool_size=strides, strides=strides, padding="same"))
    if len(n_hidden_neurons) > 0:
        model.add(
            LayerBlock(
                unit_type,
                layers_kwargs={
                    "units": n_hidden_neurons,
                    "dropout": [dropout],
                    "recurrent_dropout": [rec_dropout],
                    "return_sequences": [False],
                },
                name=f"{unit_type}_layer_block",
            )
        )
    else:
        # flatten the 2d output before the regression layer
        model.add(Flatten())
    model.add(Dense(n_forward * n_features, kernel_initializer=INIT_FOR_ACT["linear"]))
    model.add(Reshape([n_forward, n_features]))
    return model


def compile_rnn(
    model: tf.keras.Model,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    adamw_weight_decay: float = 0.0,
):
    """Compiles the provided RNN inplace using the specified optimization hyperparameters.

    Args:
        model: keras RNN model to compile.
        optimizer: optimization algorithm used for training the RNN.
        learning_rate: learning rate used by the optimization algorithm.
        adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
    """
    optimizer = get_optimizer(optimizer, learning_rate, adamw_weight_decay)
    model.compile(loss="mse", optimizer=optimizer)


def get_rnn_dataset(
    X: np.array,
    n_forward: int = 20,
    shuffling_buffer_prop: float = 1.0,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Returns the forecasting tf.data.Dataset corresponding to `X`.

    Args:
        X: samples of shape `(n_samples, window_size, n_features)`.
        n_forward: number of time steps to forecast ahead within the provided windows.
        shuffling_buffer_prop: proportion of training data to use as a shuffling buffer.
        batch_size: mini-batch size.

    Returns:
        Corresponding shuffled, batched and prefetched dataset.
    """

    def get_inputs_and_targets(window_batch: tf.Tensor):
        """Shape of `window_batch`: (batch_size, window_size, n_features)."""
        return window_batch[:, :-n_forward], window_batch[:, -n_forward:]

    buffer_size = int(shuffling_buffer_prop * X.shape[0])
    return (
        tf.data.Dataset.from_tensor_slices(X)
        .cache()
        .shuffle(buffer_size, seed=21)
        .batch(batch_size)
        .map(get_inputs_and_targets)
        .prefetch(tf.data.AUTOTUNE)
    )
