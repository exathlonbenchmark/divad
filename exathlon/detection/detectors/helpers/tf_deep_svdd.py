from typing import Optional, Sequence

from utils.guarding import check_value_in_choices

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    InputLayer,
    Flatten,
    Reshape,
    Activation,
    Dense,
    RepeatVector,
    TimeDistributed,
)

from utils.guarding import check_is_percentage
from detection.detectors.helpers.tf_helpers import (
    INIT_CLASS_FOR_ACT,
    LayerBlock,
    add_dropout_layer,
    get_optimizer,
    sample_squared_euclidean_distance,
)


def get_deep_svdd(
    window_size: int,
    n_features: int,
    type_: str = "dense",
    dense_hidden_activations: str = "relu",
    rec_unit_type: str = "lstm",
    rec_dropout: float = 0.0,
    conv1d_strides: Optional[Sequence] = None,
    n_hidden_neurons: Optional[Sequence] = None,
    output_dim: int = 10,
    output_activation: str = "relu",
    batch_normalization: bool = False,
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    random_seed: Optional[int] = None,
) -> tf.keras.Model:
    """Returns a deep SVDD network with the specified architecture hyperparameters.

    The deep SVDD architecture should satisfy that every layer uses:

    - No bias term.
    - An activation function that is either unbounded or zero-bounded.

    Args:
        window_size: size of input samples in number of records.
        n_features: number of input features.
        type_: type of deep SVDD to build (either "dense" or "rec").
        dense_hidden_activations: hidden layers activation for dense architectures.
        rec_unit_type: type of recurrent unit (either "lstm" or "gru") for recurrent architecture.
        rec_dropout: dropout rate for the hidden state of recurrent layers.
        conv1d_strides: strides for each Conv1D layer before the hidden layers, as
          an empty string for no layer, an integer for a single layer, or a string of
          space-separated integers for multiple layers.
        n_hidden_neurons: number of units for each hidden layer before the output, as
          an empty string for no layer, an integer for a single layer, or a string of
          space-separated integers for multiple layers.
        output_dim: number of units for the output layer.
        output_activation: output layer activation.
        batch_normalization: whether to apply batch normalization (before layer activations).
        input_dropout: dropout rate for the input layer.
        hidden_dropout: dropout rate for other feed-forward layers (except the output).
        random_seed: random seed for reproducibility (e.g., of the hypersphere
          centroid) across calls.

    Returns:
        The deep SVDD keras model.
    """
    if n_hidden_neurons is None:
        n_hidden_neurons = []
    if conv1d_strides is None:
        conv1d_strides = []
    check_value_in_choices(type_, "type_", ["dense", "rec"])
    for k, v in zip(
        ["input_dropout", "hidden_dropout"], [input_dropout, hidden_dropout]
    ):
        check_is_percentage(value=v, var_name=k)
    if type_ == "rec":
        if len(conv1d_strides) == 0 and batch_normalization:
            raise ValueError(
                "Batch normalization is only supported for non-purely recurrent architectures."
            )
        check_is_percentage(rec_dropout, "rec_dropout")

    # deep SVDD network
    deep_svdd = Sequential(
        [InputLayer(input_shape=(window_size, n_features))], name="deep_svdd"
    )
    add_dropout_layer(deep_svdd, input_dropout)

    if len(conv1d_strides) > 0:
        # convolution-based downsampling
        kernel_sizes = [[5]] + ([[3]] * (len(conv1d_strides) - 1))
        deep_svdd.add(
            LayerBlock(
                "conv1d",
                layers_kwargs={
                    "filters": [n_features],
                    "kernel_size": kernel_sizes,
                    "padding": ["same"],
                    "strides": conv1d_strides,
                    "use_bias": [False],
                    "activation": ["relu"],
                },
                batch_normalization=batch_normalization,
                name="conv1d_block",
            )
        )

    # type-dependent encoder architectures
    if len(n_hidden_neurons) > 0:
        if type_ == "dense":
            deep_svdd.add(Flatten())
            deep_svdd.add(
                LayerBlock(
                    "dense",
                    layers_kwargs={
                        "units": n_hidden_neurons,
                        "use_bias": [False],
                        "activation": [dense_hidden_activations],
                        "kernel_initializer": [
                            INIT_CLASS_FOR_ACT[dense_hidden_activations](
                                seed=random_seed
                            )
                        ],
                    },
                    batch_normalization=batch_normalization,
                    dropout=hidden_dropout,
                    name="dense_block",
                )
            )
        else:
            deep_svdd.add(
                LayerBlock(
                    rec_unit_type,
                    layers_kwargs={
                        "units": n_hidden_neurons,
                        "use_bias": [False],
                        "dropout": [hidden_dropout],
                        "recurrent_dropout": [rec_dropout],
                        "return_sequences": [True],
                    },
                    name=f"{rec_unit_type}_block",
                )
            )

    deep_svdd.add(
        Dense(
            output_dim,
            use_bias=False,
            activation=output_activation,
            kernel_initializer=INIT_CLASS_FOR_ACT[output_activation](seed=random_seed),
            name="output",
        )
    )
    return deep_svdd


class TensorFlowDeepSVDD(Model):
    def __init__(
        self,
        window_size: int,
        n_features: int,
        type_: str = "dense",
        dense_hidden_activations: str = "relu",
        rec_unit_type: str = "lstm",
        rec_dropout: float = 0.0,
        conv1d_strides: Optional[Sequence] = None,
        n_hidden_neurons: Optional[Sequence] = None,
        output_dim: int = 10,
        output_activation: str = "relu",
        batch_normalization: bool = False,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        loss_: str = "one_class",
        nu: float = 0.01,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        check_value_in_choices(loss_, "loss_", ["one_class", "soft_boundary"])
        self.model = get_deep_svdd(
            window_size=window_size,
            n_features=n_features,
            type_=type_,
            dense_hidden_activations=dense_hidden_activations,
            rec_unit_type=rec_unit_type,
            rec_dropout=rec_dropout,
            conv1d_strides=conv1d_strides,
            n_hidden_neurons=n_hidden_neurons,
            output_dim=output_dim,
            output_activation=output_activation,
            batch_normalization=batch_normalization,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            random_seed=random_seed,
        )
        self.centroid = None
        self.loss_ = loss_
        self.nu = nu if self.loss_ == "soft_boundary" else None

    def build(self, inputs_shape):
        self.model.build(inputs_shape)

    def set_centroid(self, centroid):
        self.centroid = centroid

    def call(self, inputs):
        z = self.model(inputs)
        if self.centroid is not None:  # not the initial pass
            if self.loss_ == "one_class":
                squared_distances = sample_squared_euclidean_distance(z, self.centroid)
                self.add_loss(squared_distances)
                self.add_metric(
                    squared_distances, name="squared_distance", aggregation="mean"
                )
            else:
                raise NotImplementedError(
                    "Soft boundary deep SVDD loss not supported yet."
                )
        return z

    def get_config(self):
        pass


def compile_deep_svdd(
    model: tf.keras.Model,
    optimizer: str = "adam",
    adamw_weight_decay: float = 0.0,
    learning_rate: float = 0.001,
):
    """Compiles the deep SVDD inplace using the specified optimization hyperparameters.

    Args:
        model: deep SVDD model to compile.
        optimizer: optimization algorithm used for training the network.
        adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
        learning_rate: learning rate used by the optimization algorithm.
    """
    optimizer = get_optimizer(optimizer, learning_rate, adamw_weight_decay)
    model.compile(loss=model.losses, optimizer=optimizer)


def get_deep_svdd_dataset(
    X: np.array,
    shuffling_buffer_prop: float = 1.0,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Returns the tf.data.Dataset corresponding to `X` for deep SVDD training/inference.

    Args:
        X: samples of shape `(n_samples, window_size, n_features)`.
        shuffling_buffer_prop: proportion of training data to use as a shuffling buffer.
        batch_size: mini-batch size.

    Returns:
        Corresponding shuffled, batched and prefetched dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(X).cache()  # no labels
    buffer_size = int(shuffling_buffer_prop * X.shape[0])
    dataset = dataset.shuffle(buffer_size, seed=21).batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)
