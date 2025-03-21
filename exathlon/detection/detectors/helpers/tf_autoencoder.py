from typing import Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Dropout,
    Flatten,
    Reshape,
    Activation,
    Dense,
    RepeatVector,
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Conv1DTranspose,
    UpSampling1D,
    Cropping1D,
    ZeroPadding1D,
)
from utils.guarding import check_value_in_choices, check_is_percentage
from detection.detectors.helpers.tf_helpers import (
    PC,
    INIT_FOR_ACT,
    LayerBlock,
    add_dropout_layer,
    get_optimizer,
)


def get_dense_autoencoder(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    hidden_activation: str = "relu",
    linear_latent_activation: bool = False,
    dec_last_activation: str = "linear",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    encoder_only: bool = False,
) -> Sequential:
    """Returns a dense autoencoder or encoder model.

    The provided `enc_n_hidden_neurons` neither account for the latent nor the output layer.
    """
    check_is_percentage(input_dropout, "input_dropout")
    check_is_percentage(hidden_dropout, "hidden_dropout")
    latent_activation = "linear" if linear_latent_activation else hidden_activation
    if enc_n_hidden_neurons is None:
        enc_n_hidden_neurons = []

    # encoder network
    encoder = Sequential(
        [InputLayer(input_shape=(window_size, n_features))], name="encoder"
    )
    add_dropout_layer(encoder, input_dropout)
    encoder.add(Flatten())
    if len(enc_n_hidden_neurons) > 0:
        encoder.add(
            LayerBlock(
                "dense",
                layers_kwargs={
                    "units": enc_n_hidden_neurons,
                    "activation": [hidden_activation],
                    "kernel_initializer": [INIT_FOR_ACT[hidden_activation]],
                },
                batch_normalization=False,
                dropout=hidden_dropout,
                name="encoding_dense_block",
            )
        )
    # latent layer
    encoder.add(
        Dense(
            latent_dim,
            activation=latent_activation,
            kernel_initializer=INIT_FOR_ACT[latent_activation],
            name="latent_layer",
        )
    )

    if encoder_only:
        return encoder

    # decoder network
    decoder = Sequential([InputLayer(input_shape=[latent_dim])], name="decoder")
    dec_n_hidden_neurons = list(reversed(enc_n_hidden_neurons))
    if len(dec_n_hidden_neurons) > 0:
        decoder.add(
            LayerBlock(
                "dense",
                layers_kwargs={
                    "units": dec_n_hidden_neurons,
                    "activation": [hidden_activation],
                    "kernel_initializer": [INIT_FOR_ACT[hidden_activation]],
                },
                batch_normalization=False,
                dropout=hidden_dropout,
                name="decoding_dense_block",
            )
        )
    # output layer
    decoder.add(
        Dense(
            window_size * n_features,
            activation=dec_last_activation,
            kernel_initializer=INIT_FOR_ACT[dec_last_activation],
        )
    )
    decoder.add(Reshape([window_size, n_features]))

    # autoencoder network
    return Sequential([encoder, decoder], name="autoencoder")


def get_conv_autoencoder(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    enc_conv1d_filters: Sequence = None,
    enc_conv1d_kernel_sizes: Sequence = None,
    enc_conv1d_strides: Sequence = None,
    conv1d_pooling: bool = False,
    conv1d_batch_norm: bool = False,
    add_dense_for_latent: bool = False,
    dense_layers_activation: str = "relu",
    linear_latent_activation: bool = False,
    dec_last_activation: str = "linear",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    encoder_only: bool = False,
) -> Sequential:
    """Returns a fully convolutional autoencoder or encoder model.

    The provided `enc_conv1d_filters`, `enc_conv1d_kernel_sizes` and `enc_conv1d_strides` also account for
    the output layer. If `linear_latent_activation` is False, they also account for the latent layer.
    """
    check_is_percentage(hidden_dropout, "hidden_dropout")
    check_is_percentage(input_dropout, "input_dropout")
    latent_activation = "linear" if linear_latent_activation else "relu"
    for text, v in zip(
        ["filters", "kernel sizes", "strides"],
        [enc_conv1d_filters, enc_conv1d_kernel_sizes, enc_conv1d_strides],
    ):
        if v is None or len(v) == 0:
            raise ValueError(
                f"Convolutional {text} must be provided for purely convolutional architectures."
            )

    # encoder network
    encoder = Sequential(
        [InputLayer(input_shape=(window_size, n_features))], name="encoder"
    )
    add_dropout_layer(encoder, input_dropout)
    for i, (filters, kernel_size, strides) in enumerate(
        zip(enc_conv1d_filters, enc_conv1d_kernel_sizes, enc_conv1d_strides)
    ):
        last_enc_layer = i == len(enc_conv1d_filters) - 1
        if last_enc_layer and not add_dense_for_latent:
            activation = latent_activation
            batch_norm = False
        else:
            activation = "relu"
            batch_norm = conv1d_batch_norm
        encoder.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                strides=1 if conv1d_pooling else strides,
                kernel_initializer=INIT_FOR_ACT[activation],
                use_bias=not batch_norm,
            )
        )
        # normalize on the filters axis
        if batch_norm:
            encoder.add(BatchNormalization(axis=-1))
        encoder.add(Activation(activation))
        if conv1d_pooling and strides > 1:  # typically i % 2 == 0
            encoder.add(
                MaxPooling1D(pool_size=strides, strides=strides, padding="same")
            )
    downsampled_size = encoder.layers[-1].output.shape[1]

    if add_dense_for_latent:
        encoder.add(Flatten())
        encoder.add(
            Dense(
                latent_dim,
                activation=latent_activation,
                kernel_initializer=INIT_FOR_ACT[latent_activation],
                name="latent_layer",
            )
        )

    if encoder_only:
        return encoder

    # decoder network
    decoder_input_shape = encoder.layers[-1].output.shape[1:]
    decoder = Sequential([InputLayer(input_shape=decoder_input_shape)], name="decoder")
    if add_dense_for_latent:
        # construct input of the Conv1DTranspose block
        last_enc_n_filters = enc_conv1d_filters[-1]
        decoder.add(
            Dense(
                downsampled_size * last_enc_n_filters,
                activation=dense_layers_activation,
                kernel_initializer=INIT_FOR_ACT[dense_layers_activation],
            )
        )
        add_dropout_layer(decoder, hidden_dropout)
        decoder.add(Reshape([downsampled_size, last_enc_n_filters]))

    dec_conv1d_filters = list(reversed(enc_conv1d_filters))
    dec_conv1d_kernel_sizes = list(reversed(enc_conv1d_kernel_sizes))
    dec_conv1d_strides = list(reversed(enc_conv1d_strides))
    if len(dec_conv1d_filters) > 0:
        dec_conv1d_filters[-1] = n_features
    for i, (filters, kernel_size, strides) in enumerate(
        zip(dec_conv1d_filters, dec_conv1d_kernel_sizes, dec_conv1d_strides)
    ):
        output_layer = i == len(dec_conv1d_filters) - 1
        if output_layer:
            activation = dec_last_activation
            batch_norm = False
        else:
            activation = "relu"
            batch_norm = conv1d_batch_norm
        decoder.add(
            Conv1DTranspose(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                # always use strided convolutions for the output layer upsampling
                strides=strides if not conv1d_pooling or output_layer else 1,
                kernel_initializer=INIT_FOR_ACT[activation],
                use_bias=not batch_norm,
            )
        )
        if batch_norm:
            # normalize on the filters axis
            decoder.add(BatchNormalization(axis=-1))
        decoder.add(Activation(activation))
        if not output_layer and conv1d_pooling and strides > 1:  # typically i % 2 == 0
            decoder.add(UpSampling1D(size=strides))
    upsampled_size = decoder.layers[-1].output.shape[1]
    # crop or pad if the upsampled window size does not match the original window size
    size_diff = upsampled_size - window_size
    if size_diff > 0:
        decoder.add(Cropping1D((0, size_diff)))
    elif size_diff < 0:
        decoder.add(ZeroPadding1D((0, -size_diff)))

    # autoencoder network
    return Sequential([encoder, decoder], name="autoencoder")


def get_rec_autoencoder(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    enc_conv1d_filters: Optional[Sequence] = None,
    enc_conv1d_kernel_sizes: Optional[Sequence] = None,
    enc_conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = False,
    conv1d_batch_norm: bool = False,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    latent_type: str = "dense",
    linear_latent_activation: bool = False,
    dec_last_activation: str = "linear",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    rec_dropout: float = 0.0,
    encoder_only: bool = False,
):
    """Returns a recurrent autoencoder or encoder model, with optional convolution-based downsampling.

    The provided `enc_n_hidden_neurons` accounts for the latent layer if `latent_type` is "rec". Otherwise,
    it does not account for the latent layer. The last Conv1D parameters for the decoder are a reversed
    version of the encoder's, except the last number of filters which is always set to the number of features.
    If provided, convolution layers therefore always account for the output layer. If not provided,
    `dec_n_hidden_neurons` does not account for the output layer, which is set to a time-distributed dense
    layer with the right number of units.
    """
    check_value_in_choices(latent_type, "latent_type", ["dense", "rec"])
    check_is_percentage(input_dropout, "input_dropout")
    check_is_percentage(hidden_dropout, "hidden_dropout")
    check_is_percentage(rec_dropout, "rec_dropout")
    if enc_n_hidden_neurons is None or len(enc_n_hidden_neurons) == 0:
        raise ValueError(
            "Call `get_conv_autoencoder()` for a purely convolutional autoencoder."
        )
    if linear_latent_activation:
        latent_activation = "linear"
    elif latent_type == "rec":
        latent_activation = activation_rec
    else:
        latent_activation = "relu"
    if enc_conv1d_filters is None:
        enc_conv1d_filters = []
    if enc_conv1d_kernel_sizes is None:
        enc_conv1d_kernel_sizes = []
    else:
        enc_conv1d_kernel_sizes = [[s] for s in enc_conv1d_kernel_sizes]
    if enc_conv1d_strides is None:
        enc_conv1d_strides = []

    # encoder network
    encoder = Sequential(
        [InputLayer(input_shape=(window_size, n_features))], name="encoder"
    )
    add_dropout_layer(encoder, input_dropout)
    for i, (filters, kernel_size, strides) in enumerate(
        zip(enc_conv1d_filters, enc_conv1d_kernel_sizes, enc_conv1d_strides)
    ):
        encoder.add(
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
            encoder.add(BatchNormalization(axis=-1))
        encoder.add(Activation("relu"))
        if conv1d_pooling and strides > 1:  # typically i % 2 == 0
            encoder.add(
                MaxPooling1D(pool_size=strides, strides=strides, padding="same")
            )
    if len(enc_conv1d_filters) > 0:
        downsampled_size = encoder.layers[-1].output.shape[1]
    else:
        downsampled_size = window_size

    # type-dependent encoder architectures
    rec_latent = latent_type == "rec"
    if rec_latent:
        if enc_n_hidden_neurons[-1] != latent_dim:
            raise ValueError(
                f"For a recurrent latent layer, the encoder block should end with "
                f"latent_dim={latent_dim} units. Received {enc_n_hidden_neurons[-1]}."
            )
        n_before_latent = len(enc_n_hidden_neurons) - 1
        activations = n_before_latent * [activation_rec] + [latent_activation]
        dropouts = n_before_latent * [hidden_dropout] + [0.0]
        rec_dropouts = n_before_latent * [rec_dropout] + [0.0]
    else:
        activations = [activation_rec]
        dropouts = [hidden_dropout]
        rec_dropouts = [rec_dropout]
    encoder.add(
        LayerBlock(
            rec_unit_type,
            layers_kwargs={
                "units": enc_n_hidden_neurons,
                "dropout": dropouts,
                "activation": activations,
                "recurrent_dropout": rec_dropouts,
                "return_sequences": [False],
            },
            name=f"encoding_{rec_unit_type}_block",
        )
    )
    # add a dense layer if specified (else the latent activation is the last recurrent state)
    if not rec_latent:
        encoder.add(
            Dense(
                latent_dim,
                activation=latent_activation,
                kernel_initializer=INIT_FOR_ACT[latent_activation],
                name="latent_layer",
            )
        )

    if encoder_only:
        return encoder

    # decoder network
    decoder = Sequential([InputLayer(input_shape=[latent_dim])], name="decoder")

    # type-dependent decoder architectures
    dec_n_hidden_neurons = list(reversed(enc_n_hidden_neurons))
    decoder.add(RepeatVector(downsampled_size, input_shape=[latent_dim]))
    decoder.add(
        LayerBlock(
            rec_unit_type,
            layers_kwargs={
                "units": dec_n_hidden_neurons,
                "dropout": [hidden_dropout],
                "activation": [activation_rec],
                "recurrent_dropout": [rec_dropout],
                "return_sequences": [True],
            },
            name=f"decoding_{rec_unit_type}_block",
        )
    )
    if len(enc_conv1d_filters) == 0:
        # output layer
        decoder.add(
            TimeDistributed(
                Dense(
                    n_features,
                    activation=dec_last_activation,
                    kernel_initializer=INIT_FOR_ACT[dec_last_activation],
                )
            )
        )
    else:
        dec_conv1d_filters = list(reversed(enc_conv1d_filters))
        dec_conv1d_kernel_sizes = list(reversed(enc_conv1d_kernel_sizes))
        dec_conv1d_strides = list(reversed(enc_conv1d_strides))
        dec_conv1d_filters[-1] = n_features
        for i, (filters, kernel_size, strides) in enumerate(
            zip(dec_conv1d_filters, dec_conv1d_kernel_sizes, dec_conv1d_strides)
        ):
            output_layer = i == len(dec_conv1d_filters) - 1
            if output_layer:
                activation = dec_last_activation
                batch_norm = False
            else:
                activation = "relu"
                batch_norm = conv1d_batch_norm
            decoder.add(
                Conv1DTranspose(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    # always use strided convolutions for the output layer upsampling
                    strides=strides if not conv1d_pooling or output_layer else 1,
                    kernel_initializer=INIT_FOR_ACT[activation],
                    use_bias=not batch_norm,
                )
            )
            if batch_norm:
                # normalize on the filters axis
                decoder.add(BatchNormalization(axis=-1))
            decoder.add(Activation(activation))
            if (
                not output_layer and conv1d_pooling and strides > 1
            ):  # typically i % 2 == 0
                decoder.add(UpSampling1D(size=strides))
        upsampled_size = decoder.layers[-1].output.shape[1]
        # crop or pad if the upsampled window size does not match the original window size
        size_diff = upsampled_size - window_size
        if size_diff > 0:
            decoder.add(Cropping1D((0, size_diff)))
        elif size_diff < 0:
            decoder.add(ZeroPadding1D((0, -size_diff)))

    # autoencoder network
    return Sequential([encoder, decoder], name="autoencoder")


def get_autoencoder(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    type_: str = "dense",
    enc_conv1d_filters: Optional[Sequence] = None,
    enc_conv1d_kernel_sizes: Optional[Sequence] = None,
    enc_conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = False,
    conv1d_batch_norm: bool = False,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    dense_layers_activation: str = "relu",
    linear_latent_activation: bool = True,
    dec_last_activation: str = "linear",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    rec_dropout: float = 0.0,
    rec_latent_type: str = "rec",
    conv_add_dense_for_latent: bool = False,
    encoder_only: bool = False,
) -> tf.keras.Model:
    """Returns an autoencoder network with the specified architecture hyperparameters.

    Note: the decoder's architecture is always set to a mirrored version of the encoder.
    Note: if `enc_n_hidden_neurons` is None or empty, we do not include any layer block (rather
    than an empty one), in order not to overcomplicate the graph.

    Neither the latent layer nor the output layer are counted in the hidden layers provided.
    However, the output layer is counted in the Conv1D filters, kernel sizes and strides provided.

    Args:
        window_size: size of input samples in number of records.
        n_features: number of input features.
        latent_dim: dimension of the latent vector representation (coding).
        type_: type of autoencoder to build.
        enc_conv1d_filters: number of filters for each Conv1D layer before the hidden layers.
        enc_conv1d_kernel_sizes: kernel sizes for each Conv1D layer before the hidden layers.
        enc_conv1d_strides: strides for each Conv1D layer before the hidden layers.
        conv1d_pooling: whether to perform downsampling and upsampling through pooling and upsampling
         layers rather than strided convolutions (the last decoder layer will always use strided convolutions).
        conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
        enc_n_hidden_neurons: number of units for each hidden layer before the coding.
        dense_layers_activation: intermediate layers activation for dense architectures.
        linear_latent_activation: whether to consider a linear activation for the latent layer.
        dec_last_activation: last decoder layer activation (either "linear" or "sigmoid").
        input_dropout: dropout rate for the input layer.
        hidden_dropout: dropout rate for other feed-forward layers (except the output).
        rec_unit_type: type of recurrent unit (either "lstm" or "gru").
        activation_rec: activation function to use for recurrent layers (not the "recurrent activation").
        rec_dropout: dropout rate for the hidden state of recurrent layers.
        rec_latent_type: type of latent layers for recurrent architectures.
        conv_add_dense_for_latent: whether to add a dense layer to output the coding for fully
         convolutional architectures.
        encoder_only: whether to return only the encoding part of the autoencoder.

    Returns:
        The autoencoder (or encoder) keras model.
    """
    check_value_in_choices(type_, "type_", ["dense", "rec"])
    check_value_in_choices(rec_latent_type, "rec_latent_type", ["dense", "rec"])
    if type_ == "dense":
        if len(enc_conv1d_filters) > 0:
            raise ValueError(
                "Conv1D layer can only be used with recurrent architectures."
            )
        model = get_dense_autoencoder(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            enc_n_hidden_neurons=enc_n_hidden_neurons,
            hidden_activation=dense_layers_activation,
            linear_latent_activation=linear_latent_activation,
            dec_last_activation=dec_last_activation,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            encoder_only=encoder_only,
        )
    elif len(enc_n_hidden_neurons) == 0:
        model = get_conv_autoencoder(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            enc_conv1d_filters=enc_conv1d_filters,
            enc_conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
            enc_conv1d_strides=enc_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            add_dense_for_latent=conv_add_dense_for_latent,
            dense_layers_activation=dense_layers_activation,
            linear_latent_activation=linear_latent_activation,
            dec_last_activation=dec_last_activation,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            encoder_only=encoder_only,
        )
    else:
        model = get_rec_autoencoder(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            enc_conv1d_filters=enc_conv1d_filters,
            enc_conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
            enc_conv1d_strides=enc_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            enc_n_hidden_neurons=enc_n_hidden_neurons,
            latent_type=rec_latent_type,
            linear_latent_activation=linear_latent_activation,
            dec_last_activation=dec_last_activation,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            rec_unit_type=rec_unit_type,
            activation_rec=activation_rec,
            rec_dropout=rec_dropout,
            encoder_only=encoder_only,
        )
    return model


def compile_autoencoder(
    model: tf.keras.Model,
    loss: str = "mse",
    optimizer: str = "adam",
    adamw_weight_decay: float = 0.0,
    learning_rate: float = 0.001,
):
    """Compiles the autoencoder inplace using the specified optimization hyperparameters.

    Args:
        model: autoencoder model to compile.
        loss: loss function to optimize (either "mse" or "bce").
        optimizer: optimization algorithm used for training the network.
        adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
        learning_rate: learning rate used by the optimization algorithm.
    """
    optimizer = get_optimizer(optimizer, learning_rate, adamw_weight_decay)
    model.compile(loss=PC.loss[loss](), optimizer=optimizer)


def get_autoencoder_dataset(
    X: np.array,
    shuffling_buffer_prop: float = 1.0,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Returns the reconstruction tf.data.Dataset corresponding to `X`.

    Args:
        X: samples of shape `(n_samples, window_size, n_features)`.
        shuffling_buffer_prop: proportion of training data to use as a shuffling buffer.
        batch_size: mini-batch size.

    Returns:
        Corresponding shuffled, batched and prefetched dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, X)).cache()
    buffer_size = int(shuffling_buffer_prop * X.shape[0])
    dataset = dataset.shuffle(buffer_size, seed=21).batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)
