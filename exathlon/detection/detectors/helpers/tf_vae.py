from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras import Sequential, Model
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
from tensorflow.keras.callbacks import Callback
from tensorflow_probability.python.layers import (
    KLDivergenceRegularizer,
    DistributionLambda,
)
from tensorflow_probability.python.distributions import (
    Distribution,
    Independent,
    Normal,
    Bernoulli,
)

from utils.guarding import check_value_in_choices, check_is_percentage
from detection.detectors.helpers.tf_helpers import (
    INIT_FOR_ACT,
    MeanMetricWrapper,
    LayerBlock,
    add_dropout_layer,
    get_optimizer,
)


def add_vae_latent_layer(
    encoder, latent_dim, kl_weight, softplus_shift, softplus_scale
):
    # latent distribution prior
    latent_prior = Independent(
        Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
    )
    encoder.add(
        DistributionLambda(
            make_distribution_fn=lambda params: Independent(
                Normal(
                    loc=params[..., :latent_dim],
                    scale=softplus_shift
                    + tf.math.softplus(softplus_scale * params[..., latent_dim:]),
                )
            ),
            convert_to_tensor_fn=Distribution.sample,
            activity_regularizer=KLDivergenceRegularizer(
                latent_prior, weight=kl_weight, use_exact_kl=True
            ),
            name="qz_x",
        ),
    )


def add_vae_output_layer(
    decoder, n_features, dec_output_dist, softplus_shift, softplus_scale
):
    if dec_output_dist == "normal":
        # inputs: normal params of shape (batch_size, window_size, 2 * n_features)
        decoder.add(
            DistributionLambda(
                make_distribution_fn=lambda params: Independent(
                    Normal(
                        loc=params[..., :n_features],
                        scale=softplus_shift
                        + tf.math.softplus(softplus_scale * params[..., n_features:]),
                    )
                ),
                convert_to_tensor_fn=Distribution.sample,
                name="px_z",
            ),
        )
    else:
        # inputs: logits (log-odds) of bernoulli params of shape (batch_size, window_size, n_features)
        decoder.add(
            DistributionLambda(
                make_distribution_fn=lambda logits: Independent(
                    Bernoulli(logits=logits)
                ),
                convert_to_tensor_fn=Distribution.sample,
                name="px_z",
            ),
        )


def get_dense_vae(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    dec_n_hidden_neurons: Optional[Sequence] = None,
    hidden_activation: str = "relu",
    dec_output_dist: str = "normal",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    kl_weight: float = 1.0,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
) -> Sequential:
    """Returns a dense vae model.

    The provided `enc_n_hidden_neurons` does not account for the latent layer. The provided
    `dec_n_hidden_neurons` does not account for the output layer.
    """
    check_is_percentage(input_dropout, "input_dropout")
    check_is_percentage(hidden_dropout, "hidden_dropout")
    if enc_n_hidden_neurons is None:
        enc_n_hidden_neurons = []
    if dec_n_hidden_neurons is None:
        dec_n_hidden_neurons = []
    dec_param_multiplier = 2 if dec_output_dist == "normal" else 1

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
    # twice the dimension to output the mean and standard deviation vectors
    encoder.add(
        Dense(
            2 * latent_dim,
            activation="linear",
            kernel_initializer=INIT_FOR_ACT["linear"],
            name="qz_x_params",
        )
    )
    add_vae_latent_layer(
        encoder=encoder,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
        softplus_shift=softplus_shift,
        softplus_scale=softplus_scale,
    )

    # decoder network
    decoder = Sequential([InputLayer(input_shape=[latent_dim])], name="decoder")
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
            window_size * dec_param_multiplier * n_features,
            activation="linear",
            kernel_initializer=INIT_FOR_ACT["linear"],
        )
    )
    decoder.add(Reshape([window_size, dec_param_multiplier * n_features]))
    add_vae_output_layer(
        decoder=decoder,
        n_features=n_features,
        dec_output_dist=dec_output_dist,
        softplus_shift=softplus_shift,
        softplus_scale=softplus_scale,
    )

    # variational autoencoder network
    return Sequential([encoder, decoder], name="vae")


def get_conv_vae(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    enc_conv1d_filters: Sequence = None,
    enc_conv1d_kernel_sizes: Sequence = None,
    enc_conv1d_strides: Sequence = None,
    conv1d_pooling: bool = False,
    conv1d_batch_norm: bool = False,
    dec_conv1d_filters: Sequence = None,
    dec_conv1d_kernel_sizes: Sequence = None,
    dec_conv1d_strides: Sequence = None,
    dense_layers_activation: str = "relu",
    dec_output_dist: str = "normal",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    kl_weight: float = 1.0,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
) -> Sequential:
    """Returns a fully convolutional vae model.

    The provided `dec_conv1d_filters`, `dec_conv1d_kernel_sizes` and `dec_conv1d_strides` account for
    the output layer. The last number of filters should therefore be either `2 * n_features` for a normal
    output distribution or `n_features` for a bernouilli output distribution.
    """
    check_is_percentage(input_dropout, "input_dropout")
    check_is_percentage(hidden_dropout, "hidden_dropout")
    for text, v in zip(
        ["filters", "kernel sizes", "strides"],
        [enc_conv1d_filters, enc_conv1d_kernel_sizes, enc_conv1d_strides],
    ):
        if v is None or len(v) == 0:
            raise ValueError(
                f"Convolutional {text} must be provided for purely convolutional architectures."
            )
    first_enc_strides = enc_conv1d_strides[0] if len(enc_conv1d_strides) > 0 else 1
    dec_param_multiplier = 2 if dec_output_dist == "normal" else 1

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
    downsampled_size = encoder.layers[-1].output.shape[1]

    # always add a dense layer to output the latent parameters here
    encoder.add(Flatten())
    encoder.add(
        Dense(
            2 * latent_dim,
            activation="linear",
            kernel_initializer=INIT_FOR_ACT["linear"],
            name="qz_x_params",
        )
    )
    add_vae_latent_layer(
        encoder=encoder,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
        softplus_shift=softplus_shift,
        softplus_scale=softplus_scale,
    )

    # decoder network
    decoder = Sequential([InputLayer(input_shape=[latent_dim])], name="decoder")
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
    for i, (filters, kernel_size, strides) in enumerate(
        zip(dec_conv1d_filters, dec_conv1d_kernel_sizes, dec_conv1d_strides)
    ):
        last_layer = i == len(dec_conv1d_filters) - 1
        if last_layer:
            if filters != dec_param_multiplier * n_features:
                raise ValueError(
                    f"The decoder should end with {dec_param_multiplier} * "
                    f"n_features={dec_param_multiplier * n_features} filters. Received {filters}."
                )
            if strides != first_enc_strides:
                raise ValueError(
                    f"The decoder should end with {first_enc_strides} strides. Received {strides}."
                )
            activation = "linear"
            batch_norm = False
        else:
            activation = "relu"
            batch_norm = conv1d_batch_norm
        # always use strided convolution for upsampling for the last layer
        decoder.add(
            Conv1DTranspose(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                strides=strides if not conv1d_pooling or last_layer else 1,
                kernel_initializer=INIT_FOR_ACT[activation],
                use_bias=not batch_norm,
            )
        )
        if batch_norm:
            # normalize on the filters axis
            decoder.add(BatchNormalization(axis=-1))
        decoder.add(Activation(activation))
        if not last_layer and conv1d_pooling and strides > 1:  # typically i % 2 == 0
            decoder.add(UpSampling1D(size=strides))
    upsampled_size = decoder.layers[-1].output.shape[1]
    # crop or pad if the upsampled window size does not match the original window size
    size_diff = upsampled_size - window_size
    if size_diff > 0:
        decoder.add(Cropping1D((0, size_diff)))
    elif size_diff < 0:
        decoder.add(ZeroPadding1D((0, -size_diff)))

    # output layer
    add_vae_output_layer(
        decoder=decoder,
        n_features=n_features,
        dec_output_dist=dec_output_dist,
        softplus_shift=softplus_shift,
        softplus_scale=softplus_scale,
    )

    # variational autoencoder network
    return Sequential([encoder, decoder], name="vae")


def get_rec_vae(
    window_size: int,
    n_features: int,
    latent_dim: int = 10,
    enc_conv1d_filters: Optional[Sequence] = None,
    enc_conv1d_kernel_sizes: Optional[Sequence] = None,
    enc_conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = False,
    conv1d_batch_norm: bool = False,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    dec_n_hidden_neurons: Optional[Sequence] = None,
    dec_conv1d_filters: Optional[Sequence] = None,
    dec_conv1d_kernel_sizes: Optional[Sequence] = None,
    dec_conv1d_strides: Optional[Sequence] = None,
    latent_type: str = "dense",
    dec_output_dist: str = "normal",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    rec_dropout: float = 0.0,
    kl_weight: float = 1.0,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
):
    """Returns a recurrent vae model, with optional convolution-based downsampling.

    The provided `enc_n_hidden_neurons` accounts for the latent layer if `latent_type` is "rec". Otherwise,
    it does not account for the latent layer. If provided, convolution layers for the decoder account
    for the output layer, and should therefore always end with `2 * n_features` filters for a normal
    output distribution, or `n_features` for a bernouilli output distribution. If not provided,
    `dec_n_hidden_neurons` does not account for the output layer, which is set to a time-distributed dense
    layer with the right number of units.
    """
    check_value_in_choices(latent_type, "latent_type", ["dense", "rec"])
    check_is_percentage(input_dropout, "input_dropout")
    check_is_percentage(hidden_dropout, "hidden_dropout")
    check_is_percentage(rec_dropout, "rec_dropout")
    for v in [enc_n_hidden_neurons, dec_n_hidden_neurons]:
        if v is None or len(v) == 0:
            raise ValueError(
                "Call `get_conv_autoencoder()` for a purely convolutional autoencoder."
            )
    if enc_conv1d_filters is None:
        enc_conv1d_filters = []
    if enc_conv1d_kernel_sizes is None:
        enc_conv1d_kernel_sizes = []
    else:
        enc_conv1d_kernel_sizes = [[s] for s in enc_conv1d_kernel_sizes]
    if enc_conv1d_strides is None:
        enc_conv1d_strides = []
    if dec_conv1d_filters is None:
        dec_conv1d_filters = []
    if dec_conv1d_kernel_sizes is None:
        dec_conv1d_kernel_sizes = []
    else:
        dec_conv1d_kernel_sizes = [[s] for s in dec_conv1d_kernel_sizes]
    if dec_conv1d_strides is None:
        dec_conv1d_strides = []
    first_enc_strides = enc_conv1d_strides[0] if len(enc_conv1d_strides) > 0 else 1
    dec_param_multiplier = 2 if dec_output_dist == "normal" else 1

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
        if enc_n_hidden_neurons[-1] != 2 * latent_dim:
            raise ValueError(
                f"For a recurrent latent layer, the encoder block should end with "
                f"2 * latent_dim={2 * latent_dim} units. Received {enc_n_hidden_neurons[-1]}."
            )
        n_before_latent = len(enc_n_hidden_neurons) - 1
        activations = n_before_latent * [activation_rec] + ["linear"]
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
                2 * latent_dim,
                activation="linear",
                kernel_initializer=INIT_FOR_ACT["linear"],
                name="qz_x_params",
            )
        )
    add_vae_latent_layer(
        encoder=encoder,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
        softplus_shift=softplus_shift,
        softplus_scale=softplus_scale,
    )

    # decoder network
    decoder = Sequential([InputLayer(input_shape=[latent_dim])], name="decoder")
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
    if len(dec_conv1d_filters) == 0:
        # output layer params
        decoder.add(
            TimeDistributed(
                Dense(
                    dec_param_multiplier * n_features,
                    activation="linear",
                    kernel_initializer=INIT_FOR_ACT["linear"],
                )
            )
        )
    else:
        for i, (filters, kernel_size, strides) in enumerate(
            zip(dec_conv1d_filters, dec_conv1d_kernel_sizes, dec_conv1d_strides)
        ):
            last_layer = i == len(dec_conv1d_filters) - 1
            if last_layer:
                if filters != dec_param_multiplier * n_features:
                    raise ValueError(
                        f"The decoder should end with {dec_param_multiplier} * "
                        f"n_features={dec_param_multiplier * n_features} filters. Received {filters}."
                    )
                if strides != first_enc_strides:
                    raise ValueError(
                        f"The decoder should end with {first_enc_strides} strides. Received {strides}."
                    )
                activation = "linear"
                batch_norm = False
            else:
                activation = "relu"
                batch_norm = conv1d_batch_norm
            # always use strided convolution for upsampling for the last layer
            decoder.add(
                Conv1DTranspose(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    strides=strides if not conv1d_pooling or last_layer else 1,
                    kernel_initializer=INIT_FOR_ACT[activation],
                    use_bias=not batch_norm,
                )
            )
            if batch_norm:
                # normalize on the filters axis
                decoder.add(BatchNormalization(axis=-1))
            decoder.add(Activation(activation))
            # typically i % 2 == 0
            if not last_layer and conv1d_pooling and strides > 1:
                decoder.add(UpSampling1D(size=strides))
        upsampled_size = decoder.layers[-1].output.shape[1]
        # crop or pad if the upsampled window size does not match the original window size
        size_diff = upsampled_size - window_size
        if size_diff > 0:
            decoder.add(Cropping1D((0, size_diff)))
        elif size_diff < 0:
            decoder.add(ZeroPadding1D((0, -size_diff)))

    # output layer
    add_vae_output_layer(
        decoder=decoder,
        n_features=n_features,
        dec_output_dist=dec_output_dist,
        softplus_shift=softplus_shift,
        softplus_scale=softplus_scale,
    )

    # variational autoencoder network
    return Sequential([encoder, decoder], name="vae")


def get_vae(
    window_size: int,
    n_features: int,
    type_: str = "dense",
    enc_conv1d_filters: Optional[Sequence] = None,
    enc_conv1d_kernel_sizes: Optional[Sequence] = None,
    enc_conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = False,
    conv1d_batch_norm: bool = False,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    dec_n_hidden_neurons: Optional[Sequence] = None,
    dec_conv1d_filters: Optional[Sequence] = None,
    dec_conv1d_kernel_sizes: Optional[Sequence] = None,
    dec_conv1d_strides: Optional[Sequence] = None,
    latent_dim: int = 10,
    dec_output_dist: str = "normal",
    dense_layers_activation: str = "relu",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    rec_dropout: float = 0.0,
    rec_latent_type: str = "dense",
    kl_weight: float = 1.0,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
) -> tf.keras.Model:
    """Returns a variational autoencoder network with the specified architecture hyperparameters.

    Note: the decoder's architecture is always set to a mirrored version of the encoder.
    Note: if `enc_n_hidden_neurons` is None or empty, we do not include any layer block (rather
    than an empty one), in order not to overcomplicate the graph.

    Args:
        window_size: size of input samples in number of records.
        n_features: number of input features.
        type_: type of variational autoencoder to build.
        enc_conv1d_filters: number of filters for each Conv1D layer before the hidden layers.
        enc_conv1d_kernel_sizes: kernel sizes for each Conv1D layer before the hidden layers.
        enc_conv1d_strides: strides for each Conv1D layer before the hidden layers.
        conv1d_pooling: whether to perform downsampling and upsampling through pooling and upsampling
         layers rather than strided convolutions (the last decoder layer will always use strided convolutions).
        conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
        enc_n_hidden_neurons: number of units for each hidden layer before the coding.
        dec_n_hidden_neurons: number of units for each hidden layer after the coding.
        dec_conv1d_filters: number of filters for each Conv1DTranspose layer after the hidden layers.
        dec_conv1d_kernel_sizes: kernel sizes for each Conv1DTranspose layer after the hidden layers.
        dec_conv1d_strides: strides for each Conv1DTranspose layer after the hidden layers.
        latent_dim: dimension of the latent vector representation (coding).
        dec_output_dist: decoder output distribution (either independent "bernoulli" or "normal").
        dense_layers_activation: intermediate layers activation for dense architectures.
        input_dropout: dropout rate for the input layer.
        hidden_dropout: dropout rate for other feed-forward layers (except the output).
        rec_unit_type: type of recurrent unit (either "lstm" or "gru").
        activation_rec: activation function to use for recurrent layers (not the "recurrent activation").
        rec_dropout: recurrent dropout rate.
        rec_latent_type: type of latent layers for recurrent architectures.
        kl_weight: KL divergence term weight in the loss, set here as an activity regularizer.
        softplus_shift: (epsilon) shift to apply after softplus when computing standard deviations.
         The purpose is to stabilize training and prevent NaN probabilities by making sure
         standard deviations of normal distributions are non-zero.
         https://github.com/tensorflow/probability/issues/751 suggests 1e-5.
         https://arxiv.org/pdf/1802.03903.pdf uses 1e-4.
        softplus_scale: scale to apply in the softplus to stabilize training.
         See https://github.com/tensorflow/probability/issues/703 for more details.

    Returns:
        The variational autoencoder keras model.
    """
    check_value_in_choices(type_, "type_", ["dense", "rec"])
    check_value_in_choices(rec_latent_type, "rec_latent_type", ["dense", "rec"])
    check_value_in_choices(dec_output_dist, "dec_output_dist", ["bernoulli", "normal"])
    if type_ == "dense":
        if len(enc_conv1d_filters) > 0:
            raise ValueError(
                "Conv1D layer can only be used with recurrent architectures."
            )
        vae = get_dense_vae(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            enc_n_hidden_neurons=enc_n_hidden_neurons,
            dec_n_hidden_neurons=dec_n_hidden_neurons,
            hidden_activation=dense_layers_activation,
            dec_output_dist=dec_output_dist,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            kl_weight=kl_weight,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
        )
    elif len(enc_n_hidden_neurons) == 0:
        vae = get_conv_vae(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            enc_conv1d_filters=enc_conv1d_filters,
            enc_conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
            enc_conv1d_strides=enc_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            dec_conv1d_filters=dec_conv1d_filters,
            dec_conv1d_kernel_sizes=dec_conv1d_kernel_sizes,
            dec_conv1d_strides=dec_conv1d_strides,
            dense_layers_activation=dense_layers_activation,
            dec_output_dist=dec_output_dist,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            kl_weight=kl_weight,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
        )
    else:
        vae = get_rec_vae(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            enc_conv1d_filters=enc_conv1d_filters,
            enc_conv1d_kernel_sizes=enc_conv1d_kernel_sizes,
            enc_conv1d_strides=enc_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            enc_n_hidden_neurons=enc_n_hidden_neurons,
            dec_conv1d_filters=dec_conv1d_filters,
            dec_conv1d_kernel_sizes=dec_conv1d_kernel_sizes,
            dec_conv1d_strides=dec_conv1d_strides,
            dec_n_hidden_neurons=dec_n_hidden_neurons,
            latent_type=rec_latent_type,
            dec_output_dist=dec_output_dist,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            rec_unit_type=rec_unit_type,
            activation_rec=activation_rec,
            rec_dropout=rec_dropout,
            kl_weight=kl_weight,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
        )
    return vae


def compile_vae(
    model: tf.keras.Model,
    optimizer: str = "adam",
    adamw_weight_decay: float = 0.0,
    learning_rate: float = 0.001,
    grad_norm_limit: float = 10.0,
):
    """Compiles the VAE inplace using the specified optimization hyperparameters.

    Args:
        model: variational autoencoder to compile as a keras model.
        optimizer: optimization algorithm used for training the network.
        adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
        learning_rate: learning rate used by the optimization algorithm.
        grad_norm_limit: gradient norm clipping value.
    """

    def negative_log_likelihood(x, x_dist):
        """Returns the negative log-likelihood of the input with respect to the output distribution."""
        return -x_dist.log_prob(x)

    optimizer = get_optimizer(
        optimizer, learning_rate, adamw_weight_decay, clipnorm=grad_norm_limit
    )
    # separately record reconstruction loss as a metric (with a custom wrapper to handle probabilistic outputs)
    model.compile(
        loss=negative_log_likelihood,
        optimizer=optimizer,
        metrics=[MeanMetricWrapper(negative_log_likelihood, name="nll_loss")],
    )


class KLLossCallback(Callback):
    """Keras callback recording the KL divergence as a remainder from the negative log-likelihood loss."""

    def on_epoch_end(self, epoch, logs=None):
        for p in ["", "val_"]:
            logs[f"{p}kl_loss"] = logs[f"{p}loss"] - logs[f"{p}nll_loss"]
