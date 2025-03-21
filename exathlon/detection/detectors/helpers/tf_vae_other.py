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


def get_vae(
    window_size: int,
    n_features: int,
    type_: str = "dense",
    enc_conv1d_filters: Optional[Sequence] = None,
    enc_conv1d_kernel_sizes: Optional[Sequence] = None,
    enc_conv1d_strides: Optional[Sequence] = None,
    enc_n_hidden_neurons: Optional[Sequence] = None,
    dec_n_hidden_neurons: Optional[Sequence] = None,
    dec_conv1d_filters: Optional[Sequence] = None,
    dec_conv1d_kernel_sizes: Optional[Sequence] = None,
    dec_conv1d_strides: Optional[Sequence] = None,
    latent_dim: int = 10,
    dec_output_dist: str = "bernoulli",
    dense_layers_activation: str = "relu",
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    rec_dropout: float = 0.0,
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
    if enc_conv1d_filters is None:
        enc_conv1d_filters = []
    if enc_conv1d_kernel_sizes is None:
        enc_conv1d_kernel_sizes = []
    else:
        enc_conv1d_kernel_sizes = [[s] for s in enc_conv1d_kernel_sizes]
    if enc_conv1d_strides is None:
        enc_conv1d_strides = []
    if enc_n_hidden_neurons is None:
        enc_n_hidden_neurons = []
    if dec_n_hidden_neurons is None:
        dec_n_hidden_neurons = []
    if dec_conv1d_filters is None:
        dec_conv1d_filters = []
    if dec_conv1d_kernel_sizes is None:
        dec_conv1d_kernel_sizes = []
    else:
        dec_conv1d_kernel_sizes = [[s] for s in dec_conv1d_kernel_sizes]
    if dec_conv1d_strides is None:
        dec_conv1d_strides = []
    check_value_in_choices(type_, "type_", ["dense", "rec"])
    check_value_in_choices(dec_output_dist, "dec_output_dist", ["bernoulli", "normal"])
    for k, v in zip(
        ["input_dropout", "hidden_dropout"], [input_dropout, hidden_dropout]
    ):
        check_is_percentage(value=v, var_name=k)
    if type_ == "rec":
        check_is_percentage(rec_dropout, "rec_dropout")

    # latent distribution prior
    latent_prior = Independent(
        Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
    )

    # encoder network
    encoder = Sequential(
        [InputLayer(input_shape=(window_size, n_features))], name="encoder"
    )
    add_dropout_layer(encoder, input_dropout)
    dec_param_multiplier = 2 if dec_output_dist == "normal" else 1

    for i, (filters, kernel_size, strides) in enumerate(
        zip(enc_conv1d_filters, enc_conv1d_kernel_sizes, enc_conv1d_strides)
    ):
        encoder.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                strides=strides,
                kernel_initializer=INIT_FOR_ACT["relu"],
                use_bias=False,
            )
        )
        # normalize on the filters axis
        encoder.add(BatchNormalization(axis=-1))
        encoder.add(Activation("relu"))
        if i % 2 == 0:
            encoder.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))
    if len(enc_conv1d_filters) > 0:
        # InputLayer does not appear in layers
        downsampled_size = encoder.layers[-1].output.shape[1]
    else:
        downsampled_size = window_size

    # type-dependent encoder architectures
    if len(enc_n_hidden_neurons) == 0:
        encoder.add(Flatten())
    else:
        if type_ == "dense":
            encoder.add(Flatten())
            encoder.add(
                LayerBlock(
                    "dense",
                    layers_kwargs={
                        "units": enc_n_hidden_neurons,
                        "activation": [dense_layers_activation],
                        "kernel_initializer": [INIT_FOR_ACT[dense_layers_activation]],
                    },
                    dropout=hidden_dropout,
                    name="encoding_dense_block",
                )
            )
        else:
            encoder.add(
                LayerBlock(
                    rec_unit_type,
                    layers_kwargs={
                        "units": enc_n_hidden_neurons,
                        "dropout": [hidden_dropout],
                        "activation": [activation_rec],
                        "recurrent_dropout": [rec_dropout],
                        "return_sequences": [False],
                    },
                    name=f"encoding_{rec_unit_type}_block",
                )
            )

    # linear activation and twice the dimension to output the mean and standard deviation vectors
    encoder.add(
        Dense(
            2 * latent_dim,
            kernel_initializer=INIT_FOR_ACT["linear"],
            name="qz_x_params",
        )
    )

    # shared q(z|x) layer
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

    # decoder network
    decoder = Sequential([InputLayer(input_shape=[latent_dim])], name="decoder")

    # type-dependent decoder architectures
    if len(dec_n_hidden_neurons) == 0 and len(dec_conv1d_strides) == 0:
        # output the parameters directly
        decoder.add(
            Dense(
                window_size * dec_param_multiplier * n_features,
                activation="linear",
                kernel_initializer=INIT_FOR_ACT["linear"],
            )
        )
        decoder.add(
            Reshape(
                [window_size, dec_param_multiplier * n_features], name="px_z_params"
            )
        )
    elif len(dec_n_hidden_neurons) == 0:
        # input of the convolution upsampling block
        decoder.add(RepeatVector(downsampled_size))
    else:
        # type-dependent hidden block
        if type_ == "dense":
            decoder.add(
                LayerBlock(
                    "dense",
                    layers_kwargs={
                        "units": dec_n_hidden_neurons,
                        "activation": [dense_layers_activation],
                        "kernel_initializer": [INIT_FOR_ACT[dense_layers_activation]],
                    },
                    dropout=hidden_dropout,
                    name="decoding_dense_block",
                )
            )
            if len(dec_conv1d_strides) == 0:
                # outputs are the px_z parameters
                decoder.add(
                    Dense(
                        window_size * dec_param_multiplier * n_features,
                        activation="linear",
                        kernel_initializer=INIT_FOR_ACT["linear"],
                        name="px_z_params",
                    )
                )
                decoder.add(Reshape([window_size, dec_param_multiplier * n_features]))
            else:
                # outputs are the inputs of the conv1d upsampling block
                decoder.add(RepeatVector(downsampled_size))
        else:
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
            if len(dec_conv1d_strides) == 0:
                # outputs are the px_z parameters
                decoder.add(
                    TimeDistributed(
                        Dense(
                            dec_param_multiplier * n_features,
                            activation="linear",
                            kernel_initializer=INIT_FOR_ACT["linear"],
                        )
                    )
                )

    # does not account for last here
    upsampling = list(reversed([i % 2 == 0 for i in range(len(dec_conv1d_filters))]))
    upsampling[-1] = False
    for i, (filters, kernel_size, strides, upsample) in enumerate(
        zip(
            dec_conv1d_filters,
            dec_conv1d_kernel_sizes,
            dec_conv1d_strides,
            upsampling,
        )
    ):
        if i == len(dec_conv1d_filters) - 1:  # last layer
            if filters != dec_param_multiplier * n_features:
                raise ValueError(
                    f"The decoder should end with {dec_param_multiplier} * "
                    f"n_features={dec_param_multiplier * n_features} filters. Received {filters}."
                )
            if strides != 2:
                raise ValueError(
                    f"The decoder should end with 2 strides. Received {strides}."
                )
            activation = "linear"
            batch_norm = False
        else:
            activation = "relu"
            batch_norm = True
        decoder.add(
            Conv1DTranspose(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                strides=strides,
                kernel_initializer=INIT_FOR_ACT[activation],
                use_bias=not batch_norm,
            )
        )
        if batch_norm:
            # normalize on the filters axis
            decoder.add(BatchNormalization(axis=-1))
        decoder.add(Activation(activation))
        if upsample:
            decoder.add(UpSampling1D(size=2))

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

    # variational autoencoder network
    return Sequential([encoder, decoder], name="vae")


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
