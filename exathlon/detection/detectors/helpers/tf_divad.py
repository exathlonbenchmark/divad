import math
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Layer,
    Input,
    InputLayer,
    Concatenate,
    Dropout,
    Flatten,
    Reshape,
    Activation,
    Dense,
    RepeatVector,
    TimeDistributed,
    Conv1D,
    Conv1DTranspose,
    LSTM,
    GRU,
    MaxPooling1D,
    BatchNormalization,
    UpSampling1D,
    Cropping1D,
)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import Callback
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import (
    Distribution,
    Independent,
    Normal,
    Bernoulli,
    MixtureSameFamily,
    Categorical,
    MultivariateNormalDiag,
    kl_divergence,
)

from utils.guarding import check_value_in_choices
from detection.detectors.helpers.tf_helpers import (
    get_optimizer,
    INIT_FOR_ACT,
    get_shape_and_cropping,
)
from detection.detectors.helpers.tfp_helpers import (
    GMPrior,
    VampPrior,
    get_kl_divergence,
)
from detection.detectors.helpers.tf_cov_weighting import TensorFlowCoVWeighting


class BetaLinearScheduler(Callback):
    def __init__(self, n_epochs=100, min_beta=0.0, max_beta=1.0):
        self.n_epochs = n_epochs
        self.min_beta = min_beta
        self.max_beta = max_beta

    def on_epoch_begin(self, epoch, logs=None):
        # always consider `max_beta` after `self.n_epochs` epochs
        beta = (
            self.min_beta
            + (self.max_beta - self.min_beta)
            * min(epoch, self.n_epochs)
            / self.n_epochs
        )
        self.model.beta = beta


class BetaLogger(Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        logs = logs or dict()
        logs.update({"beta": self.model.beta})
        tf.summary.scalar("beta", data=self.model.beta, step=batch)


def get_qz_x(
    window_size: int = 1,
    n_features: int = 237,
    type_: str = "dense",
    conv1d_filters: Optional[Sequence] = None,
    conv1d_kernel_sizes: Optional[Sequence] = None,
    conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = True,
    conv1d_batch_norm: bool = True,
    n_hidden: Optional[Sequence] = None,
    latent_dim: int = 64,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    rec_weight_decay: float = 0.0,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
    weight_decay: float = 0.0,
    name: str = "qz_x",
) -> (tf.keras.Model, int):
    """Returns both `qz_x` and the downsampled size resulting from `conv1d_strides`."""
    if conv1d_filters is None:
        conv1d_filters = []
    if conv1d_kernel_sizes is None:
        conv1d_kernel_sizes = []
    if conv1d_strides is None:
        conv1d_strides = []
    if n_hidden is None:
        n_hidden = []
    if type_ == "rec":
        check_value_in_choices(rec_unit_type, "rec_unit_type", ["lstm", "gru"])
    qz_x = Sequential([InputLayer(input_shape=(window_size, n_features))], name=name)
    # shared convolution-based downsampling
    if len(conv1d_strides) == 0:
        downsampled_size = window_size
    else:
        for filters, kernel_size, strides in zip(
            conv1d_filters, conv1d_kernel_sizes, conv1d_strides
        ):
            qz_x.add(
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
                qz_x.add(BatchNormalization(axis=-1))
            qz_x.add(Activation("relu"))
            if conv1d_pooling and strides > 1:  # typically i % 2 == 0
                qz_x.add(
                    MaxPooling1D(pool_size=strides, strides=strides, padding="same")
                )
        # InputLayer does not appear in layers
        downsampled_size = qz_x.layers[-1].output.shape[1]

    # type-dependent hidden layers
    if type_ == "dense" or len(n_hidden) == 0:
        # input of either Dense block or latent layer directly
        qz_x.add(Flatten())
    if type_ == "dense":
        for n in n_hidden:
            qz_x.add(
                Dense(
                    n,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    kernel_regularizer=L2(weight_decay),
                )
            )
    elif type_ == "rec":
        rec_layer_class = LSTM if rec_unit_type == "lstm" else GRU
        for i, n in enumerate(n_hidden):
            qz_x.add(
                rec_layer_class(
                    n,
                    activation=activation_rec,
                    kernel_initializer=INIT_FOR_ACT[activation_rec],
                    return_sequences=i < len(n_hidden) - 1,  # `False` for last only
                    kernel_regularizer=L2(weight_decay),
                    recurrent_regularizer=L2(rec_weight_decay),
                )
            )
    else:
        raise ValueError('Invalid `type_`, please choose from ["dense", "rec"].')
    qz_x.add(
        Dense(
            2 * latent_dim,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
    )
    qz_x.add(
        DistributionLambda(
            make_distribution_fn=lambda params: Independent(
                Normal(
                    loc=params[..., :latent_dim],
                    scale=softplus_shift
                    + tf.math.softplus(softplus_scale * params[..., latent_dim:]),
                ),
                reinterpreted_batch_ndims=1,  # equivalent to `None`: all but first batch axis
            ),
            convert_to_tensor_fn=Distribution.sample,
        )
    )
    return qz_x, downsampled_size


class PseudoInputEncodingLayer(Layer):
    """Uses `encoder` to map a batch of pseudo-inputs of shape `(B, K, W, D)` to their corresponding
      means and standard deviations, each of shape `(B, K * L)`.

    The purpose of this layer is to temporally "merge" the batch and component dimensions `B` and `K`.

    Args:
        encoder: the encoder `qz_x`, whose `Independent(Normal)` output distribution is used to compute
         the means and scales.
        **kwargs: other keyword arguments of `tf.keras.layers.Layer`.
    """

    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def call(self, u):
        B = tf.shape(u)[0]
        K = tf.shape(u)[1]
        W = tf.shape(u)[2]
        D = tf.shape(u)[3]
        # encode batch of `B * K` pseudo-inputs
        u = tf.reshape(u, [B * K, W, D])
        qz_u_dists = self.encoder(u)  # shape `(B * K, L)`
        qz_u_loc = qz_u_dists.distribution.loc  # shape `(B * K, L)`
        qz_u_scale = qz_u_dists.distribution.scale  # shape `(B * K, L)`
        L = tf.shape(qz_u_loc)[-1]
        qz_u_loc = tf.reshape(qz_u_loc, [B, K * L])
        qz_u_scale = tf.reshape(qz_u_scale, [B, K * L])
        # returned shapes: `(B, K * L)` and `(B, K * L)`
        return qz_u_loc, qz_u_scale

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "encoder": self.encoder}


def get_pz_class(
    n_classes: int = 7,
    latent_dim: int = 64,
    n_hidden: Optional[Sequence] = None,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
    weight_decay: float = 0.0,
    output_dist: str = "normal",
    output_gm_n_components: int = 16,
    output_vamp_window_size: int = 1,
    output_vamp_n_features: int = 237,
    output_vamp_n_components: int = 16,
    output_vamp_encoder: Model = None,
    z_name: str = "zd",
    class_name: str = "d",
):
    """Returns `p(z | class)`, where `class` is either the domain or anomaly-related class."""
    if n_hidden is None:
        n_hidden = []
    check_value_in_choices(output_dist, "output_dist", ["normal", "gm", "vamp"])
    if output_dist in ["normal", "gm"]:
        # use sequential API
        pz_class = Sequential(
            [InputLayer(input_shape=n_classes)], name=f"p{z_name}_{class_name}"
        )
        for n in n_hidden:
            pz_class.add(
                Dense(
                    n,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    kernel_regularizer=L2(weight_decay),
                )
            )
        if output_dist == "normal":
            pz_class.add(
                Dense(
                    2 * latent_dim,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )
            pz_class.add(
                DistributionLambda(
                    make_distribution_fn=lambda params: Independent(
                        Normal(
                            loc=params[..., :latent_dim],
                            scale=softplus_shift
                            + tf.math.softplus(
                                softplus_scale * params[..., latent_dim:]
                            ),
                        ),
                        reinterpreted_batch_ndims=1,  # equivalent to `None`: all but first batch axis
                    ),
                    convert_to_tensor_fn=Distribution.sample,
                )
            )
        else:
            L = latent_dim
            K = output_gm_n_components
            pz_class.add(
                Dense(
                    K + K * L + K * L,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )
            # shape: `(B, K + K * L + K * L)`, where `B` is `batch_size`
            pz_class.add(
                DistributionLambda(
                    make_distribution_fn=lambda params: MixtureSameFamily(
                        mixture_distribution=Categorical(logits=params[..., :K]),
                        components_distribution=MultivariateNormalDiag(
                            loc=tf.reshape(
                                params[..., K : K + K * L], [tf.shape(params)[0], K, L]
                            ),
                            scale_diag=tf.reshape(
                                softplus_shift
                                + tf.math.softplus(
                                    softplus_scale * params[..., K + K * L :]
                                ),
                                [tf.shape(params)[0], K, L],
                            ),
                        ),
                    ),
                    convert_to_tensor_fn=Distribution.sample,
                ),
            )
    else:
        # use functional API
        inputs = Input(shape=n_classes)
        h = inputs
        for n in n_hidden:
            h = Dense(
                n,
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=L2(weight_decay),
            )(h)
        # outputs the weights and pseudo-inputs
        W = output_vamp_window_size
        D = output_vamp_n_features
        L = latent_dim
        K = output_vamp_n_components
        # mixture weights of shape `(B, K)`, where `B` is `batch_size`
        w = Dense(K, kernel_initializer="glorot_uniform", bias_initializer="zeros")(h)
        # pseudo-inputs
        u = Dense(
            K * W * D, kernel_initializer="glorot_uniform", bias_initializer="zeros"
        )(h)
        u = Reshape([K, W, D])(u)
        # shapes of each: `(B, K * L)`
        qz_u_loc, qz_u_scale = PseudoInputEncodingLayer(encoder=output_vamp_encoder)(u)
        # shape `(B, K + K * L + K * L)`
        pz_class_params = Concatenate(axis=-1)([w, qz_u_loc, qz_u_scale])
        # scales are already constrained outputs: no softplus
        outputs = DistributionLambda(
            make_distribution_fn=lambda params: MixtureSameFamily(
                mixture_distribution=Categorical(logits=params[..., :K]),
                components_distribution=MultivariateNormalDiag(
                    loc=tf.reshape(
                        params[..., K : K + K * L], [tf.shape(params)[0], K, L]
                    ),
                    scale_diag=tf.reshape(
                        params[..., K + K * L :], [tf.shape(params)[0], K, L]
                    ),
                ),
            ),
            convert_to_tensor_fn=Distribution.sample,
        )(pz_class_params)
        pz_class = Model(inputs=inputs, outputs=outputs, name=f"p{z_name}_{class_name}")
    return pz_class


def get_px_zs(
    window_size: int = 1,
    n_features: int = 237,
    type_: str = "dense",
    conv1d_filters: Optional[Sequence] = None,
    conv1d_kernel_sizes: Optional[Sequence] = None,
    conv1d_strides: Optional[Sequence] = None,
    conv1d_pooling: bool = True,
    conv1d_batch_norm: bool = True,
    n_hidden: Optional[Sequence] = None,
    latent_dim: int = 64,
    n_zs: int = 2,
    downsampled_size: int = 1,
    rec_unit_type: str = "lstm",
    activation_rec: str = "tanh",
    dec_output_dist: str = "bernoulli",
    rec_weight_decay: float = 1e-4,
    softplus_shift: float = 1e-4,
    softplus_scale: float = 1.0,
    weight_decay: float = 0.0,
    z_names: str = "zyd",
) -> tf.keras.Model:
    """Returns `p(x | z's)`, where `z's` can be either `(zx, zy, zd)` (i.e., `zxyd`) or just `(zy, zd)` (`zyd`).

    Note: the last element of `conv1d_filters` should be `n_features`.
    """
    if conv1d_filters is None:
        conv1d_filters = []
    if conv1d_kernel_sizes is None:
        conv1d_kernel_sizes = []
    if conv1d_strides is None:
        conv1d_strides = []
    if n_hidden is None:
        n_hidden = []
    if type_ == "rec":
        check_value_in_choices(rec_unit_type, "rec_unit_type", ["lstm", "gru"])
    check_value_in_choices(dec_output_dist, dec_output_dist, ["normal", "bernoulli"])
    if len(conv1d_strides) == 0 and downsampled_size != window_size:
        raise ValueError(
            "Should have `downsampled_size == window_size` when no Conv1D upsampling."
        )
    dec_param_multiplier = 2 if dec_output_dist == "normal" else 1
    px_zs = Sequential(
        [InputLayer(input_shape=n_zs * latent_dim)], name=f"px_{z_names}"
    )
    # type-dependent hidden layers
    if type_ == "dense":
        for n in n_hidden:
            px_zs.add(
                Dense(
                    n,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    kernel_regularizer=L2(weight_decay),
                )
            )
        if len(conv1d_strides) == 0:
            # outputs are the px_zs parameters of shape `(window_size, 1|2 * n_features)`
            px_zs.add(
                Dense(
                    window_size * dec_param_multiplier * n_features,
                    activation="linear",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )
            px_zs.add(Reshape([window_size, dec_param_multiplier * n_features]))
        else:
            # outputs are the inputs of the conv1d upsampling block
            px_zs.add(
                Dense(
                    downsampled_size * n_zs * latent_dim,
                    activation="relu",
                    kernel_initializer=INIT_FOR_ACT["relu"],
                    kernel_regularizer=L2(weight_decay),
                )
            )
            px_zs.add(Reshape([downsampled_size, n_zs * latent_dim]))
    elif type_ == "rec":
        px_zs.add(RepeatVector(downsampled_size))
        rec_layer_class = LSTM if rec_unit_type == "lstm" else GRU
        for n in n_hidden:
            px_zs.add(
                rec_layer_class(
                    n,
                    activation=activation_rec,
                    kernel_initializer=INIT_FOR_ACT[activation_rec],
                    return_sequences=True,
                    kernel_regularizer=L2(weight_decay),
                    recurrent_regularizer=L2(rec_weight_decay),
                )
            )
        if len(conv1d_strides) == 0:
            # outputs are the px_zs parameters of shape `(window_size, 1|2 * n_features)`
            px_zs.add(
                TimeDistributed(
                    Dense(
                        dec_param_multiplier * n_features,
                        activation="linear",
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                    )
                )
            )
    else:
        raise ValueError('Invalid `type_`, please choose from ["dense", "rec"].')
    # shared convolution-based upsampling
    if len(conv1d_strides) > 0:
        # input shape: `(None, downsampled_size, n_features)`, with "relu" activation
        for i, (filters, kernel_size, strides) in enumerate(
            zip(conv1d_filters, conv1d_kernel_sizes, conv1d_strides)
        ):
            last_layer = i == len(conv1d_strides) - 1
            if not last_layer:
                layer_filters = filters
                activation = "relu"
                batch_norm = conv1d_batch_norm
            else:
                # outputs the px_zs parameters of shape `(window_size, 1|2 * n_features)`
                if filters != n_features:
                    raise ValueError(
                        f"The decoder should end with n_features={n_features} filters. Received {filters}."
                    )
                layer_filters = dec_param_multiplier * filters
                activation = "linear"
                # never use batch norm for the last trainable layer
                batch_norm = False
            px_zs.add(
                Conv1DTranspose(
                    filters=layer_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    # always use strided convolutions for the last layer upsampling
                    strides=strides if not conv1d_pooling or last_layer else 1,
                    kernel_initializer=INIT_FOR_ACT[activation],
                    use_bias=not batch_norm,
                )
            )
            # normalize on the filters axis
            if batch_norm:
                px_zs.add(BatchNormalization(axis=-1))
            px_zs.add(Activation(activation))
            # typically reversed([i % 2 == 0 for i in range(len(time_conv1d_filters))])
            if not last_layer and conv1d_pooling and strides > 1:
                px_zs.add(UpSampling1D(size=strides))
        # crop Conv1DTranspose output to match the window size if not a multiple of the reduction factor
        _, cropping = get_shape_and_cropping(
            window_size=window_size,
            n_features=dec_param_multiplier * n_features,
            reduction_factor=math.ceil(window_size / downsampled_size),
        )
        if cropping[0][0] > 0 or cropping[0][1] > 0:
            px_zs.add(Cropping1D(cropping[0]))

    if dec_output_dist == "normal":
        # input shape: `(None, window_size, 2 * n_features)`, with "linear" activation
        assert dec_param_multiplier == 2
        px_zs.add(
            DistributionLambda(
                make_distribution_fn=lambda params: Independent(
                    Normal(
                        loc=params[..., :n_features],
                        scale=softplus_shift
                        + tf.math.softplus(softplus_scale * params[..., n_features:]),
                    ),
                    reinterpreted_batch_ndims=2,  # equivalent to `None`: all but first batch axis
                ),
                convert_to_tensor_fn=Distribution.sample,
            )
        )
    elif dec_output_dist == "bernoulli":
        # input shape: `(None, window_size, n_features)`, with "linear" activation
        assert dec_param_multiplier == 1
        px_zs.add(
            DistributionLambda(
                make_distribution_fn=lambda logits: Independent(
                    Bernoulli(logits=logits),
                    reinterpreted_batch_ndims=2,  # equivalent to `None`: all but first batch axis
                ),
                convert_to_tensor_fn=Distribution.sample,
            )
        )
    else:
        raise ValueError(
            'Invalid `dec_output_dist`, please choose from ["normal", "bernoulli"].'
        )
    return px_zs


def get_qclass_z(
    latent_dim: int = 64, n_classes: int = 7, z_name: str = "zd", class_name: str = "d"
):
    """Returns `q(class | z)`, where `class` can be either the domain or anomaly-related class.

    This classifier should be quite "simple", as we should be able to easily classify the class from z.
    """
    return Sequential(
        [
            InputLayer(input_shape=latent_dim),
            Activation("relu"),  # z_d is unbounded
            Dense(
                n_classes, kernel_initializer="glorot_uniform", bias_initializer="zeros"
            ),
            DistributionLambda(
                make_distribution_fn=lambda logits: Categorical(logits=logits),
                convert_to_tensor_fn=Distribution.sample,
            ),
        ],
        name=f"q{class_name}_{z_name}",
    )


class TensorFlowDivad(Model):
    def __init__(
        self,
        window_size: int = 1,
        n_features: int = 237,
        n_domains: int = 7,
        type_: str = "dense",
        pzy_dist: str = "standard",
        pzy_kl_n_samples: int = 1,
        pzy_gm_n_components: int = 16,
        pzy_gm_softplus_scale: float = 1.0,
        pzy_vamp_n_components: int = 16,
        qz_x_conv1d_filters: Optional[Sequence] = None,
        qz_x_conv1d_kernel_sizes: Optional[Sequence] = None,
        qz_x_conv1d_strides: Optional[Sequence] = None,
        qz_x_n_hidden: Optional[Sequence] = None,
        pzd_d_n_hidden: Optional[Sequence] = None,
        px_z_conv1d_filters: Optional[Sequence] = None,
        px_z_conv1d_kernel_sizes: Optional[Sequence] = None,
        px_z_conv1d_strides: Optional[Sequence] = None,
        px_z_n_hidden: Optional[Sequence] = None,
        conv1d_pooling: bool = True,
        conv1d_batch_norm: bool = True,
        latent_dim: int = 64,
        rec_unit_type: str = "lstm",
        activation_rec: str = "tanh",
        rec_weight_decay: float = 0.0,
        dec_output_dist: str = "bernoulli",
        softplus_shift: float = 1e-4,
        softplus_scale: float = 1.0,
        weight_decay: float = 0.0,
        dropout: float = 0.0,  # TODO: currently unused.
        min_beta: float = 0.0,
        loss_weighting: str = "fixed",
        d_classifier_weight: float = 100000.0,
        mean_sort: str = "full",
        mean_decay_param: float = 1.0,
        **kwargs,
    ):
        """TensorFlow DIVAD implementation.

        Args:
            window_size: size of input samples in number of records.
            n_features: number of input features.
            n_domains: number of training domains.
            type_: type of `qz_x` and `px_z` networks (either "dense" or "rec").
            pzy_dist: prior distribution for zy (either "standard", "gm" or "vamp").
            pzy_kl_n_samples: number of MC samples to estimate the KL with pzy if `pzy_dist` is not "standard".
            pzy_gm_n_components: number of GM components if `pzy_dist` is "gm".
            pzy_gm_softplus_scale: softplus scale to apply to GM components to stabilize training if
             `pzy_dist` is "gm".
            pzy_vamp_n_components: number of pseudo inputs (i.e., mixture components) if `pzy_dist` is "vamp".
            qz_x_conv1d_filters: 1d convolution filters of the `qz_x` networks.
            qz_x_conv1d_kernel_sizes: 1d convolution kernel sizes of the `qz_x` networks.
            qz_x_conv1d_strides: `qz_x` 1d convolution strides of the `qz_x` networks.
            qz_x_n_hidden: number of neurons for each hidden layer of the `qz_x` networks.
            pzd_d_n_hidden: number of neurons for each hidden layer of the `pzd_d` network.
            px_z_conv1d_filters: 1d convolution filters of the `px_z` network (the last number of filters
             should be the number of features).
            px_z_conv1d_kernel_sizes: 1d convolution kernel sizes of the `px_z` network.
            px_z_conv1d_strides: `qz_x` 1d convolution strides of the `px_z` network.
            px_z_n_hidden: number of neurons for each hidden layer of the `px_z` network.
            conv1d_pooling: whether to perform downsampling and upsampling through pooling and upsampling
             layers rather than strided convolutions (the last decoder layer will always use strided convolutions).
            conv1d_batch_norm: whether to apply batch normalization for Conv1D layers.
            latent_dim: dimension of the latent vector representation (coding).
            rec_unit_type: type of recurrent unit used for recurrent DIVAD (either "lstm" or "gru").
            activation_rec: activation function used by recurrent DIVAD units (not the "recurrent activation").
            rec_weight_decay: L2 weight decay to apply to recurrent layers.
            dec_output_dist: output distribution of `px_z`.
            softplus_shift: (epsilon) shift to apply after softplus when computing standard deviations.
             The purpose is to stabilize training and prevent NaN probabilities by making sure
             standard deviations of normal distributions are non-zero.
             https://github.com/tensorflow/probability/issues/751 suggests 1e-5.
             https://arxiv.org/pdf/1802.03903.pdf uses 1e-4.
            softplus_scale: scale to apply in the softplus to stabilize training.
            weight_decay: L2 weight decay to apply to feed-forward layers.
             See https://github.com/tensorflow/probability/issues/703 for more details.
            min_beta: minimum KL divergence weight, which can later be updated by a `BetaLinearScheduler`
             callback.
            loss_weighting: loss weighting strategy (either "cov_weighting" or "fixed").
            d_classifier_weight: weight of the domain classification loss in the total loss if
             `loss_weighting` is "fixed" (ELBO is one).
            mean_sort: either "full" or "decay", if `loss_weighting` is "cov_weighting".
            mean_decay_param: what decay to use with mean decay, if `loss_weighting` is "cov_weighting".
            **kwargs: other keyword arguments of `tf.keras.Model`.
        """
        super().__init__(**kwargs)
        check_value_in_choices(type_, "type_", ["dense", "rec"])
        check_value_in_choices(pzy_dist, "pzy_dist", ["standard", "gm", "vamp"])
        if type_ == "rec":
            check_value_in_choices(rec_unit_type, "rec_unit_type", ["lstm", "gru"])
        check_value_in_choices(
            dec_output_dist, "dec_output_dist", ["bernoulli", "normal"]
        )
        check_value_in_choices(
            loss_weighting, "loss_weighting", ["cov_weighting", "fixed"]
        )
        check_value_in_choices(mean_sort, "mean_sort", ["full", "decay"])
        if qz_x_conv1d_kernel_sizes is not None:
            qz_x_conv1d_kernel_sizes = [[s] for s in qz_x_conv1d_kernel_sizes]
        if px_z_conv1d_kernel_sizes is not None:
            px_z_conv1d_kernel_sizes = [[s] for s in px_z_conv1d_kernel_sizes]
        self.latent_dim = latent_dim
        self.qzy_x, downsampled_size = get_qz_x(
            window_size=window_size,
            n_features=n_features,
            type_=type_,
            conv1d_filters=qz_x_conv1d_filters,
            conv1d_kernel_sizes=qz_x_conv1d_kernel_sizes,
            conv1d_strides=qz_x_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            n_hidden=qz_x_n_hidden,
            latent_dim=latent_dim,
            rec_unit_type=rec_unit_type,
            activation_rec=activation_rec,
            rec_weight_decay=rec_weight_decay,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
            weight_decay=weight_decay,
            name="qzy_x",
        )
        self.qzd_x, _ = get_qz_x(
            window_size=window_size,
            n_features=n_features,
            type_=type_,
            conv1d_filters=qz_x_conv1d_filters,
            conv1d_kernel_sizes=qz_x_conv1d_kernel_sizes,
            conv1d_strides=qz_x_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            n_hidden=qz_x_n_hidden,
            latent_dim=latent_dim,
            rec_unit_type=rec_unit_type,
            activation_rec=activation_rec,
            rec_weight_decay=rec_weight_decay,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
            weight_decay=weight_decay,
            name="qzd_x",
        )
        self.pzd_d = get_pz_class(
            n_classes=n_domains,
            latent_dim=latent_dim,
            n_hidden=pzd_d_n_hidden,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
            weight_decay=weight_decay,
            output_dist="normal",
            z_name="zd",
            class_name="d",
        )
        self.px_zyd = get_px_zs(
            window_size=window_size,
            n_features=n_features,
            type_=type_,
            conv1d_filters=px_z_conv1d_filters,
            conv1d_kernel_sizes=px_z_conv1d_kernel_sizes,
            conv1d_strides=px_z_conv1d_strides,
            conv1d_pooling=conv1d_pooling,
            conv1d_batch_norm=conv1d_batch_norm,
            n_hidden=px_z_n_hidden,
            latent_dim=latent_dim,
            n_zs=2,
            downsampled_size=downsampled_size,
            rec_unit_type=rec_unit_type,
            activation_rec=activation_rec,
            rec_weight_decay=rec_weight_decay,
            dec_output_dist=dec_output_dist,
            softplus_shift=softplus_shift,
            softplus_scale=softplus_scale,
            weight_decay=weight_decay,
            z_names="zyd",
        )
        self.pzy_dist = pzy_dist
        self.pzy_kl_n_samples = pzy_kl_n_samples
        if self.pzy_dist == "standard":
            self.pzy = Independent(
                Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
            )
        elif self.pzy_dist == "gm":
            self.pzy = GMPrior(
                latent_dim,
                n_components=pzy_gm_n_components,
                softplus_shift=softplus_shift,
                softplus_scale=pzy_gm_softplus_scale,
            )
        elif self.pzy_dist == "vamp":
            self.pzy = VampPrior(
                window_size=window_size,
                n_features=n_features,
                latent_dim=latent_dim,
                n_components=pzy_vamp_n_components,
                encoder=self.qzy_x,
            )
        else:
            self.pzy = None
        self.qd_zd = get_qclass_z(
            latent_dim=latent_dim, n_classes=n_domains, z_name="zd", class_name="d"
        )
        self.beta = min_beta  # can be updated by `BetaLinearScheduler`
        self.loss_weighting = loss_weighting
        if loss_weighting == "cov_weighting":
            self.cov_weighting = TensorFlowCoVWeighting(
                num_losses=2, mean_sort=mean_sort, mean_decay_param=mean_decay_param
            )
            self.d_classifier_weight = None
        else:
            self.cov_weighting = None
            self.d_classifier_weight = d_classifier_weight

    def build(self, inputs_shape):
        x_shape, d_shape = inputs_shape
        self.qzy_x.build(x_shape)
        self.qzd_x.build(x_shape)
        self.pzd_d.build(d_shape)
        self.px_zyd.build([x_shape[0], 2 * self.latent_dim])

    def call(self, inputs, training=False):
        x, d = inputs
        qzy_x = self.qzy_x(x, training=training)
        qzd_x = self.qzd_x(x, training=training)
        pzd_d = self.pzd_d(d, training=training)
        if self.pzy_dist == "standard":
            # exact kl divergence
            zy_kl = kl_divergence(qzy_x, self.pzy)  # broadcasting
        else:
            # approximate kl divergence
            zy_kl = get_kl_divergence(
                qzy_x, self.pzy, exact=False, n_samples=self.pzy_kl_n_samples
            )
        # exact kl divergence
        zd_kl = kl_divergence(qzd_x, pzd_d)
        px_zyd = self.px_zyd(tf.concat([qzy_x, qzd_x], -1), training=training)
        # negative elbo
        nll = -px_zyd.log_prob(x)
        neg_elbo = nll + self.beta * zy_kl + self.beta * zd_kl
        # domain classification loss (for each example, negative log-prob output for the relevant class)
        qd_zd = self.qd_zd(qzd_x, training=training)
        d_true = tf.math.argmax(d, axis=-1)
        d_loss = -qd_zd.log_prob(d_true)
        # total loss
        if self.loss_weighting == "fixed":
            loss = neg_elbo + self.d_classifier_weight * d_loss
            self.add_loss(tf.reduce_mean(loss))
        else:
            loss = self.cov_weighting.get_combined_loss(
                tf.convert_to_tensor(
                    [tf.math.reduce_mean(neg_elbo), tf.math.reduce_mean(d_loss)],
                    dtype=tf.float32,
                ),
                training=training,
            )
            if training:
                self.add_metric(
                    self.cov_weighting.alphas[0],
                    name="neg_elbo_weight",
                    aggregation="mean",
                )
                self.add_metric(
                    self.cov_weighting.alphas[1],
                    name="d_loss_weight",
                    aggregation="mean",
                )
            self.add_loss(loss)
        # other logged metrics
        self.add_metric(neg_elbo, name="neg_elbo", aggregation="mean")
        self.add_metric(d_loss, name="d_loss", aggregation="mean")
        self.add_metric(nll, name="nll", aggregation="mean")
        self.add_metric(zy_kl, name="zy_kl", aggregation="mean")
        self.add_metric(zd_kl, name="zd_kl", aggregation="mean")
        # domain classification accuracy
        d_preds = tf.math.argmax(qd_zd.logits, axis=-1)
        d_acc = tf.math.reduce_mean(tf.cast(tf.math.equal(d_preds, d_true), tf.float32))
        self.add_metric(d_acc, name="d_acc", aggregation="mean")
        if self.pzy_dist in ["gm", "vamp"]:
            # check the mixture gets updated
            self.add_metric(
                tf.math.reduce_mean(self.pzy.get_dist().mixture_distribution.logits),
                name=f"mean_{self.pzy_dist}_weight_logits",
                aggregation="mean",
            )
            self.add_metric(
                tf.math.reduce_mean(
                    tf.math.reduce_mean(
                        self.pzy.get_dist().components_distribution.loc, axis=1
                    )
                ),
                name=f"mean_{self.pzy_dist}_avg_loc",
                aggregation="mean",
            )
            self.add_metric(
                tf.math.reduce_mean(
                    tf.math.reduce_mean(
                        self.pzy.get_dist().components_distribution.scale.to_dense(),
                        axis=[1, 2],
                    )
                ),
                name=f"mean_{self.pzy_dist}_avg_scale",
                aggregation="mean",
            )

    def get_config(self):
        pass


def compile_divad(
    model: tf.keras.Model,
    optimizer: str = "adam",
    adamw_weight_decay: float = 0.0,
    learning_rate: float = 0.001,
    grad_norm_limit: float = 10.0,
):
    """Compiles the DIVAD model inplace using the specified optimization hyperparameters.

    Args:
        model: DIVAD model to compile as a keras model.
        optimizer: optimization algorithm used for training the network.
        adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.
        learning_rate: learning rate used by the optimization algorithm.
        grad_norm_limit: gradient norm clipping value.
    """
    optimizer = get_optimizer(
        optimizer, learning_rate, adamw_weight_decay, clipnorm=grad_norm_limit
    )
    model.compile(optimizer=optimizer)
