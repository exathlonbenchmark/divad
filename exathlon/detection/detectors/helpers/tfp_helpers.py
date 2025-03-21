import tensorflow as tf
from tensorflow_probability.python.distributions import (
    MixtureSameFamily,
    Categorical,
    MultivariateNormalDiag,
    kl_divergence,
)


def get_kl_divergence(distribution_a, distribution_b, exact=False, n_samples: int = 1):
    """Returns the KL divergences between batches of `distribution_a` and `distribution_b`.

    Args:
        distribution_a: first distribution batch of shape `(batch_size, event_dim)`.
        distribution_b: second distribution batch of shape `(batch_size, event_dim)`.
        exact: whether to compute the exact KL or an MC approximation.
        n_samples: number of MC samples if `exact` is `False`.

    Returns:
        The batch of KL divergences of shape `(batch_size)`.
    """
    if exact:
        return kl_divergence(distribution_a, distribution_b)
    # shape `(n_samples, batch_size, event_dim)`
    a_samples = distribution_a.sample(n_samples)
    # shape `(batch_size,)` (log_probs have shape `(n_samples, batch_size)`)
    kls = tf.math.reduce_mean(
        distribution_a.log_prob(a_samples) - distribution_b.log_prob(a_samples), axis=0
    )
    return kls


class GMPrior(tf.Module):
    """Gaussian Mixture (GM) prior.

    The class being a `tf.Module` exposes trainable Variable attributes as `self.trainable_variables`,
    which enables their automatic update through the loss.

    Args:
        latent_dim: latent dimension.
        n_components: number of mixture components.
        softplus_shift: (epsilon) shift to apply after softplus when computing standard deviations.
         The purpose is to stabilize training and prevent NaN probabilities by making sure
         standard deviations of normal distributions are non-zero.
         https://github.com/tensorflow/probability/issues/751 suggests 1e-5.
         https://arxiv.org/pdf/1802.03903.pdf uses 1e-4.
        softplus_scale: scale to apply in the softplus to stabilize training.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_components: int = 16,
        softplus_shift: float = 1e-4,
        softplus_scale: float = 1.0,
    ):
        super().__init__(name="gm_prior")
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.softplus_shift = softplus_shift
        self.softplus_scale = softplus_scale
        # mixture initialization
        self.mixture_weights = tf.Variable(
            tf.zeros(n_components), trainable=True, name="mixture_weights"
        )
        # actual means
        self.means = tf.Variable(
            tf.random.normal([n_components, latent_dim]), trainable=True, name="means"
        )
        # parameters leading to (constrained) standard deviations
        self.std_params = tf.Variable(
            tf.random.normal([n_components, latent_dim]),
            trainable=True,
            name="std_params",
        )

    def get_dist(self):
        return MixtureSameFamily(
            mixture_distribution=Categorical(logits=self.mixture_weights),
            components_distribution=MultivariateNormalDiag(
                loc=self.means,
                scale_diag=self.softplus_shift
                + tf.math.softplus(self.softplus_scale * self.std_params),
            ),
        )

    @tf.function
    def sample(self, sample_shape):
        gm = self.get_dist()
        return gm.sample(sample_shape)

    @tf.function
    def log_prob(self, z):
        gm = self.get_dist()
        return gm.log_prob(z)


class VampPrior(tf.Module):
    def __init__(self, window_size, n_features, latent_dim, n_components, encoder):
        super().__init__(name="vamp_prior")
        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder = encoder
        # pseudo-inputs
        self.u = tf.Variable(
            tf.random.normal([n_components, window_size, n_features]),
            trainable=True,
            name="u",
        )
        self.w = tf.Variable(tf.zeros(n_components), trainable=True, name="w")

    def get_dist(self):
        qz_u_dists = self.encoder(self.u)
        # `qz_u_dists.distribution` is a batch of `(K, L)` univariate Normal distributions
        return MixtureSameFamily(
            mixture_distribution=Categorical(logits=self.w),
            components_distribution=MultivariateNormalDiag(
                loc=qz_u_dists.distribution.loc,
                scale_diag=qz_u_dists.distribution.scale,
            ),
        )

    @tf.function
    def sample(self, sample_shape):
        vamp = self.get_dist()
        return vamp.sample(sample_shape)

    @tf.function
    def log_prob(self, z):
        vamp = self.get_dist()
        return vamp.log_prob(z)
