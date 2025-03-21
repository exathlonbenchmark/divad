""""Human-Interpretable" Dimensionality Reduction (HIDR) module.
"""
import os
import time

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from utils.guarding import (
    check_value_in_choices,
    check_is_not_none,
    check_all_files_exist,
)
from data.helpers import load_files, save_files
from detection.detectors.helpers.tf_helpers import LayerBlock
from detection.detectors.autoencoder import (
    get_autoencoder,
    compile_autoencoder,
)


def fit_autoencoder(
    X,
    model,
    model_name,
    batch_size,
    n_epochs,
    logging_path,
    output_path,
    initial_epoch=0,
):
    """Fits the provided autoencoder model to the provided `X` samples.

    Args:
        X (ndarray): samples to fit the model to, of shape `(n_samples, n_features)`.
        model (keras.model): autoencoder keras model to fit.
        model_name (str): name of the model used for checkpointing (without file extension).
        batch_size (int): batch size to use when fitting the autoencoder.
        n_epochs (int): number of epochs after which to stop training the autoencoder, (knowing
            it was already trained for `initial_epoch` epochs).
        logging_path (str): logging path to use in the tensorboard callback.
        output_path (str): output path to save model checkpoints and tensorboard information to.
        initial_epoch (int): number of epochs the autoencoder was already trained for.
    """
    # project samples on the cluster features and define them as windows of size 1
    tensorboard = TensorBoard(logging_path, histogram_freq=1, write_graph=True)
    # main checkpoint (one per hyperparameters set)
    checkpoint_params = {"monitor": "loss", "save_best_only": True}
    checkpoint_a = ModelCheckpoint(
        os.path.join(output_path, f"{model_name}.h5"), **checkpoint_params
    )
    # backup checkpoint (one per run instance)
    checkpoint_b = ModelCheckpoint(
        os.path.join(logging_path, f"{model_name}.h5"), **checkpoint_params
    )
    callbacks = [tensorboard, checkpoint_a, checkpoint_b]
    model.fit(
        X,
        X,
        epochs=n_epochs,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )


class HIDR:
    """Human-Interpretable Dimensionality Reduction (HIDR) model class.

    This method has two main components:
    - The correlation clusters of the features, used to group feature sets together
        based on their linear correlations.
    - The autoencoder networks (one per cluster), used to derive a 1-dimensional coding
        per feature cluster.

    Args:
        correlations_training: whether to compute correlations on all periods ("all.training")
          or the largest only ("largest.training").
        autoencoders_training: whether the train autoencoders on all data ("global") or trace
          by trace ("trace").
        output_path (str): path to save the models and fitting information to.
    """

    def __init__(
        self,
        correlations_training: str = "largest.training",
        autoencoders_training: str = "trace",
        output_path: str = ".",
    ):
        check_value_in_choices(
            correlations_training,
            "correlations_training",
            ["all.training", "largest.training"],
        )
        check_value_in_choices(
            autoencoders_training,
            "autoencoders_training",
            ["global", "trace"],
        )
        self.correlations_training = correlations_training
        self.autoencoders_training = autoencoders_training
        self.output_path = output_path
        # dictionaries of feature indices and encoders, where keys are cluster numbers (starting from 1)
        self.clustered_features, self.encoders = dict(), dict()

    def fit(self, X, periods_lengths=None):
        """Fits the HIDR model to the provided `X` samples.

        Args:
            X (ndarray): samples to fit the model to, of shape `(n_samples, n_features)`.
            periods_lengths (list|None): ordered list of periods lengths that can be used to
                get the largest period and/or unflatten `X` samples.
        """
        # load or compute the feature pairwise correlation matrix
        if check_all_files_exist(self.output_path, ["ft_correlation_matrix.pkl"]):
            ft_corr_matrix = load_files(
                self.output_path, ["ft_correlation_matrix"], "pickle"
            )["ft_correlation_matrix"]
        else:
            # compute the correlations using either all periods or only the largest one
            if self.correlations_training == "all.undisturbed":
                corr_train = X
            else:
                m = "`periods_lengths` should be provided if fitting correlations to the largest period."
                check_is_not_none(periods_lengths, m)
                max_idx = np.argmax(periods_lengths)
                start_idx = sum(periods_lengths[:max_idx])
                corr_train = X[start_idx : start_idx + periods_lengths[max_idx]]
            ft_corr_matrix = np.nan_to_num(pd.DataFrame(corr_train).corr().values)
            save_files(
                self.output_path, {"ft_correlation_matrix": ft_corr_matrix}, "pickle"
            )

        # load or compute the features grouped by cluster number
        if check_all_files_exist(self.output_path, ["clustered_features.json"]):
            self.clustered_features = load_files(
                self.output_path, ["clustered_features"], "json", parse_keys=True
            )["clustered_features"]
        else:
            # absolute coefficients
            abs_ft_corr_matrix = np.abs(ft_corr_matrix)
            # pairwise euclidean distances between feature correlation vectors
            pairwise_ft_distances = sch.distance.pdist(abs_ft_corr_matrix)
            # hierarchical clustering based on the pairwise distances
            linkage = sch.linkage(pairwise_ft_distances, method="complete")
            # flat feature clusters derived from the hierarchical clustering of correlation vectors
            feature_clusters = sch.fcluster(
                linkage, 0.5 * np.amax(pairwise_ft_distances), "distance"
            )

            # group features by cluster number (cluster numbers start from 1)
            self.clustered_features = dict()
            for ft_idx, cluster in enumerate(feature_clusters):
                # convert `intc` cluster numbers to regular ints
                cluster = int(cluster)
                if cluster not in self.clustered_features:
                    self.clustered_features[cluster] = []
                self.clustered_features[cluster].append(ft_idx)
            save_files(
                self.output_path,
                {"clustered_features": self.clustered_features},
                "json",
            )

        # load or build/compile an autoencoder per cluster
        autoencoders = dict()
        autoencoder_paths = {
            cluster: os.path.join(self.output_path, f"autoencoder_{cluster}.h5")
            for cluster in self.clustered_features
        }
        if all([os.path.exists(path) for path in autoencoder_paths.values()]):
            custom_objects = {"LayerBlock": LayerBlock}
            for cluster in self.clustered_features:
                print(f"loading encoder of cluster {cluster}...", end=" ", flush=True)
                autoencoders[cluster] = load_model(
                    autoencoder_paths[cluster], custom_objects=custom_objects
                )
                self.encoders[cluster] = autoencoders[cluster].get_layer("encoder")
                print("done.")
        else:
            # architecture, optimization and training hyperparameters
            arch_hp = {
                "latent_dim": 1,
                "type_": "dense",
                "enc_n_hidden_neurons": [],
                "dec_last_activation": "elu",
                "dense_layers_activation": "elu",
            }
            opt_hp = {
                "loss": "mse",
                "optimizer": "adadelta",
                "adamw_weight_decay": 0.0,
                "learning_rate": 0.001,
            }
            train_hp = {
                "n_epochs": 200 if self.autoencoders_training == "global" else 20,
                "batch_size": 256,
            }
            for cluster, ft_ids in self.clustered_features.items():
                # define a deeper architecture for higher dimensional input spaces
                n_features = len(ft_ids)
                arch_hp["enc_n_hidden_neurons"] = (
                    [(2 * n_features) // 3] if n_features >= 1000 else []
                )
                autoencoders[cluster] = get_autoencoder(
                    window_size=1, n_features=n_features, **arch_hp
                )
                compile_autoencoder(autoencoders[cluster], **opt_hp)
                self.encoders[cluster] = autoencoders[cluster].get_layer("encoder")

            # fit each autoencoder (either globally or trace by trace)
            if self.autoencoders_training == "global":
                for cluster, ft_ids in self.clustered_features.items():
                    model_name = f"autoencoder_{cluster}"
                    logging_path = os.path.join(
                        self.output_path,
                        f"{model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S')}",
                    )
                    print(f"## Fitting Cluster {cluster} Autoencoder ##")
                    # project samples on the cluster features and define them as windows of size 1
                    cluster_X = X[:, ft_ids].reshape((len(X), 1, len(ft_ids)))
                    fit_autoencoder(
                        cluster_X,
                        autoencoders[cluster],
                        model_name,
                        train_hp["batch_size"],
                        train_hp["n_epochs"],
                        logging_path,
                        self.output_path,
                    )
            else:
                m = "`periods_lengths` should be provided if fitting autoencoders trace by trace."
                check_is_not_none(periods_lengths, m)
                cursor = 0
                cluster_to_logging, cluster_to_name = dict(), dict()
                for i, p_len in enumerate(periods_lengths):
                    # number of epochs the model was previously trained for
                    initial_epochs = i * train_hp["n_epochs"]
                    print(f"## Fitting Autoencoders to Period {i+1} ##")
                    for cluster, ft_ids in self.clustered_features.items():
                        if cluster not in cluster_to_logging:
                            cluster_to_name[cluster] = f"autoencoder_{cluster}"
                            cluster_to_logging[cluster] = os.path.join(
                                self.output_path,
                                f"{cluster_to_name[cluster]}_{time.strftime('%Y_%m_%d-%H_%M_%S')}",
                            )
                        print(f"# Fitting Cluster {cluster} Autoencoder #")
                        # project period samples on the cluster features and define them as windows of size 1
                        cluster_X = X[cursor : cursor + p_len, ft_ids].reshape(
                            (p_len, 1, len(ft_ids))
                        )
                        fit_autoencoder(
                            cluster_X,
                            autoencoders[cluster],
                            cluster_to_name[cluster],
                            train_hp["batch_size"],
                            initial_epochs + train_hp["n_epochs"],
                            cluster_to_logging[cluster],
                            self.output_path,
                            initial_epoch=initial_epochs,
                        )
                    cursor += p_len

    def transform(self, X):
        """Returns the samples of `X` transformed using the HIDR encoders.

        Args:
            X (ndarray): input samples to transform, of shape `(n_samples, n_features)`.

        Returns:
            ndarray: the transformed `X` samples.
        """
        transformed = np.zeros((X.shape[0], len(self.clustered_features.keys())))
        for cluster, ft_ids in self.clustered_features.items():
            # project samples on the cluster features and define them as windows of size 1
            cluster_X = X[:, ft_ids].reshape((len(X), 1, len(ft_ids)))
            transformed[:, cluster - 1] = (
                self.encoders[cluster].predict(cluster_X).reshape(X.shape[0])
            )
        return transformed
