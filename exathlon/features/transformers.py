"""Features transformation module, gathering all Transformer classes.
"""
import os
import functools
from abc import abstractmethod
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis

from utils.data import TRAIN_SET_NAME
from utils.guarding import check_value_in_choices, check_is_percentage
from data.helpers import save_files
from features.hidr import HIDR


def get_scaler(
    type_: str = "std", minmax_range: Optional[list] = None
) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
    """Returns the Scaler object corresponding to the provided scaling method.

    Args:
        type_: type of scaler (must be either "std", "minmax" or "robust").
        minmax_range: optional output range if minmax scaling (set to `[0, 1]` if needed
          and not provided).

    Returns:
        Scaler object.
    """
    check_value_in_choices(type_, "type_", ["std", "minmax", "robust"])
    if type_ == "std":
        return StandardScaler()
    if type_ == "minmax":
        return MinMaxScaler(
            feature_range=(0, 1) if minmax_range is None else tuple(minmax_range),
            clip=False,
        )
    return RobustScaler()


class BaseTransformer:
    """Base Transformer class.

    Provides methods for fitting and transforming period arrays.

    Techniques directly extending this class are assumed to fit their model on undisturbed traces
    and apply it to all. The `model_training` argument then specifies whether the model is fit to
    *all* undisturbed traces or only the largest one.

    Args:
        transform_fit_normal_only: whether to fit the transformer to normal data only.
        model_name: name/prefix of the trained model file(s) to save.
        model_training: either "all_training" or "largest_training".
        output_path: path to save the model(s) and fitting information to.
    """

    def __init__(
        self,
        transform_fit_normal_only: bool = True,
        model_name: str = "transformer",
        model_training: Optional[str] = "all_training",
        output_path: str = ".",
    ):
        check_value_in_choices(
            model_training, "model_training", ["all_training", "largest_training"]
        )
        self.fit_normal_only = transform_fit_normal_only
        self.model = None
        self.model_name = model_name
        self.model_training = model_training
        self.output_path = output_path

    def fit_transform_datasets(self, datasets, datasets_labels, datasets_info):
        """Returns the provided datasets transformed by (a) transformation model(s).

        By default, a single model is fit to training periods and applied to all datasets.

        Args:
            datasets (dict): of the form {`set_name`: `periods`};
                with `periods` a `(n_periods, period_length, n_features)` array, `period_length`
                depending on the period.
            datasets_labels (dict): datasets labels of the form {`set_name`: `periods_labels`}.
            datasets_info (dict): of the form {`set_name`: `periods_info`};
                with `periods_info` a list of the form `[file_name, trace_type, period_rank]`
                for each period of the set.

        Returns:
            dict: the transformed datasets, in the same format.
        """
        transformed = dict()
        transformed[TRAIN_SET_NAME] = self.fit_transform(
            datasets[TRAIN_SET_NAME], datasets_labels[TRAIN_SET_NAME], self.model_name
        )
        for ds_name in [n for n in list(datasets.keys()) if n != TRAIN_SET_NAME]:
            transformed[ds_name] = self.transform(datasets[ds_name])
        return transformed

    def fit(self, periods, periods_labels, model_file_name):
        """Fits the transformer's model to the provided `periods` array and saves it.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                `period_length` depending on the period.
            periods_labels (ndarray): shape (n_periods, period_length)`.
            model_file_name (str): name of the model file to save.
        """
        # fit and save the transformation model on concatenated periods records
        if self.model_training == "all_training":
            concatenated_periods = np.concatenate(periods, axis=0)
            fitting_mask = (
                np.concatenate(periods_labels, axis=0) == 0
                if self.fit_normal_only
                else np.ones(concatenated_periods.shape[0], dtype=bool)
            )
            self.model.fit(concatenated_periods[fitting_mask])
        else:
            largest_period_idx = np.argmax([len(p) for p in periods])
            largest_period = periods[largest_period_idx]
            largest_period_labels = periods_labels[largest_period_idx]
            fitting_mask = (
                largest_period_labels == 0
                if self.fit_normal_only
                else np.ones(largest_period.shape[0], dtype=bool)
            )
            self.model.fit(largest_period[fitting_mask])
        # save the transformation model as a pickle file (if can be directly pickled, else save within the class)
        try:
            save_files(self.output_path, {model_file_name: self.model}, "pickle")
        except TypeError:
            print(
                "Warning: the model could not be saved through the Transformer (ignore if saved elsewhere)"
            )
            os.remove(os.path.join(self.output_path, f"{model_file_name}.pkl"))

    def transform(self, periods):
        """Returns the provided periods transformed by the transformer's model.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                `period_length` depending on the period.

        Returns:
            ndarray: the periods transformed by the model, in the same format.
        """
        # get transformed concatenated periods data
        transformed = self.model.transform(np.concatenate(periods, axis=0))
        # return the transformed periods back as a 3d-array
        return unravel_periods(transformed, [period.shape[0] for period in periods])

    def fit_transform(self, periods, periods_labels, model_file_name):
        """Fits the transformer's model to the provided periods and returns the transformed periods.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                `period_length` depending on the period.
            periods_labels (ndarray): shape (n_periods, period_length,)`.
            model_file_name (str): name of the model file to save.

        Returns:
            ndarray: the periods transformed by the model, in the same format.
        """
        self.fit(periods, periods_labels, model_file_name)
        return self.transform(periods)


class OneHotAppAdder(BaseTransformer):
    """One-Hot App adder.

    Adds one-hot application ID to the input features.
    """

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_name = "one_hot_app_adder"
        sparse_kwarg = (
            {"sparse": False}
            if sklearn.__version__ == "1.0.2"
            else {"sparse_output": False}
        )
        self.model = OneHotEncoder(dtype=np.float32, **sparse_kwarg)

    def fit_transform_datasets(self, datasets, datasets_labels, datasets_info):
        set_to_seq_app_ids = dict()
        for set_name, set_seqs_info in datasets_info.items():
            set_to_seq_app_ids[set_name] = []
            for i, seq_info in enumerate(set_seqs_info):
                seq_app_id = float(seq_info[0].split("_")[0])
                seq_app_ids = np.repeat(seq_app_id, datasets[set_name][i].shape[0])
                set_to_seq_app_ids[set_name].append(seq_app_ids.reshape(-1, 1))
        train_app_ids = np.concatenate(set_to_seq_app_ids[TRAIN_SET_NAME], axis=0)
        self.model.fit(train_app_ids)
        # save the transformation model as a pickle file (if can be directly pickled, else save within the class)
        save_files(self.output_path, {self.model_name: self.model}, "pickle")
        ohe_categories = [c for c in self.model.categories_[0]]
        save_files(self.output_path, {"categories": ohe_categories}, "json")
        transformed = dict()
        for set_name, set_seqs in datasets.items():
            transformed_set_seqs = []
            for i, seq in enumerate(set_seqs):
                seq_app_ids = set_to_seq_app_ids[set_name][i]
                transformed_app_ids = self.model.transform(seq_app_ids)
                extended_seq = np.concatenate([seq, transformed_app_ids], axis=1)
                transformed_set_seqs.append(extended_seq)
            transformed[set_name] = np.array(transformed_set_seqs, dtype=object)
        return transformed


class RegularKernelPCA(BaseTransformer):
    """RegularKernelPCA class.

    A single kernel PCA model is fit to training periods and applied to all.

    Args:
        n_components: number of components (as a valid arg to `sklearn.decomposition.PCA`
          or `sklearn.decomposition.KernelPCA`).
        kernel: kernel to use.
        **base_kwargs: keyword arguments of `BaseTransformer`.
    """

    def __init__(
        self,
        n_components: Union[int, float] = 10,
        kernel: str = "linear",
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.model_name = "pca"
        self.n_components = n_components
        self.kernel = kernel
        # we use the standard PCA class for a linear kernel to more easily plot the evolution of explained variance
        if self.kernel == "linear":
            self.model = PCA(n_components=self.n_components, svd_solver="full")
        else:
            self.model = KernelPCA(n_components=self.n_components, kernel=self.kernel)

    def fit(self, periods, periods_labels, model_file_name):
        """Overrides Transformer's method to save an explained variance evolution figure.

        Note: this figure is only saved if not using an explicit number of output components.
        """
        super().fit(periods, periods_labels, model_file_name)
        if not isinstance(self.n_components, int) and isinstance(self.model, PCA):
            plot_explained_variance(self.model, self.n_components, self.output_path)


class RegularFactorAnalysis(BaseTransformer):
    """RegularFactorAnalysis class.

    A single Factor Analysis (FA) model is fit to training periods and applied to all.

    Args:
        n_components: number of components (as a valid arg to `sklearn.decomposition.FactorAnalysis`).
        **base_kwargs: keyword arguments of `BaseTransformer`.
    """

    def __init__(self, n_components: Union[int, float] = 10, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model_name = "fa"
        self.model = FactorAnalysis(n_components=n_components)


class RegularScaler(BaseTransformer):
    """RegularScaler class.

    A single scaler model is fit to training periods and applied to all.

    Args:
        type_: type of scaler (must be either "std", "minmax" or "robust").
        minmax_range: optional output range if minmax scaling (set to `[0, 1]` if needed
          and not provided).
        **base_kwargs: keyword arguments of `BaseTransformer`.
    """

    def __init__(
        self, type_: str = "std", minmax_range: Optional[list] = None, **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.model_name = "scaler"
        self.model = get_scaler(type_, minmax_range)


class RegularHIDR(BaseTransformer):
    """RegularHIDR class.

    A "Human-Interpretable" Dimensionality Reduction (HIDR) model is fit to training
    periods and applied to all.

    Args:
        correlations_training: whether to compute correlations on all periods ("all_training")
          or the largest only ("largest_training").
        autoencoders_training: whether the train autoencoders on all data ("global") or trace
          by trace ("trace").
        **base_kwargs: keyword arguments of `BaseTransformer`.
    """

    def __init__(
        self,
        correlations_training: str = "largest_training",
        autoencoders_training: str = "trace",
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.model_name = "hidr"
        self.model = HIDR(
            correlations_training, autoencoders_training, self.output_path
        )

    def fit(self, periods, periods_labels, model_file_name):
        """Overrides Transformer's method to pass periods lengths to the model fitting method.

        Note: we also remove the model saving part since its components are saved within the model class.
        """
        # fit and save the transformation model on concatenated period records
        concatenated_periods = np.concatenate(periods, axis=0)
        if self.fit_normal_only:
            periods_fitting_masks = [labels == 0 for labels in periods_labels]
            period_lengths = [sum(mask) for mask in periods_fitting_masks]
            fitting_mask = np.concatenate(periods_fitting_masks, axis=0)
        else:
            fitting_mask = np.ones(concatenated_periods.shape[0], dtype=bool)
            period_lengths = [p.shape[0] for p in periods]
        self.model.fit(concatenated_periods[fitting_mask], period_lengths)


class TraceTransformer(BaseTransformer):
    """Base TraceTransformer class.

    Instead of fitting and saving a single model to all training periods, a model
    is fit and saved per trace file.

    Periods from all datasets are grouped by file name and chronologically sorted within
    each group.

    Args:
        **base_kwargs: keyword arguments of `BaseTransformer`.
    """

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def fit_transform_datasets(self, datasets, datasets_labels, datasets_info):
        """Overrides Transformer's method to fit a model per trace file."""
        # group periods by file name according to their chronological rank
        periods_by_file, labels_by_file = dict(), dict()
        for set_name in datasets:
            for period, period_labels, (file_name, _, p_rank) in zip(
                datasets[set_name], datasets_labels[set_name], datasets_info[set_name]
            ):
                # the chronological rank of a period corresponds to its index in the group
                if file_name in periods_by_file:
                    # the group already contains at least one period
                    cur_group_length = len(periods_by_file[file_name])
                    if p_rank == cur_group_length:
                        periods_by_file[file_name].append(period)
                        labels_by_file[file_name].append(period_labels)
                    else:
                        if p_rank > cur_group_length:
                            n_added = (p_rank - cur_group_length) + 1
                            periods_by_file[file_name] += [None for _ in range(n_added)]
                            labels_by_file[file_name] += [None for _ in range(n_added)]
                        periods_by_file[file_name][p_rank] = period
                        labels_by_file[file_name][p_rank] = period_labels
                else:
                    # the group is currently empty
                    periods_by_file[file_name] = [None for _ in range(p_rank + 1)]
                    labels_by_file[file_name] = [None for _ in range(p_rank + 1)]
                    periods_by_file[file_name][p_rank] = period
                    labels_by_file[file_name][p_rank] = period_labels
        # convert each group to an ndarray
        for file_name in periods_by_file:
            periods_by_file[file_name] = np.array(
                periods_by_file[file_name], dtype=object
            )
            labels_by_file[file_name] = np.array(
                labels_by_file[file_name], dtype=object
            )

        # transform periods by file name
        transformed_by_file = dict()
        for file_name in periods_by_file:
            transformed_by_file[file_name] = self.fit_transform_trace(
                periods_by_file[file_name], labels_by_file[file_name], file_name
            )

        # return transformed periods back in the original `datasets` format
        transformed = {k: [] for k in datasets}
        for set_name in datasets:
            # each dataset period can be recovered using its file name and chronological rank
            for file_name, _, p_rank in datasets_info[set_name]:
                transformed[set_name].append(
                    transformed_by_file[file_name][p_rank].astype(float)
                )
            transformed[set_name] = np.array(transformed[set_name], dtype=object)
        return transformed

    @abstractmethod
    def fit_transform_trace(self, trace_periods, trace_periods_labels, file_name):
        """Fits a model and transforms the provided trace periods.

        Args:
            trace_periods (ndarray): chronologically sorted trace periods to transform.
            trace_periods_labels (ndarray): corresponding trace periods labels.
            file_name (str): trace file name, used when saving the trained model.

        Returns:
            ndarray: the transformed trace periods in the same format.
        """


class FullTraceTransformer(TraceTransformer):
    """FullTraceTransformer class. All trace records are used to fit each file's model.

    /!\\ With this method, we assume all data of a given file to be available upfront.
    Applying it could therefore be unrealistic in an online setting.

    Another drawback of this method is that *all test data will be used for
    training the transformation model*, which may lead to overestimating the
    subsequent model's performance on the test set, unless we assume that traces
    encountered in production will be in the continuity or statistically similar
    to the ones used in this pipeline.

    Args:
        **base_kwargs: keyword arguments of `TraceTransformer`.
    """

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def fit_transform_trace(self, trace_periods, trace_periods_labels, file_name):
        """The trace periods are fit-transformed in batch mode."""
        return self.fit_transform(
            trace_periods, trace_periods_labels, f"{self.model_name}_{file_name}"
        )


class FullTraceScaler(FullTraceTransformer):
    """FullTraceScaler class.

    Traces are rescaled separately using a model trained on all their records.

    Args:
        type_: type of scaler (must be either "std", "minmax" or "robust").
        minmax_range: optional output range if minmax scaling (set to `[0, 1]` if needed
          and not provided).
        **base_kwargs: keyword arguments of `FullTraceTransformer`.
    """

    def __init__(
        self, type_: str = "std", minmax_range: Optional[list] = None, **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.model_name = "scaler"
        self.model = get_scaler(type_, minmax_range)


class HeadTransformer(TraceTransformer):
    """HeadTransformer class. Only head records are used to fit each trace file's model.

    With this method, we only assume `head_size` records to be available upfront for each
    trace to fit their transformation model.

    We also add the option of fit-transforming all training traces, and use the resulting
    model as a weighted pre-trained model for fit-transforming test traces.

    Args:
        head_size: number of records used at the beginning of each trace to fit their transformer.
        regular_pretraining_weight: weight of regular pretraining to add to online training.
        **base_kwargs: keyword arguments of `TraceTransformer`.
    """

    def __init__(
        self,
        head_size: int = 1800,
        regular_pretraining_weight: float = 0.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        check_is_percentage(regular_pretraining_weight, "regular_pretraining_weight")
        self.head_size = head_size
        self.regular_model = None
        self.regular_pretraining_weight = regular_pretraining_weight

    def fit_transform_datasets(self, datasets, datasets_labels, datasets_info):
        """Overrides TraceTransformer's method to fit a model per trace file's "head".

        We also add the option of fit-transforming all training traces, and use the resulting
        model as a weighted pre-trained model for fit-transforming test traces.
        """
        # using a regular transformation model pretraining
        if self.regular_pretraining_weight > 0.0:
            transformed = dict()
            # train a model on the training periods and transform them in batch mode
            transformed[TRAIN_SET_NAME] = self.fit_transform(
                datasets[TRAIN_SET_NAME],
                datasets_labels[TRAIN_SET_NAME],
                self.model_name,
            )
            self.regular_model = deepcopy(self.model)
            # transform test periods fitting their heads on top of this pre-trained model
            disturbed = dict()
            for key, item in zip(
                ["data", "labels", "info"], [datasets, datasets_labels, datasets_info]
            ):
                disturbed[key] = {n: v for n, v in item.items() if n != TRAIN_SET_NAME}
            # get transformed periods using `TraceTransformer.fit_transform_datasets`
            disturbed["transformed"] = super().fit_transform_datasets(
                disturbed["data"], disturbed["labels"], disturbed["info"]
            )
            for n in disturbed["transformed"]:
                transformed[n] = disturbed["transformed"][n]
        else:
            # get all transformed periods using `TraceTransformer.fit_transform_datasets`
            transformed = super().fit_transform_datasets(
                datasets, datasets_labels, datasets_info
            )
        return transformed

    def fit_transform_trace(self, trace_periods, trace_periods_labels, file_name):
        """Transforms trace periods using a model fit to its first `head_size` records."""
        head_size = self.get_head_size(trace_periods_labels)
        self.fit_trace_head(trace_periods, file_name, head_size=head_size)
        return self.transform(trace_periods)

    def get_head_size(self, trace_periods_labels):
        """Returns the head size to use based on `trace_periods_labels`.

        Args:
            trace_periods_labels (ndarray): chronologically sorted trace periods labels.

        Returns:
            int: head size to use, defined as `min(first_ano_start_idx, self.head_size)`.
        """
        trace_labels = np.concatenate(trace_periods_labels, axis=0)
        anomaly_ids = np.where(trace_labels > 0)[0]
        first_ano_start_idx = anomaly_ids[0] if anomaly_ids.shape[0] > 0 else np.inf
        return min(first_ano_start_idx, self.head_size)

    def fit_trace_head(self, trace_periods, file_name, head_size=None):
        """Fits a transformer model to the `head_size` first records of the trace periods.

        Trace periods will be concatenated, as if they were a single trace. An array of periods
        has to be provided in case the first trace period is shorter than `head_size`.

        If using a "regular model pretraining", the model fit to the trace's head
        will be combined with a model priorly trained on the `train` periods.

        Args:
            trace_periods (ndarray): the trace periods to fit the model to.
            file_name (str): the name of the trace file, used when saving the model.
            head_size (int): optional head size to use instead of `self.head_size`.
        """
        if head_size is None:
            head_size = self.head_size
        trace_head = np.concatenate(trace_periods, axis=0)[:head_size, :]
        trace_head_labels = np.zeros(trace_head.shape[0])  # head is always normal
        # consider the trace head as a single period to fit the model to
        self.fit(
            periods=np.array([trace_head]),
            periods_labels=np.array([trace_head_labels]),
            model_file_name=f"{self.model_name}_{file_name}",
        )
        # combine the fit model with the one fit on `train` if using regular pretraining
        if self.regular_pretraining_weight > 0.0:
            self.combine_regular_head()

    @abstractmethod
    def combine_regular_head(self):
        """Combines the models fit on training periods and fit on a period's head.

        The model fit on train periods is available as `self.regular_model`.
        The model fit on the head of the current period is available as `self.model`.
        The combined model should be `self.model` modified inplace.
        """


class HeadScaler(HeadTransformer):
    """HeadScaler class.

    Implements `HeadTransformer` with features rescaling as the transformation.

    Args:
        type_: type of scaler (must be either "std", "minmax" or "robust").
        minmax_range: optional output range if minmax scaling (set to `[0, 1]` if needed
          and not provided).
        **base_kwargs: keyword arguments of `HeadTransformer`.
    """

    def __init__(
        self, type_: str = "std", minmax_range: Optional[list] = None, **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.model_name = "scaler"
        self.model = get_scaler(type_, minmax_range)
        # model pretrained on training periods if relevant
        if self.regular_pretraining_weight > 0.0:
            self.regular_model = get_scaler(type_, minmax_range)

    def combine_regular_head(self):
        """Combines the models fit on training periods and fit on a period's head."""
        # combine standard scalers
        if isinstance(self.model, StandardScaler):
            # the mean and std are combined using convex combinations
            regular_mean, regular_std = (
                self.regular_model.mean_,
                self.regular_model.scale_,
            )
            head_mean, head_std = self.model.mean_, self.model.scale_
            combined_mean = (
                self.regular_pretraining_weight * regular_mean
                + (1 - self.regular_pretraining_weight) * head_mean
            )
            combined_std = (
                self.regular_pretraining_weight * regular_std
                + (1 - self.regular_pretraining_weight) * head_std
            )
            combined_var = combined_std**2
            self.model.mean_ = combined_mean
            self.model.scale_ = combined_std
            self.model.var_ = combined_var
        # combine minmax scalers
        else:
            # the min and max are combined keeping the min and max values only
            regular_min, regular_max = (
                self.regular_model.data_min_,
                self.regular_model.data_max_,
            )
            head_min, head_max = self.model.data_min_, self.model.data_max_
            combined_min = np.array(
                [min(regular_min[i], head_min[i]) for i in range(regular_min.shape[0])]
            )
            combined_max = np.array(
                [max(regular_max[i], head_max[i]) for i in range(regular_max.shape[0])]
            )
            combined_range = combined_max - combined_min
            self.model.data_min_ = combined_min
            self.model.data_max_ = combined_max
            self.model.data_range_ = combined_range


class HeadOnlineTransformer(HeadTransformer):
    """HeadOnlineTransformer class. The traces are also transformed online in addition to the head model.

    Each trace's transformer is fit to its first `head_size` records in batch mode, then
    incrementally updated using an expanding or rolling window to transform newly arriving records.

    Using this method, we enable the transformer models to adapt to changing statistics
    within traces.

    Note: if a rolling window is used, its size will be fixed to `head_size`.

    Note: like for simple head transformation, we add the option of fit-transforming
    all training traces, and use the resulting model as a weighted pre-trained model
    for fit-transforming test traces.

    Args:
        online_window_type: whether to use an "expanding" or "rolling" window when
          using head-online scaling.
        **base_kwargs: keyword arguments of `HeadTransformer`.
    """

    def __init__(self, online_window_type: str = "expanding", **base_kwargs):
        super().__init__(**base_kwargs)
        check_value_in_choices(
            online_window_type, "online_window_type", ["expanding", "rolling"]
        )
        self.online_window_type = online_window_type

    def fit_transform_trace(self, trace_periods, trace_periods_labels, file_name):
        """Fit-transforms the trace in batch mode for its first `head_size` records, online for others."""
        # fit a model to the head of the trace (possibly combined with a prior regular model)
        head_size = self.get_head_size(trace_periods_labels)
        self.fit_trace_head(trace_periods, file_name, head_size=head_size)
        # transform trace periods online
        transformed_online = self.transform_trace_online(
            trace_periods, head_size=head_size
        )
        # replace the first `head_size` transformations with the output of the trained model
        flattened_transformed = np.concatenate(transformed_online, axis=0)
        flattened_transformed[:head_size, :] = self.transform(
            np.array([np.concatenate(trace_periods, axis=0)[:head_size, :]])
        )[0]
        return unravel_periods(
            flattened_transformed, [period.shape[0] for period in trace_periods]
        )

    @abstractmethod
    def transform_trace_online(self, trace_periods, head_size=None):
        """Fits and transforms trace periods using expanding/rolling window estimates.

        /!\\ The transformed periods will be returned with the same number of records,
        with features that could not be transformed being replaced with `None` values.

        Args:
            trace_periods (ndarray): chronologically sorted trace periods to transform online.
            head_size (int): optional head size to use instead of `self.head_size`.

        Returns:
             ndarray: the transformed trace periods in the same format.
        """


class HeadOnlineScaler(HeadOnlineTransformer):
    """HeadOnlineScaler class.

    Implements `HeadOnlineTransformer` with features rescaling as the transformation.

    Args:
        type_: type of scaler (must be either "std", "minmax" or "robust").
        minmax_range: optional output range if minmax scaling (set to `[0, 1]` if needed
          and not provided).
        **base_kwargs: keyword arguments of `HeadOnlineTransformer`.
    """

    def __init__(
        self, type_: str = "std", minmax_range: Optional[list] = None, **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.model_name = "scaler"
        self.model = get_scaler(type_, minmax_range)
        # model pretrained on `train` periods if relevant
        if self.regular_pretraining_weight > 0.0:
            self.regular_model = get_scaler(type_, minmax_range)

    def transform_trace_online(self, trace_periods, head_size=None):
        """Transforms trace periods using expanding/rolling window estimates for the offset and scale.

        We only consider Standard and MinMax scalers here. If a value of 0 is encountered
        for the scaling factor, it will be replaced by 1.
        """
        if head_size is None:
            head_size = self.head_size
        a_1 = "supported online scalers are currently limited to `std` and `minmax`"
        assert isinstance(self.model, MinMaxScaler) or isinstance(
            self.model, StandardScaler
        ), a_1
        # derive a whole trace DataFrame from its sorted period arrays
        trace_df = pd.DataFrame(np.concatenate(trace_periods, axis=0))
        # define the windowing method based on the online window type
        a_2 = "supported online window types only include expanding and rolling"
        assert self.online_window_type in ["expanding", "rolling"], a_2
        if self.online_window_type == "expanding":
            trace_windowing = functools.partial(trace_df.expanding)
        else:
            trace_windowing = functools.partial(trace_df.rolling, head_size)
        if isinstance(self.model, MinMaxScaler):
            min_df, max_df = trace_windowing().min().shift(
                1
            ), trace_windowing().max().shift(1)
            offset_df, scale_df = min_df, (max_df - min_df)
        else:
            offset_df, scale_df = trace_windowing().mean().shift(
                1
            ), trace_windowing().std().shift(1)
        transformed_df = (trace_df - offset_df) / scale_df.replace(0, 1)
        return unravel_periods(
            transformed_df.values, [period.shape[0] for period in trace_periods]
        )

    def combine_regular_head(self):
        """Combines the models fit on training periods and fit on a period's head."""
        # combine standard scalers
        if isinstance(self.model, StandardScaler):
            # the mean and std are combined using convex combinations
            regular_mean, regular_std = (
                self.regular_model.mean_,
                self.regular_model.scale_,
            )
            head_mean, head_std = self.model.mean_, self.model.scale_
            combined_mean = (
                self.regular_pretraining_weight * regular_mean
                + (1 - self.regular_pretraining_weight) * head_mean
            )
            combined_std = (
                self.regular_pretraining_weight * regular_std
                + (1 - self.regular_pretraining_weight) * head_std
            )
            combined_var = combined_std**2
            self.model.mean_ = combined_mean
            self.model.scale_ = combined_std
            self.model.var_ = combined_var
        # combine minmax scalers
        else:
            # the min and max are combined keeping the min and max values only
            regular_min, regular_max = (
                self.regular_model.data_min_,
                self.regular_model.data_max_,
            )
            head_min, head_max = self.model.data_min_, self.model.data_max_
            combined_min = np.array(
                [min(regular_min[i], head_min[i]) for i in range(regular_min.shape[0])]
            )
            combined_max = np.array(
                [max(regular_max[i], head_max[i]) for i in range(regular_max.shape[0])]
            )
            combined_range = combined_max - combined_min
            self.model.data_min_ = combined_min
            self.model.data_max_ = combined_max
            self.model.data_range_ = combined_range


def unravel_periods(raveled, period_lengths):
    """Returns the unraveled equivalent of `raveled`, separating back contiguous periods.

    Args:
        raveled (ndarray): raveled array of shape `(n_records, n_features)`.
        period_lengths (list): lengths of the contiguous periods to extract from the raveled array.

    Returns:
        ndarray: unraveled array of shape `(n_periods, period_size, n_features)` where `period_size`
            possibly depends on the period.
    """
    start_idx = 0
    unraveled = []
    for period_length in period_lengths:
        unraveled.append(raveled[start_idx : start_idx + period_length])
        start_idx += period_length
    if len(unraveled) == 1 or len(set([a.shape for a in unraveled])) == 1:
        # return with original data type
        return np.array(unraveled)
    # else return with data type "object"
    return np.array(unraveled, dtype=object)


def plot_explained_variance(pca_model, explained_variance, output_path):
    """Plots the data variance explained by the model with respect to the number of output components.

    We also show the actual number of components that has been kept by the performed transformation.

    Args:
        pca_model (sklearn.decomposition.PCA): the PCA object to describe, already fit to the training data.
        explained_variance (float): the actual amount of explained variance that was specified.
        output_path (str): the path to save the figure to.
    """
    dim = pca_model.n_components_
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
    plt.plot(
        [dim],
        [explained_variance],
        marker="o",
        markersize=6,
        color="red",
        label=f"Reduced to {dim} dimensions",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Variance (%)")
    plt.title("Explained Variance vs. Dimensionality")
    plt.grid(True)
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "variance_vs_dimension.png"))
    plt.close()


# dictionary gathering references to the defined transformation classes
transformation_classes = {
    "regular_scaling": RegularScaler,
    "trace_scaling": FullTraceScaler,
    "head_scaling": HeadScaler,
    "head_online_scaling": HeadOnlineScaler,
    "pca": RegularKernelPCA,
    "fa": RegularFactorAnalysis,
    "hidr": RegularHIDR,
    "oh_app_ext": OneHotAppAdder,
}
