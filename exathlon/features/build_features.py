"""Features building module.

Turns the raw period columns to the final features that will be used by the models.
If specified, this pipeline step can also resample the records and/or labels to new sampling periods.
"""
import os
import logging
import importlib
from omegaconf import DictConfig

from utils.guarding import check_value_in_choices
from utils.data import get_dataset_names
from data.helpers import (
    load_files,
    save_files,
    extract_save_labels,
    resample_sequences,
    get_numpy_from_dfs,
    get_dfs_from_numpy,
)
from features.transformers import transformation_classes, FullTraceScaler, HeadScaler


def main(cfg: DictConfig, step_to_out_path: dict) -> None:
    logging.info(cfg)
    logging.info(step_to_out_path)
    make_datasets_path = step_to_out_path["make_datasets"]
    build_features_path = step_to_out_path["build_features"]
    dataset_names = get_dataset_names(make_datasets_path)
    datasets = load_files(make_datasets_path, dataset_names, "pickle")

    # extract, optionally downsample, and save labels from the "anomaly" columns of the sequences
    os.makedirs(build_features_path, exist_ok=True)
    original_sampling_period = cfg.dataset.sampling_period
    labels_sampling_period = cfg.build_features.labels_sampling_period
    labels_downsampling = labels_sampling_period != original_sampling_period
    if labels_downsampling:
        logging_str = "Downsampling and saving"
    else:
        labels_sampling_period = None
        logging_str = "Saving"
    logging.info(f"{logging_str} labels to {build_features_path}.")
    for k in datasets:
        extract_save_labels(
            datasets[k],
            f"y_{k}",
            build_features_path,
            sampling_period=labels_sampling_period,
            original_sampling_period=original_sampling_period,
        )

    # resample sequences before doing anything if specified and relevant
    data_sampling_period = cfg.build_features.data_sampling_period
    data_downsampling = data_sampling_period != original_sampling_period
    data_downsampling_position = None
    if data_downsampling:
        data_downsampling_position = cfg.build_features.data_downsampling_position
        check_value_in_choices(
            data_downsampling_position,
            "data_downsampling_position",
            ["first", "middle", "last"],
        )
    if data_downsampling and data_downsampling_position == "first":
        for k in datasets:
            resample_sequences(
                datasets[k],
                data_sampling_period,
                anomaly_col=False,
                original_sampling_period=original_sampling_period,
            )
    # optional features alteration bundle
    if cfg.build_features.feature_crafter.bundle_idx != -1:
        crafter_module = importlib.import_module(
            f"features.crafters.{cfg.dataset.name}_crafter"
        )
        feature_crafter = getattr(
            crafter_module, f"{cfg.dataset.name.capitalize()}Crafter"
        )(**cfg.build_features.feature_crafter)
        datasets = feature_crafter.get_altered_features(datasets)

    input_feature_names = list(datasets[dataset_names[-1]][0].columns)
    output_feature_names = None

    # resample periods after alteration but before transformation if specified and relevant
    if data_downsampling and data_downsampling_position == "middle":
        for k in datasets:
            resample_sequences(
                datasets[k],
                data_sampling_period,
                anomaly_col=False,
                original_sampling_period=original_sampling_period,
            )

    # optional features transformation chain
    datasets_info = load_files(
        make_datasets_path,
        [f"{n}_info" for n in dataset_names],
        "pickle",
        drop_info_suffix=True,
    )
    datasets_labels = load_files(
        build_features_path,
        [f"y_{n}" for n in dataset_names],
        "numpy",
        drop_labels_prefix=True,
    )
    # turn datasets to `(n_periods, period_size, n_features)` ndarrays
    logging.info("Converting datasets to numpy arrays...")
    for k in datasets:
        datasets[k] = get_numpy_from_dfs(datasets[k])
    logging.info("Done.")
    transform_chain = cfg.build_features.transform_chain
    for transform_step in [ts for ts in transform_chain.split(".") if len(ts) > 0]:
        logging.info(f'Applying "{transform_step}" to period features...')
        if transform_step != "trace_head_scaling":
            # TODO: /!\ `transform_fit_normal_only` is never considered and always True by default.
            if transform_step in cfg.build_features:
                transformer_kwargs = cfg.build_features[transform_step]
            else:
                transformer_kwargs = dict()
            transformer = transformation_classes[transform_step](
                **transformer_kwargs, output_path=build_features_path
            )
            # TODO: handle this better (should have one large array instead of a list).
            #  should include the full name in the transform chain: should rename trace_head_scaling
            #  to trace_head_scaler, and a trace_head_scaler is not a scaler but a trace_head_scaler.
            datasets = transformer.fit_transform_datasets(
                datasets, datasets_labels, datasets_info
            )
        else:
            # full trace scaling for train, head scaling for (val and) test
            transformers = []
            for k, t in zip(
                ["trace_scaling", "head_scaling", "head_scaling"],
                [FullTraceScaler, HeadScaler, HeadScaler],
            ):
                transformers.append(
                    t(**cfg.build_features[k], output_path=build_features_path)
                )
            for set_name, transformer in zip(dataset_names, transformers):
                dataset = {set_name: datasets[set_name]}
                dataset_labels = {set_name: datasets_labels[set_name]}
                dataset_info = {set_name: datasets_info[set_name]}
                datasets[set_name] = transformer.fit_transform_datasets(
                    dataset, dataset_labels, dataset_info
                )[set_name]
        logging.info("Done.")

    # resample sequences after every transformations if specified and relevant
    if data_downsampling and data_downsampling_position == "last":
        for k in datasets:
            resample_sequences(
                get_dfs_from_numpy(datasets[k], original_sampling_period),
                data_sampling_period,
                anomaly_col=False,
                original_sampling_period=original_sampling_period,
            )
            datasets[k] = get_numpy_from_dfs(datasets[k])

    # save periods with updated features along with features information
    n_input_features = len(input_feature_names)
    n_output_features = (
        n_input_features if output_feature_names is None else len(output_feature_names)
    )
    save_files(
        build_features_path,
        files_dict={
            "features_info": {
                "input_feature_names": input_feature_names,
                "n_input_features": n_input_features,
                "output_feature_names": output_feature_names,
                "n_output_features": n_output_features,
            }
        },
        file_format="json",
    )
    save_files(build_features_path, datasets, "numpy")
