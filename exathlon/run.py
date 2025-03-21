"""Runs a pipeline step."""
# TODO: rename train_* scripts to train_predict_* scripts.
import os
import copy
import time
import shutil
import hydra
import logging
import importlib
from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig, OmegaConf
from typing import List
from multiprocessing import cpu_count

from inputimeout import inputimeout, TimeoutOccurred


from utils.guarding import check_value_in_choices
from utils.config import (
    add_data_path_to_config,
    filter_conditional_config,
    get_same_config_path,
    get_same_run_id_path,
    get_config_path,
    save_config,
)
from utils.explanation import (
    get_explainer_type,
    is_detector_relevant,
    get_last_detector_step,
)
from data import make_datasets
from features import build_features
from detection import (
    make_window_datasets,
    train_window_model,
    train_window_scorer,
    train_online_scorer,
    train_online_detector,
    evaluate_online_scorer,
    evaluate_online_detector,
)
from explanation import train_explainer, evaluate_explainer


def get_outputs_root():
    load_dotenv(find_dotenv())
    return os.getenv("OUTPUTS")


OmegaConf.register_new_resolver("outputs_root", get_outputs_root)


STEPS = [
    "make_datasets",
    "build_features",
    "make_window_datasets",
    "train_window_model",
    "train_window_scorer",
    "train_online_scorer",
    "evaluate_online_scorer",
    "train_online_detector",
    "evaluate_online_detector",
    "train_explainer",
    "evaluate_explainer",
]
STEP_TO_MODULE = {
    "make_datasets": make_datasets,
    "build_features": build_features,
    "make_window_datasets": make_window_datasets,
    "train_window_model": train_window_model,
    "train_window_scorer": train_window_scorer,
    "train_online_scorer": train_online_scorer,
    "evaluate_online_scorer": evaluate_online_scorer,
    "train_online_detector": train_online_detector,
    "evaluate_online_detector": evaluate_online_detector,
    "train_explainer": train_explainer,
    "evaluate_explainer": evaluate_explainer,
}
DEFAULT_INPUTS_TIMEOUT = 10.0  # in seconds


def get_leaderboard_steps(evaluation_step: str) -> list:
    """Returns the steps to use as leaderboard columns when running `evaluation_step`."""
    evaluation_idx = STEPS.index(evaluation_step)
    return [s for s in STEPS[:evaluation_idx] if "evaluate" not in s] + [
        evaluation_step
    ]


def input_timeout(
    prompt: str = "(y/n): ", timeout: float = DEFAULT_INPUTS_TIMEOUT, default: str = "y"
):
    try:
        return inputimeout(
            f'{prompt[:-2]} ["{default}" AFTER {timeout}s]: ', timeout=timeout
        )
    except TimeoutOccurred:
        return default


def get_step_sequence(cfg: DictConfig) -> List[str]:
    """Returns the sequence of steps ran/to run up to the provided step."""
    if cfg.step in ["make_datasets", "build_features", "make_window_datasets"]:
        # the step sequence is always the same
        step_idx = STEPS.index(cfg.step)
        step_sequence = STEPS[: step_idx + 1]
    else:
        # the step sequence depends on the detection and/or explanation method
        step_sequence = ["make_datasets", "build_features"]
        detector_name = cfg.detector.name
        explainer_name = cfg.explainer.name
        if is_detector_relevant(
            cfg.step,
            explainer_name,
            cfg.explainer.evaluate_explainer.explained_anomalies,
        ):
            # relevant steps from the detector
            detector_module = importlib.import_module(
                f"detection.detectors.{detector_name}"
            )
            detector_class = getattr(
                detector_module, detector_name.title().replace("_", "")
            )
            last_detector_step = get_last_detector_step(
                step=cfg.step,
                explainer_name=explainer_name,
                explained_anomalies=cfg.explainer.evaluate_explainer.explained_anomalies,
            )
            prev_detector_steps = detector_class.get_previous_relevant_steps(
                last_detector_step
            )
            step_sequence += prev_detector_steps
        if cfg.step == "evaluate_explainer":
            # "train_explainer" sets parameters and saves the object: always relevant to explainers
            step_sequence.append("train_explainer")
        step_sequence.append(cfg.step)
    return step_sequence


def get_formatted_step_config(cfg: DictConfig, step: str) -> dict:
    """Returns the formatted configuration for the step."""
    explainer_type = get_explainer_type(cfg.explainer.name)
    if step in ["make_datasets", "build_features"]:
        step_cfg = cfg.dataset[step]
    elif (
        step not in ["train_explainer", "evaluate_explainer"]
        or explainer_type == "detector"
    ):
        step_cfg = cfg.detector[step]
    else:
        step_cfg = cfg.explainer[step]
    return OmegaConf.to_container(step_cfg)


def get_step_to_id(step_subsequence: list, run_id: str) -> dict:
    return {s: id_ for s, id_ in zip(step_subsequence, run_id.split("__"))}


def get_should_override(prompt: str, existing_output_path: str) -> bool:
    """Returns the decision of overriding the step with `existing_output_path`.

    If overriding is chosen, `existing_output_path` will be deleted.

    Args:
        prompt: message to display when asking to override.
        existing_output_path: existing output path to delete if overriding.

    Returns:
        `True` if the step should be overriden, `False` otherwise.
    """
    override = None
    should_override = False
    while override not in ["y", "Y", "n", "N"]:
        override = input_timeout(prompt)
    if override in ["y", "Y"]:
        logging.info(f"deleted directory {existing_output_path}.")
        shutil.rmtree(existing_output_path)
        should_override = True
    return should_override


@hydra.main(
    version_base=None,
    config_path=os.path.join(__file__, os.pardir, os.pardir, "conf"),
    config_name="config",
)
def main(cfg: DictConfig):
    """Runs an exathlon pipeline step.

    Args:
        cfg: hydra configuration object.
    """
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    logging.info(f"Available CPUs: {cpu_count()}")
    check_value_in_choices(cfg.step, "step", STEPS)
    add_data_path_to_config(cfg)
    cfg = filter_conditional_config(cfg)
    run_cfg = {"dataset": cfg.dataset.metadata}
    step_sequence = get_step_sequence(cfg)
    target_step_idx = len(step_sequence) - 1
    step_to_out_path = dict()
    run_id = ""
    for i, step in enumerate(step_sequence):
        should_run_step = False
        run_cfg[step] = get_formatted_step_config(cfg, step)
        # add detector name once if a detector is involved
        if (
            step not in ["make_datasets", "build_features", "make_window_datasets"]
            and "detector" not in run_cfg
        ):
            run_cfg["detector"] = cfg.detector.name
            run_cfg["explainer"] = cfg.explainer.name
        step_id = run_cfg[step].pop("step_id")
        if step_id == "":
            step_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_id = step_id if run_id == "" else "__".join([run_id, step_id])
        same_cfg_path = get_same_config_path(run_cfg, step)
        same_run_id_path = get_same_run_id_path(run_cfg, step, run_id)
        if same_cfg_path == "" and same_run_id_path != "":
            prompt_start = "Required step" if i < target_step_idx else "Step"
            config_exists = os.path.exists(
                os.path.join(same_run_id_path, "config.yaml")
            )
            if config_exists:
                current_step_to_id = {step: step_id}
                raise ValueError(
                    f"{prompt_start} {current_step_to_id} given {get_step_to_id(step_sequence[:i], run_id)} "
                    "already exists for a different configuration, please rename."
                )
            else:
                should_run_step = get_should_override(
                    f'{prompt_start} "{step}" was run with errors at "{same_run_id_path}". Override? (y/n): ',
                    same_run_id_path,
                )
                if not should_run_step:
                    return None
        if i < target_step_idx:  # steps required before target
            if same_cfg_path == "":  # config does not exist
                if same_run_id_path == "":  # id does not exist
                    run_missing = None
                    while run_missing not in ["y", "Y", "n", "N"]:
                        run_missing = input_timeout(
                            f'Required step "{step}" was not run. Run? (y/n): '
                        )
                    if run_missing in ["y", "Y"]:
                        should_run_step = True
                    else:
                        return None
            elif same_run_id_path == "":  # config exists but for a different id
                existing_run_id = os.path.basename(same_cfg_path)
                rename_to_existing = None
                while rename_to_existing not in ["y", "Y", "n", "N"]:
                    existing_step_id = existing_run_id.split("__")[-1]
                    rename_to_existing = input_timeout(
                        f'Required step "{step}" given {get_step_to_id(step_sequence[:i], run_id)} '
                        f'was run with different id "{existing_step_id}". '
                        f'Rename "{step_id}" to "{existing_step_id}"? (y/n): '
                    )
                if rename_to_existing in ["y", "Y"]:
                    run_id = existing_run_id
                else:
                    return None
        else:  # target step
            if same_cfg_path != "":  # config already exists...
                if same_run_id_path != "":  # ...with the same id: propose to override
                    should_run_step = get_should_override(
                        f'Step "{step}" already run with same config at "{same_cfg_path}". Override? (y/n): ',
                        same_cfg_path,
                    )
                    if not should_run_step:
                        return None
                else:  # config exists for a different id: propose to change name and override
                    existing_run_id = os.path.basename(same_cfg_path)
                    rename_to_existing = None
                    while rename_to_existing not in ["y", "Y", "n", "N"]:
                        existing_step_id = existing_run_id.split("__")[-1]
                        rename_to_existing = input_timeout(
                            f'Step "{step}" given {get_step_to_id(step_sequence[:i], run_id)} was run '
                            f'with different id "{existing_step_id}". '
                            f'Rename "{step_id}" to "{existing_step_id}" and override? (y/n): '
                        )
                    if rename_to_existing in ["y", "Y"]:
                        run_id = existing_run_id
                        logging.info(f"deleted directory {same_cfg_path}.")
                        shutil.rmtree(same_cfg_path)
                        should_run_step = True
                    else:
                        return None
            else:  # config does not exist
                should_run_step = True
        step_to_out_path[step] = get_config_path(
            run_cfg["dataset"]["name"], step, run_id
        )
        if should_run_step:
            run_cfg = OmegaConf.create(run_cfg)
            # copy config to save and pass along to next steps in case `main()` alters it
            run_cfg_copy = copy.deepcopy(run_cfg)
            STEP_TO_MODULE[step].main(run_cfg, step_to_out_path)
            run_cfg = run_cfg_copy
            # save configuration ran for the pipeline step
            save_config(run_cfg, step_to_out_path[step])


if __name__ == "__main__":
    raise SystemExit(main())
