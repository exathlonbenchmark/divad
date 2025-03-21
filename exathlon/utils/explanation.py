import importlib
from typing import Optional


def get_explainer_type(explainer_name: str) -> str:
    """Returns the type of explainer based on `explainer_name`.

    - "detector": the detector model is assumed capable of explaining anomalies.
    - "data": a standalone explainer is used to explain anomalous *data*.
    - "model": a standalone explainer is used to explain the predictions of a *model*.

    Args:
        explainer_name: explainer name, as specified in the pipeline configuration.

    Returns:
        Type of explainer: either "detector", "data" or "model".
    """
    if explainer_name == "detector":
        return "detector"
    explainer_package = None
    for package in ["data_explainers", "model_explainers"]:
        try:
            importlib.import_module(
                f"explanation.explainers.{package}.{explainer_name}"
            )
            explainer_package = package
        except ImportError:
            pass
    if explainer_package is None:
        raise ImportError(f"Could not find explainer {explainer_name}.")
    return explainer_package.split("_")[0]


def is_detector_relevant(
    step: str, explainer_name: str, explained_anomalies: str
) -> bool:
    """Returns `True` if a detector model is relevant for the provided parameters, `False` otherwise.

    The two cases where a detector is not relevant are:

    - When fitting a standalone data explainer.
    - When evaluating a standalone data explainer on ground-truth labels.
    """
    explainer_type = get_explainer_type(explainer_name)
    data_exp_evaluation = (
        step == "evaluate_explainer"
        and explainer_type == "data"
        and explained_anomalies == "ground_truth"
    )
    data_exp_fitting = step == "train_explainer" and explainer_type == "data"
    return not (data_exp_evaluation or data_exp_fitting)


def get_last_detector_step(
    step: str, explainer_name: str, explained_anomalies: Optional[str] = None
) -> str:
    explainer_type = get_explainer_type(explainer_name)
    if (
        step not in ["train_explainer", "evaluate_explainer"]
        or explainer_type == "detector"
    ):
        # pipeline specified by the detector
        last_detector_step = step
    elif step == "evaluate_explainer" and explained_anomalies == "model_preds":
        # evaluating binary model predictions - TODO: could be "train_online_detector", refactor.
        last_detector_step = "evaluate_online_detector"
    elif explainer_type == "model":
        # explaining window scoring function - TODO: could be "train_window_scorer", refactor.
        last_detector_step = "evaluate_online_scorer"
    else:
        raise ValueError(
            "All cases should have been covered if a detector was relevant."
        )
    return last_detector_step
