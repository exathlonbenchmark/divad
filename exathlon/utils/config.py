"""Helper functions to manipulate configuration objects."""
import os
import copy
import yaml
import operator
from dotenv import load_dotenv, find_dotenv


from typing import Generator, Tuple, List, Any, Callable, Optional, Mapping
from omegaconf import DictConfig, OmegaConf, open_dict


def add_data_path_to_config(cfg: DictConfig) -> None:
    """Adds `cfg.dataset.metadata.path` based on `cfg.dataset.metadata.name`, inplace."""
    load_dotenv(find_dotenv())
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.dataset.metadata.path = os.getenv(cfg.dataset.metadata.name.upper())


def filter_conditional_config(cfg: DictConfig) -> DictConfig:
    """Removes conditional arguments that have no effect on the output."""
    # loop on reversed list to process the deepest paths first
    for key, value, path in reversed(list(config_full_paths(cfg))):
        if key == "value":
            param = copy.deepcopy(get_from_path(cfg, path))
            set_from_path(cfg, path, param["value"])
            for condition in {k for k in param if k != "value"}:
                if get_relevance_filter(condition)(param["value"]):
                    for k, v in param[condition].items():
                        set_from_path(cfg, path[:-1] + [k], v)
    return cfg


def get_same_config_path(cfg: dict, step: str) -> str:
    step_path = os.path.join(os.getcwd(), cfg["dataset"]["name"], step)
    try:
        for dir_ in os.listdir(step_path):
            dir_path = os.path.join(step_path, dir_)
            if os.path.isdir(dir_path):
                try:
                    ran_cfg = OmegaConf.load(os.path.join(dir_path, "config.yaml"))
                    if cfg == ran_cfg:
                        return os.path.abspath(os.path.join(step_path, dir_))
                except FileNotFoundError:  # if no "config.yaml" not in `dir_path`
                    pass
    except FileNotFoundError:  # case `step_path` does not exist
        pass
    return ""


def get_same_run_id_path(cfg: dict, step: str, run_id: str) -> str:
    step_path = os.path.join(os.getcwd(), cfg["dataset"]["name"], step)
    try:
        for dir_ in os.listdir(step_path):
            if dir_ == run_id:
                return os.path.abspath(os.path.join(step_path, dir_))
    except FileNotFoundError:
        pass
    return ""


def get_config_path(dataset: str, step: str, run_id: str) -> str:
    return os.path.join(os.getcwd(), dataset, step, run_id)


def save_config(cfg: DictConfig, cfg_path: str):
    os.makedirs(cfg_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg_path, "config.yaml"))


def get_relevance_filter(filter_: str) -> Callable[[Any], bool]:
    """`filter_` should be of the form: "OP__value", where OP is a simple operator."""
    str_to_op = {
        "lt__": operator.lt,
        "le__": operator.le,
        "eq__": operator.eq,
        "ne__": operator.ne,
        "ge__": operator.ge,
        "gt__": operator.gt,
        "contains__": operator.contains,
    }
    try:
        op_str, op = [(s, op) for s, op in str_to_op.items() if s in filter_][0]
    except IndexError:
        raise ValueError(f'Unknown config filter "{filter_}".')
    value = filter_[len(op_str) :]
    # convert string to key to YAML-loaded type (will be same as `x`)
    value = yaml.load(value, yaml.FullLoader)
    return lambda x: op(x, value)


def config_full_paths(
    cfg: DictConfig, path: Optional[list] = None
) -> Generator[Tuple[Any, Any, List[Any]], None, None]:
    if path is None:
        path = []
    for k, v in cfg.items():
        updated_path = path + [k]
        if isinstance(v, DictConfig):
            for yielded in config_full_paths(v, updated_path):
                yield yielded
        else:
            yield k, v, path


def set_from_path(cfg: Mapping, path: list, value: Any):
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            for k in path[:-1]:
                cfg = cfg[k]
            cfg.pop(path[-1], None)  # remove any existing value
            cfg[path[-1]] = value
    else:
        for k in path[:-1]:
            cfg = cfg[k]
        cfg.pop(path[-1], None)
        cfg[path[-1]] = value


def get_from_path(cfg: Mapping, path: list) -> Any:
    for k in path[:-1]:
        cfg = cfg[k]
    return cfg[path[-1]]


def remove_from_path(cfg: Mapping, path: list):
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            for k in path[:-1]:
                cfg = cfg[k]
            cfg.pop(path[-1], None)
    else:
        for k in path[:-1]:
            cfg = cfg[k]
        cfg.pop(path[-1], None)


def hyper_to_str(*hps: list) -> str:
    """Returns the formatted string corresponding to the provided hyperparameters."""
    # empty string and empty list parameters are ignored, and `[a, b]` lists are turned to `a-b`
    str_hps = []
    for hp in hps:
        parsed_str_hp = str(hp).replace(", ", "-").replace("[", "").replace("]", "")
        if parsed_str_hp != "":
            str_hps.append(parsed_str_hp)
    return "_".join(str_hps)
