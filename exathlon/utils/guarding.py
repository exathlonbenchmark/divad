"""Provides a set of guarding functions for checking variable types and contents.

Note: exceptions thrown by guarding functions should not be caught by callers.
"""
import os
from typing import Union, Any, Sequence, Mapping


class InvalidArgumentException(Exception):
    """Raised when an invalid argument is passed in view of the data."""

    pass


def check_is_not_none(value: Any, var_name: str):
    if value is None:
        raise ValueError(f"Received invalid `None` value for `{var_name}`.")


def check_is_percentage(value: Union[int, float], var_name: str):
    if value < 0 or value > 1:
        raise ValueError(f"`{var_name}` should be a percentage, received {value}.")


def check_is_non_zero(value: Union[int, float], var_name: str):
    if value == 0.0:
        raise ValueError(f"`{var_name}` should be non-zero.")


def check_all_values_in_choices(values: Sequence, var_name: str, choices: list):
    if not all([v in choices for v in values]):
        raise ValueError(
            f"All values in `{var_name}` should be in {choices}, received {values}."
        )


def check_value_in_choices(value: Any, var_name: str, choices: list):
    if value not in choices:
        raise ValueError(f"`{var_name}` should be in {choices}, received {value}.")


def check_contains_all_keys(mapping: Mapping, var_name: str, keys: list):
    absent_keys = set(keys) - set(mapping.keys())
    if len(absent_keys) > 0:
        raise KeyError(
            f"All of {keys} must be in {var_name} (absent keys: {absent_keys})."
        )


def check_all_files_exist(root_path, file_names):
    """Check that all (full) `file_names` exist at `root_path`."""
    for fn in file_names:
        if not os.path.isfile(os.path.join(root_path, fn)):
            return False
    return True
