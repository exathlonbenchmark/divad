"""Logging-related utilities."""
from typing import Callable


def get_verbose_print(verbose: bool) -> Callable:
    return print if verbose else lambda *args, **kwargs: None
