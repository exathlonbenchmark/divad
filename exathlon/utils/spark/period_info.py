"""Utilities related to "period information" for spark data."""


def get_parsed_period_info(period_info: list) -> dict:
    """Returns the provided `period_info` parsed as a dictionary.

    Args:
        period_info: list of "information" about the period. For spark data, this corresponds
          to `[file_name, trace_type, period_rank]`, where `period_rank` is the chronological rank
          of the period in its file.

    Returns:
        The parsed `{key: value}` period information, where `key` is the information description
          and `value` the content.
    """
    file_name = period_info[0]
    keys = ["app_id", "trace_type", "input_rate"]
    return dict(
        {"file_name": file_name},
        **{k: get_item_from_file_name(file_name, i) for k, i in zip(keys, range(3))},
    )


# getter function returning the (possibly casted) item at `idx` in a "_"-separated trace file name
def get_item_from_file_name(file_name, idx, casting_f=int):
    return casting_f(file_name.split("_")[idx])


def get_period_title_from_info(period_info):
    """Returns the title used to describe a specific period's plot from its information list.

    Args:
        period_info (list): period information of the form `[file_name, trace_type, period_rank]`.

    Returns:
        str: the period's title based on its information list.
    """
    return f"{period_info[0]} ({get_trace_type_title(period_info[1])})"


def get_trace_type_title(trace_type_str):
    """Returns the provided trace type string as a formatted title to show in figures."""
    return trace_type_str.title().replace("_", " ").replace("Cpu", "CPU")
