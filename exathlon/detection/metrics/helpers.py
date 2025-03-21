"""Anomaly detection metrics helpers.
"""
import numpy as np
from sklearn.metrics import auc


def get_peak_fprt(f_scores, precisions, recalls, thresholds):
    """Returns the "peak" F-score from `f_scores` along with its corresponding precision, recall and threshold.

    Args:
        f_scores (ndarray): 1d array of F-scores (one for each threshold value).
        precisions (ndarray): 1d array of corresponding precision scores.
        recalls (ndarray): 1d array of corresponding recall scores.
        thresholds (ndarray): 1d array of corresponding threshold values.

    Returns:
        float, float, float, float: "peak" F-score with its corresponding precision, recall and threshold.
    """
    peak_idx = np.argmax(f_scores)
    return (
        f_scores[peak_idx],
        precisions[peak_idx],
        recalls[peak_idx],
        thresholds[peak_idx],
    )


def get_f_beta_score(precision, recall, beta):
    """Returns the F_{`beta`}-score for the provided `precision` and `recall`.

    The F-score is defined as zero if both Precision and Recall are zero.
    """
    beta_squared = beta**2
    try:
        f_score = np.nan_to_num(
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall)
        )
        return f_score
    except (
        ZeroDivisionError
    ):  # only thrown if `precision` and `recall` are not ndarrays
        return 0.0


def get_auc(x, y):
    """Returns the Area Under the Curve (AUC) for the provided `(x, y)` points.

    Prior to computing the AUC, the (x, y) points are sorted by `x` values to guarantee their monotonicity.

    Args:
        x (array-like): 1d-array of x-values.
        y (array-like): 1d-array of y-values.

    Returns:
        float: the AUC for the `(x, y)` points.
    """
    sorting_ids = np.argsort(x)
    return auc(x[sorting_ids], y[sorting_ids])


def extract_binary_ranges_ids(y):
    """Returns the start and (excluded) end indices of all contiguous ranges of 1s in the binary array `y`.

    Args:
        y (ndarray): 1d-array of binary elements.

    Returns:
        ndarray: array of `start, end` indices: `[[start_1, end_1], [start_2, end_2], ...]`.
    """
    y_diff = np.diff(y)
    start_ids = np.concatenate(
        [[0] if y[0] == 1 else np.array([], dtype=int), np.where(y_diff == 1)[0] + 1]
    )
    end_ids = np.concatenate(
        [
            np.where(y_diff == -1)[0] + 1,
            [len(y)] if y[-1] == 1 else np.array([], dtype=int),
        ]
    )
    return np.array(list(zip(start_ids, end_ids)))


def extract_multiclass_ranges_ids(y):
    """Returns the start and (excluded) end ids of all contiguous ranges for each non-zero label in `y`.

    The lists of ranges are returned as a dictionary with as keys the positive labels in `y`
    and as values the corresponding range tuples.

    Args:
        y (ndarray): 1d-array of multiclass labels.

    Returns:
        dict: for each non-zero label, array of `start, end` indices:
            => `[[start_1, end_1], [start_2, end_2], ...]`.
    """
    # distinct non-zero labels in `y` (i.e. "positive classes")
    y_classes = np.unique(y)
    pos_classes = y_classes[y_classes != 0]
    ranges_ids_dict = dict()
    for pc in pos_classes:
        # extract binary ranges setting the class label to 1 and all others to 0
        ranges_ids_dict[pc] = extract_binary_ranges_ids((y == pc).astype(int))
    return ranges_ids_dict


def get_overlapping_ids(target_range, ranges, only_first=False):
    """Returns either the minimum or the full list of overlapping indices between `target_range` and `ranges`.

    If there are no overlapping indices between `target_range` and `ranges`, None is returned if
    `only_first` is True, else an empty list is returned.

    Note: the end indices can be either included or not, what matters here is the overlap
    between the ids, and not what they represent.

    Args:
        target_range (ndarray): target range to match as a `[start, end]` array.
        ranges (ndarray): candidate ranges, as a 2d `[[start_1, end_1], [start_2, end_2], ...]` array.
        only_first (bool): whether to return only the first (i.e. minimum) overlapping index.

    Returns:
        int|list: the first overlapping index if `only_first` is True, else the full list of overlapping indices.
    """
    target_set, overlapping_ids = (
        set(range(target_range[0], target_range[1] + 1)),
        set(),
    )
    for range_ in ranges:
        overlapping_ids = overlapping_ids | (
            set(range(range_[0], range_[1] + 1)) & target_set
        )
    if only_first and len(overlapping_ids) == 0:
        return None
    else:
        return min(overlapping_ids) if only_first else list(overlapping_ids)


def get_overlapping_range(range_1, range_2):
    """Returns the start and end indices of the overlap between `range_1` and `range_2`.

    If `range_1` and `range_2` do not overlap, None is returned.

    Args:
        range_1 (ndarray): `[start, end]` indices of the first range.
        range_2 (ndarray): `[start, end]` indices of the second range.

    Returns:
        ndarray|None: `[start, end]` indices of the overlap between the 2 ranges, None if none.
    """
    overlapping_ids = get_overlapping_ids(
        range_1, np.array([range_2]), only_first=False
    )
    if len(overlapping_ids) == 0:
        return None
    return np.array([min(overlapping_ids), max(overlapping_ids)])


def get_overlapping_ranges(target_range, ranges):
    """Returns the start and end indices of all overlapping ranges between `target_range` and `ranges`.

    If none of the ranges overlap with `target_range`, an empty ndarray is returned.

    Args:
        target_range (ndarray): target range to overlap as a `[start, end]` array.
        ranges (ndarray): candidate ranges, as a 2d `[[start_1, end_1], [start_2, end_2], ...]` array.

    Returns:
        ndarray: all overlaps as a `[[start_1, end_1], [start_2, end_2], ...]` array.
    """
    overlaps = []
    for range_ in ranges:
        overlap = get_overlapping_range(target_range, range_)
        if overlap is not None:
            overlaps.append(overlap)
    return np.array(overlaps)
