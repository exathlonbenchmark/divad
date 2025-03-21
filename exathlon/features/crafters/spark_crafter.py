import numpy as np
import pandas as pd

from utils.spark.metadata import USED_FEATURES, CUMULATIVE_FEATURES
from utils.guarding import check_value_in_choices
from features.crafters.base_crafter import BaseCrafter


class SparkCrafter(BaseCrafter):
    """Spark-specific feature engineering/crafting class.

    Args:
        **base_kwargs: keyword arguments of `BaseCrafter`.
    """

    # list of feature alteration bundles relevant to spark data
    bundles_list = [
        # bundle #0: set of 19 custom features of the paper
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value",
            ): "identity",
            # features to 1-difference, dropping the original ones
            (
                "driver_StreamingMetrics_streaming_totalCompletedBatches_value",
                "driver_StreamingMetrics_streaming_totalProcessedRecords_value",
                "driver_StreamingMetrics_streaming_totalReceivedRecords_value",
                "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value",
                "driver_BlockManager_memory_memUsed_MB_value",
                "driver_jvm_heap_used_value",
                *[f"node{i}_CPU_ALL_Idle%" for i in range(5, 9)],
            ): "difference_1_drop",
            # features to average across active executors and 1-difference, dropping the original inputs every time
            (
                *[f"{i}_executor_filesystem_hdfs_write_ops_value" for i in range(1, 5)],
                *[f"{i}_executor_cpuTime_count" for i in range(1, 5)],
                *[f"{i}_executor_runTime_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsRead_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsWritten_count" for i in range(1, 5)],
                *[f"{i}_jvm_heap_used_value" for i in range(1, 5)],
            ): "execavg_drop.difference_1_drop",
        },
        # bundle #1: set of 19 custom features removing some differencing
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value",
                "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value",
                "driver_BlockManager_memory_memUsed_MB_value",
                "driver_jvm_heap_used_value",
                *[f"node{i}_CPU_ALL_Idle%" for i in range(5, 9)],
            ): "identity",
            # features to 1-difference, dropping the original ones
            (
                "driver_StreamingMetrics_streaming_totalCompletedBatches_value",
                "driver_StreamingMetrics_streaming_totalProcessedRecords_value",
                "driver_StreamingMetrics_streaming_totalReceivedRecords_value",
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            (*[f"{i}_jvm_heap_used_value" for i in range(1, 5)],): "execavg_drop",
            # features to average across active executors and 1-difference, dropping the original inputs every time
            (
                *[f"{i}_executor_filesystem_hdfs_write_ops_value" for i in range(1, 5)],
                *[f"{i}_executor_cpuTime_count" for i in range(1, 5)],
                *[f"{i}_executor_runTime_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsRead_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsWritten_count" for i in range(1, 5)],
            ): "execavg_drop.difference_1_drop",
        },
        # bundle #2: set of 16 custom features (simpler version of bundle #1)
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value",
                *[f"node{i}_CPU_ALL_Idle%" for i in range(5, 9)],
            ): "identity",
            # features to 1-difference, dropping the original ones
            (
                "driver_StreamingMetrics_streaming_totalCompletedBatches_value",
                "driver_StreamingMetrics_streaming_totalProcessedRecords_value",
                "driver_StreamingMetrics_streaming_totalReceivedRecords_value",
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            (*[f"{i}_jvm_heap_used_value" for i in range(1, 5)],): "execavg_drop",
            # features to average across active executors and 1-difference, dropping the original inputs every time
            (
                *[f"{i}_executor_filesystem_hdfs_write_ops_value" for i in range(1, 5)],
                *[f"{i}_executor_cpuTime_count" for i in range(1, 5)],
                *[f"{i}_executor_runTime_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsRead_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsWritten_count" for i in range(1, 5)],
            ): "execavg_drop.difference_1_drop",
        },
        # bundle #3: same as #2 without total delay
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
                *[f"node{i}_CPU_ALL_Idle%" for i in range(5, 9)],
            ): "identity",
            # features to 1-difference, dropping the original ones
            (
                "driver_StreamingMetrics_streaming_totalCompletedBatches_value",
                "driver_StreamingMetrics_streaming_totalProcessedRecords_value",
                "driver_StreamingMetrics_streaming_totalReceivedRecords_value",
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            (*[f"{i}_jvm_heap_used_value" for i in range(1, 5)],): "execavg_drop",
            # features to average across active executors and 1-difference, dropping the original inputs every time
            (
                *[f"{i}_executor_filesystem_hdfs_write_ops_value" for i in range(1, 5)],
                *[f"{i}_executor_cpuTime_count" for i in range(1, 5)],
                *[f"{i}_executor_runTime_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsRead_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsWritten_count" for i in range(1, 5)],
            ): "execavg_drop.difference_1_drop",
        },
        # bundle #4: only the processing delay (i.e., processing time) of the last completed batch
        {
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
            ): "identity",
        },
        # bundle #5 - all "used" features, differencing "cumulative"
        {
            # features to add as is
            tuple(
                ft
                for ft in USED_FEATURES
                if not any([ft[:2] == f"{e}_" for e in range(1, 5)])
                and ft not in CUMULATIVE_FEATURES
            ): "identity",
            # features to 1-difference, dropping the original ones
            tuple(
                ft
                for ft in CUMULATIVE_FEATURES
                if not any([ft[:2] == f"{e}_" for e in range(1, 5)])
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            tuple(
                ft
                for ft in USED_FEATURES
                if any([ft[:2] == f"{e}_" for e in range(1, 5)])
                and ft not in CUMULATIVE_FEATURES
            ): "execavg_drop",
            # features to 1-difference and average across active executors, dropping originals every time
            tuple(
                ft
                for ft in CUMULATIVE_FEATURES
                if any([ft[:2] == f"{e}_" for e in range(1, 5)])
            ): "execdifference_1_drop.execavg_drop",
        },
        # bundle #6 (16 features): bundle #1 with fixed executor differencing, and:
        # 1. (-) Removed "driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value"
        # => Simply the sum of other delays.
        # 2. (-) Removed diff("driver_StreamingMetrics_streaming_totalReceivedRecords_value")
        # => Same as "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value",
        #   except it is zero when the latter is constant.
        # 3. (-) Removed diff("driver_StreamingMetrics_streaming_totalProcessedRecords_value")
        # => Same as "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value", except
        #   it is delayed (received will be completed a bit later) and zero instead of constant.
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
                "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value",
                "driver_BlockManager_memory_memUsed_MB_value",
                "driver_jvm_heap_used_value",
                *[f"node{i}_CPU_ALL_Idle%" for i in range(5, 9)],
            ): "identity",
            # features to 1-difference, dropping the original ones
            (
                "driver_StreamingMetrics_streaming_totalCompletedBatches_value",
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            (*[f"{i}_jvm_heap_used_value" for i in range(1, 5)],): "execavg_drop",
            # features to 1-difference and average across active executors, dropping originals every time
            (
                *[f"{i}_executor_filesystem_hdfs_write_ops_value" for i in range(1, 5)],
                *[f"{i}_executor_cpuTime_count" for i in range(1, 5)],
                *[f"{i}_executor_runTime_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsRead_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsWritten_count" for i in range(1, 5)],
            ): "execdifference_1_drop.execavg_drop",
        },
        # bundle #7: bundle #5 ("raw" feature set) without OS features
        {
            # features to add as is
            tuple(
                ft
                for ft in USED_FEATURES
                if not any([ft[:6] == f"node{i}_" for i in range(5, 9)])
                and not any([ft[:2] == f"{e}_" for e in range(1, 5)])
                and ft not in CUMULATIVE_FEATURES
            ): "identity",
            # features to 1-difference, dropping the original ones
            tuple(
                ft
                for ft in CUMULATIVE_FEATURES
                if not any([ft[:6] == f"node{i}_" for i in range(5, 9)])
                and not any([ft[:2] == f"{e}_" for e in range(1, 5)])
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            tuple(
                ft
                for ft in USED_FEATURES
                if any([ft[:2] == f"{e}_" for e in range(1, 5)])
                and ft not in CUMULATIVE_FEATURES
            ): "execavg_drop",
            # features to 1-difference and average across active executors, dropping originals every time
            tuple(
                ft
                for ft in CUMULATIVE_FEATURES
                if any([ft[:2] == f"{e}_" for e in range(1, 5)])
            ): "execdifference_1_drop.execavg_drop",
        },
        # bundle #8: bundle #6 ("custom" feature set) without OS features
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
                "driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
                "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value",
                "driver_BlockManager_memory_memUsed_MB_value",
                "driver_jvm_heap_used_value",
            ): "identity",
            # features to 1-difference, dropping the original ones
            (
                "driver_StreamingMetrics_streaming_totalCompletedBatches_value",
            ): "difference_1_drop",
            # features to average across active executors, dropping the original ones
            (*[f"{i}_jvm_heap_used_value" for i in range(1, 5)],): "execavg_drop",
            # features to 1-difference and average across active executors, dropping originals every time
            (
                *[f"{i}_executor_filesystem_hdfs_write_ops_value" for i in range(1, 5)],
                *[f"{i}_executor_cpuTime_count" for i in range(1, 5)],
                *[f"{i}_executor_runTime_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsRead_count" for i in range(1, 5)],
                *[f"{i}_executor_shuffleRecordsWritten_count" for i in range(1, 5)],
            ): "execdifference_1_drop.execavg_drop",
        },
        # bundle #9: only "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value"
        {
            # features to add as is
            (
                "driver_StreamingMetrics_streaming_lastReceivedBatch_records_value",
            ): "identity"
        },
    ]

    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def get_alteration_functions_extension(self):
        return {
            "execdifference": add_exec_differencing,
            "execavg": add_executors_avg,
            "nodeavg": add_nodes_avg,
            "time": add_time_spent,
        }


def add_exec_differencing(period_df, diff_factor_str, original_treatment):
    """Adds executor feature differences, either keeping or dropping the original ones.

    Same as regular differencing, except we preserve the fact that -1 features mean
    the executor is inactive, and should therefore stay -1.

    We also define the "difference" of a non-minus-1 value following a -1 value as its
    original value, which is the difference from the value and its previous "non-existing" state.

    Args:
        period_df (pd.DataFrame): input period DataFrame.
        diff_factor_str (str): differencing factor as a string integer.
        original_treatment (str): either `keep` or `drop`, specifying what to do with original features.

    Returns:
        pd.DataFrame: the input DataFrame with differenced executor features, with or without
            the original ones.
    """
    check_value_in_choices(original_treatment, "original_treatment", ["drop", "keep"])
    check_value_in_choices(diff_factor_str, "diff_factor_str", ["1"])
    # -1 feature spots throughout the sequence
    inactive_mask = period_df == -1.0
    # feature spots following -1 spots (and not -1 themselves)
    following_inactive_mask = inactive_mask.shift(1).fillna(False) & (~inactive_mask)
    # apply first-order differencing, leaving -1's and keeping original values after -1's
    difference_df = period_df.diff(int(diff_factor_str))
    difference_df[inactive_mask] = -1.0
    difference_df[following_inactive_mask] = period_df[following_inactive_mask]
    difference_df = difference_df.dropna()
    difference_df.columns = [
        f"{c}_{diff_factor_str}_diff" for c in difference_df.columns
    ]
    # prepend original input features if we choose to keep them (implicit join if different counts)
    if original_treatment == "keep":
        difference_df = pd.concat([period_df, difference_df], axis=1)
    return difference_df


def add_executors_avg(period_df, original_treatment):
    """Adds executor features averaged across active executors, keeping or not the original ones.

    An executor is defined as "inactive" for a given feature if the value of the feature for
    this executor is -1.

    If all executors are inactive for a particular feature, we set its average to -1.
    """
    assert original_treatment in [
        "drop",
        "keep",
    ], "original features treatment can only be `keep` or `drop`"

    # make sure to only try to average executor features
    exec_ft_names = [c[2:] for c in period_df.columns if c[:2] == "1_"]
    # features groups to average across, each group of the form [`1_ft`, `2_ft`, ..., `5_ft`]
    avg_groups = [
        [c for c in period_df.columns if c[2:] == efn] for efn in exec_ft_names
    ]

    # add features groups averaged across active executors to the result DataFrame
    averaged_df = pd.DataFrame()
    for group in avg_groups:
        # create `avg_ft` from [`1_ft`, `2_ft`, ..., `5_ft`]
        averaged_df = averaged_df.assign(
            **{
                f"avg_{group[0][2:]}": period_df[group]
                .replace(-1, np.nan)
                .mean(axis=1)
                .fillna(-1)
            }
        )
    # prepend original input features if we choose to keep them
    if original_treatment == "keep":
        averaged_df = pd.concat([period_df, averaged_df], axis=1)
    return averaged_df


def add_nodes_avg(period_df, original_treatment):
    """Adds node features averaged across nodes, keeping or not the original ones."""
    assert original_treatment in [
        "drop",
        "keep",
    ], "original features treatment can only be `keep` or `drop`"

    # make sure to only try to average node features
    node_ft_names = [c[6:] for c in period_df.columns if c[:4] == "node"]
    # features groups to average across, each group of the form [`node5_ft`, `node6_ft`, ..., `node8_ft`]
    avg_groups = [
        [c for c in period_df.columns if c[6:] == nfn] for nfn in node_ft_names
    ]

    # add features groups averaged across nodes to the result DataFrame
    averaged_df = pd.DataFrame()
    for group in avg_groups:
        # create `avg_node_ft` from [`node5_ft`, `node6_ft`, ..., `node8_ft`]
        averaged_df = averaged_df.assign(
            **{f"avg_node_{group[0][6:]}": period_df[group].mean(axis=1)}
        )
    # prepend original input features if we choose to keep them
    if original_treatment == "keep":
        averaged_df = pd.concat([period_df, averaged_df], axis=1)
    return averaged_df


def add_time_spent(period_df, original_treatment):
    """Adds the time spent on a given task based on features whose names contain "StartTime" and "EndTime"."""
    assert original_treatment in [
        "drop",
        "keep",
    ], "original features treatment can only be `keep` or `drop`"
    start_str, end_str = "StartTime", "EndTime"
    a_t = (
        "time spent can only be computed on a DataFrame with only two columns: "
        'one whose name contains "StartTime", and the other whose name contains "EndTime"'
    )
    col_idx_dict = dict()
    for k, str_ in zip(["start", "end"], [start_str, end_str]):
        relevant_col_ids = [i for i, c in enumerate(period_df.columns) if str_ in c]
        assert len(relevant_col_ids) == 1, a_t
        col_idx_dict[k] = relevant_col_ids[0]
    # add feature for the time spent
    start_ft, end_ft = (
        period_df.columns[col_idx_dict["start"]],
        period_df.columns[col_idx_dict["end"]],
    )
    time_ft = start_ft.replace(start_str, "")
    result_df = pd.DataFrame()
    # replace negative values by zero, in case there was some noise in the start and end times
    result_df[time_ft] = (period_df[end_ft] - period_df[start_ft]).apply(
        lambda x: 0 if x < 0 else x
    )
    # prepend original input features if we choose to keep them
    if original_treatment == "keep":
        result_df = pd.concat([period_df, result_df], axis=1)
    return result_df
