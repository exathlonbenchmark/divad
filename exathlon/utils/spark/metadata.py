"""Spark dataset metadata."""

import os
import json


APP_IDS = list(range(1, 11))
TRACE_TYPES = [
    "undisturbed",
    "bursty_input",
    "bursty_input_crash",
    "stalled_input",
    "cpu_contention",
    "process_failure",
]
ANOMALY_TYPES = [
    "bursty_input",
    "bursty_input_crash",
    "stalled_input",
    "cpu_contention",
    "driver_failure",
    "executor_failure",
    "unknown",
]
FEATURES = json.load(
    open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "features.json"),
        "r",
    )
)
DROPPED_FEATURES = FEATURES["dropped"]
USED_FEATURES = FEATURES["used"]
CUMULATIVE_FEATURES = FEATURES["cumulative"]
TRACE_CONSTANT_FEATURES = FEATURES["trace_constant"]

TRACE_NAMES = [
    "1_0_1000000_14",
    "1_0_100000_15",
    "1_0_100000_16",
    "1_0_10000_17",
    "1_0_500000_18",
    "1_0_500000_19",
    "1_2_100000_68",
    "1_4_1000000_80",
    "1_5_1000000_86",
    "2_0_100000_20",
    "2_0_100000_22",
    "2_0_1200000_21",
    "2_1_100000_60",
    "2_2_200000_69",
    "2_5_1000000_87",
    "2_5_1000000_88",
    "3_0_100000_24",
    "3_0_100000_25",
    "3_0_100000_26",
    "3_0_1200000_23",
    "3_2_500000_70",
    "3_2_1000000_71",
    "3_4_1000000_81",
    "3_5_1000000_89",
    "4_0_1000000_31",
    "4_0_100000_27",
    "4_0_100000_28",
    "4_0_100000_29",
    "4_0_100000_30",
    "4_0_100000_32",
    "4_1_100000_61",
    "4_5_1000000_90",
    "5_0_100000_33",
    "5_0_100000_34",
    "5_0_100000_35",
    "5_0_100000_36",
    "5_0_100000_37",
    "5_0_100000_40",
    "5_0_50000_38",
    "5_0_50000_39",
    "5_1_100000_63",
    "5_1_100000_64",
    "5_1_500000_62",
    "5_2_1000000_72",
    "5_4_1000000_82",
    "5_5_1000000_91",
    "5_5_1000000_92",
    "6_0_100000_42",
    "6_0_100000_43",
    "6_0_100000_44",
    "6_0_100000_45",
    "6_0_100000_46",
    "6_0_100000_52",
    "6_0_50000_48",
    "6_0_50000_49",
    "6_0_1200000_41",
    "6_0_300000_50",
    "6_0_50000_47",
    "6_0_50000_51",
    "6_1_500000_65",
    "6_3_200000_76",
    "6_5_1000000_93",
    "7_0_100000_53",
    "7_0_100000_54",
    "7_0_100000_55",
    "7_0_100000_57",
    "7_0_100000_59",
    "7_0_50000_56",
    "7_0_50000_58",
    "8_3_200000_73",
    "8_4_1000000_77",
    "8_5_1000000_83",
    "9_0_100000_3",
    "9_0_300000_5",
    "9_0_100000_1",
    "9_0_100000_4",
    "9_0_100000_6",
    "9_0_1200000_2",
    "9_2_1000000_66",
    "9_3_500000_74",
    "9_4_1000000_78",
    "9_5_1000000_84",
    "10_0_300000_12",
    "10_0_100000_10",
    "10_0_100000_11",
    "10_0_100000_13",
    "10_0_100000_8",
    "10_0_100000_9",
    "10_0_1200000_7",
    "10_2_1000000_67",
    "10_3_1000000_75",
    "10_4_1000000_79",
    "10_5_1000000_85",
]

APP_TO_BATCH_INTERVAL = {3: 20}
for k in [1, 2, 4, 5, 6, 7, 8, 9, 10]:
    APP_TO_BATCH_INTERVAL[k] = 5 if k <= 7 else 10
TRACE_TO_BATCH_INTERVAL = {
    s: APP_TO_BATCH_INTERVAL[int(s.split("_")[0])] for s in TRACE_NAMES
}
TRACE_TYPE_TO_N_RUNNING_EXECUTORS = {0: 2, 1: 2, 2: 3, 3: 2, 4: 2, 5: 3}
TRACE_TO_N_RUNNING_EXECUTORS = {
    s: TRACE_TYPE_TO_N_RUNNING_EXECUTORS[int(s.split("_")[1])] for s in TRACE_NAMES
}
# T1 exception (ignoring apps 7 and 8)
TRACE_TO_N_RUNNING_EXECUTORS["4_1_100000_61"] = 3
# T4 exceptions (ignoring apps 7 and 8)
for k in ["3_4_1000000_81", "9_4_1000000_78"]:
    TRACE_TO_N_RUNNING_EXECUTORS[k] = 3
# T5 exceptions (ignoring apps 7 and 8)
for k in ["5_5_1500000_92", "10_5_1000000_85"]:
    TRACE_TO_N_RUNNING_EXECUTORS[k] = 2

# in GB (also affects max memory set for the driver's block manager and executors' garbage collectors)
TRACE_TYPE_TO_MAX_EXEC_MEMORY = {
    0: 30,
    1: 9,
    2: 15,
    3: 15,
    4: 15,
    5: 15,
}
TRACE_TO_MAX_EXEC_MEMORY = {
    s: TRACE_TYPE_TO_MAX_EXEC_MEMORY[int(s.split("_")[1])] for s in TRACE_NAMES
}
# T0 exception (ignoring apps 7 and 8)
TRACE_TO_MAX_EXEC_MEMORY["9_0_100000_3"] = 9
# T1 exceptions (ignoring apps 7 and 8)
for k in ["5_1_500000_62", "6_1_500000_65"]:
    TRACE_TO_MAX_EXEC_MEMORY[k] = 15
# T3 exception (ignoring apps 7 and 8)
TRACE_TO_MAX_EXEC_MEMORY["6_3_200000_76"] = 30
# T5 exceptions (ignoring apps 7 and 8)
for k in ["2_5_1000000_87", "5_5_1000000_91"]:
    TRACE_TO_MAX_EXEC_MEMORY[k] = 9

# renamed traces accounting for corrected input rates (all name-related variables use the original names)
TRACE_TO_RENAMED = {
    "1_5_1000000_86": "1_5_2500000_86",
    "1_4_1000000_80": "1_4_2500000_80",  # really ~2.2M, but same group as 2.5M
    "3_5_1000000_89": "3_5_500000_89",
    "3_2_1000000_71": "3_2_250000_71",
    "3_2_500000_70": "3_2_125000_70",
    "3_0_1200000_23": "3_0_300000_23",
    "3_0_100000_26": "3_0_25000_26",
    "3_0_100000_25": "3_0_25000_25",
    "3_0_100000_24": "3_0_25000_24",
    "4_5_1000000_90": "4_5_2500000_90",  # really ~2.3M, but same group as 2.5M
    "5_5_1000000_92": "5_5_1500000_92",  # really ~1.6M
    "5_5_1000000_91": "5_5_1500000_91",  # really ~1.8M
    "5_4_1000000_82": "5_4_1500000_82",  # really ~1.6M
    "6_5_1000000_93": "6_5_1500000_93",  # really ~1.8M
    "6_0_50000_49": "6_0_10000_49",
    "6_0_50000_48": "6_0_10000_48",
    "8_3_200000_73": "8_3_100000_73",
    "9_2_1000000_66": "9_2_500000_66",
    "9_0_300000_5": "9_0_150000_5",
    "9_0_1200000_2": "9_0_600000_2",
    "9_0_100000_3": "9_0_50000_3",
    "9_0_100000_6": "9_0_50000_6",
    "9_0_100000_4": "9_0_50000_4",
    "9_0_100000_1": "9_0_50000_1",
    "10_3_1000000_75": "10_3_500000_75",  # really ~480,000
    "10_2_1000000_67": "10_2_500000_67",
    "10_0_300000_12": "10_0_150000_12",
    "10_0_1200000_7": "10_0_600000_7",
    "10_0_100000_9": "10_0_50000_9",
    "10_0_100000_8": "10_0_50000_8",
    "10_0_100000_13": "10_0_50000_13",
    "10_0_100000_11": "10_0_50000_11",
    "10_0_100000_10": "10_0_50000_10",
}

REMOVAL_OPTIONS = [
    # option #0: no trace removal
    [],
    # option #1: removal of the only trace with an input rate of 10,000 records/sec
    ["1_0_10000_17"],
    # option #2: additional removal of the T2 trace having very few normal records ("generalization" setup)
    ["1_0_10000_17", "1_2_100000_68"],
]
PRUNING_OPTIONS = json.load(
    open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "pruned_ranges_options.json",
        ),
        "r",
    )
)
