"""Helper variables and functions to manipulate datasets."""
import os

TRAIN_SET_NAME = "train"
VAL_SET_NAME = "val"
TEST_SET_NAME = "test"
DATASET_NAMES = [TRAIN_SET_NAME, VAL_SET_NAME, TEST_SET_NAME]


def get_dataset_names(input_path: str) -> list:
    dataset_names = []
    for n in DATASET_NAMES:
        for file in os.listdir(input_path):
            if n in file:
                dataset_names.append(n)
                break
    return dataset_names
