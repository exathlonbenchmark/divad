#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
step_name=make_datasets

app_ids=$1
test_app_ids=$2
val_prop=$3

make_datasets_id=$(get_make_datasets_id "$app_ids" "$test_app_ids" "$val_prop")
echo "make_datasets_id: ${make_datasets_id}"

exathlon \
dataset="$dataset_name" \
step="$step_name" \
dataset.make_datasets.step_id="$make_datasets_id" \
dataset.make_datasets.data_manager.app_ids="$app_ids" \
dataset.make_datasets.data_manager.test_app_ids="$test_app_ids" \
dataset.make_datasets.data_manager.val_prop="$val_prop"