#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

setup=$1
generalization_test_prop=$2
app_ids=$3
label_as_unknown=$4
trace_removal_idx=$5
data_pruning_idx=$6
val_prop=$7
train_val_split=$8

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" "$data_pruning_idx" "$val_prop" "$train_val_split")
echo "make_datasets_id: ${make_datasets_id}"

exathlon \
dataset=spark \
step=make_datasets \
dataset.make_datasets.step_id="$make_datasets_id" \
dataset.make_datasets.data_manager.setup.value="$setup" \
dataset.make_datasets.data_manager.setup.eq__generalization.test_prop="$generalization_test_prop" \
dataset.make_datasets.data_manager.app_ids="$app_ids" \
dataset.make_datasets.data_manager.label_as_unknown="$label_as_unknown" \
dataset.make_datasets.data_manager.trace_removal_idx="$trace_removal_idx" \
dataset.make_datasets.data_manager.data_pruning_idx="$data_pruning_idx" \
dataset.make_datasets.data_manager.val_prop.value="$val_prop" \
dataset.make_datasets.data_manager.val_prop.gt__0.train_val_split.value="$train_val_split"
