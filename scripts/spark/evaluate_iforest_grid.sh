#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

# make_datasets args
setup=$1
generalization_test_prop=$2
app_ids=$3
label_as_unknown=$4
trace_removal_idx=$5
data_pruning_idx=$6
val_prop=$7
train_val_split=$8

# build_features args
bundle_idx=$9
transform_chain=${10}

# make_window_datasets args
window_size=${11}
spark_balancing=${12}

# train_window_scorer args
drop_anomalies=${13}
max_samples=${14}
max_features=${15}

for n_estimators in 50 100 200 500 1000; do
  ./evaluate_iforest.sh "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" \
  "$data_pruning_idx" "$val_prop" "$train_val_split" "$bundle_idx" "$transform_chain" "$window_size" "$spark_balancing" \
  "$drop_anomalies" "$n_estimators" "$max_samples" "$max_features"
done
