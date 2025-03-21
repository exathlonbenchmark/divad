#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=xgboost_

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
regular_scaling_method=${11}
transform_fit_normal_only=no

# make_window_datasets args
window_size=${12}
window_step=${13}
spark_balancing=${14}
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_scorer args
binary_mode=${15}
imbalance_handling=${16}
n_estimators=${17}
max_depth=${18}
min_child_weight=${19}
subsample=${20}
learning_rate=${21}
gamma=${22}
max_delta_step=${23}

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" "$data_pruning_idx" "$val_prop" "$train_val_split")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$bundle_idx" "$transform_chain" "std" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$spark_balancing" "$normal_data_prop" "$anomaly_augmentation" \
"$ano_augment_n_per_normal")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_scorer_id=$(get_train_window_scorer_id "$detector_name" "$binary_mode" "$imbalance_handling" "$n_estimators" "$max_depth" \
"$min_child_weight" "$subsample" "$learning_rate" "$gamma" "$max_delta_step")
echo "train_window_scorer_id: ${train_window_scorer_id}"

exathlon \
dataset="$dataset_name" \
detector="$detector_name" \
step=train_window_scorer \
dataset.make_datasets.step_id="$make_datasets_id" \
dataset.make_datasets.data_manager.setup.value="$setup" \
dataset.make_datasets.data_manager.setup.eq__generalization.test_prop="$generalization_test_prop" \
dataset.make_datasets.data_manager.app_ids="$app_ids" \
dataset.make_datasets.data_manager.label_as_unknown="$label_as_unknown" \
dataset.make_datasets.data_manager.trace_removal_idx="$trace_removal_idx" \
dataset.make_datasets.data_manager.data_pruning_idx="$data_pruning_idx" \
dataset.make_datasets.data_manager.val_prop.value="$val_prop" \
dataset.make_datasets.data_manager.val_prop.gt__0.train_val_split.value="$train_val_split" \
dataset.build_features.step_id="$build_features_id" \
dataset.build_features.feature_crafter.bundle_idx="$bundle_idx" \
dataset.build_features.transform_chain.value="$transform_chain" \
dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="$regular_scaling_method" \
dataset.build_features.transform_fit_normal_only="$transform_fit_normal_only" \
detector.make_window_datasets.step_id="$make_window_datasets_id" \
detector.make_window_datasets.window_manager.window_size="$window_size" \
detector.make_window_datasets.window_manager.window_step="$window_step" \
detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
detector.make_window_datasets.window_manager.dataset_name.eq__spark.spark_balancing="$spark_balancing" \
detector.make_window_datasets.window_manager.normal_data_prop.value="$normal_data_prop" \
detector.make_window_datasets.window_manager.anomaly_augmentation.value="$anomaly_augmentation" \
detector.make_window_datasets.window_manager.anomaly_augmentation.ne__none.ano_augment_n_per_normal="$ano_augment_n_per_normal" \
detector.train_window_scorer.step_id="$train_window_scorer_id" \
detector.train_window_scorer.binary_mode="$binary_mode" \
detector.train_window_scorer.imbalance_handling="$imbalance_handling" \
detector.train_window_scorer.n_estimators="$n_estimators" \
detector.train_window_scorer.max_depth="$max_depth" \
detector.train_window_scorer.min_child_weight="$min_child_weight" \
detector.train_window_scorer.subsample="$subsample" \
detector.train_window_scorer.learning_rate="$learning_rate" \
detector.train_window_scorer.gamma="$gamma" \
detector.train_window_scorer.max_delta_step="$max_delta_step"
