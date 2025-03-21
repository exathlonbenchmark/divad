#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=lstm_ad

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
transform_fit_normal_only="yes"

# make_window_datasets args
window_size=${11}
window_step=${12}
downsampling_size=${13}
downsampling_step=${14}
downsampling_func=${15}
spark_balancing=${16}
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
conv1d_filters=${17}
conv1d_kernel_sizes=${18}
conv1d_strides=${19}
conv1d_pooling=${20}
conv1d_batch_norm=${21}
unit_type=${22}
n_hidden_neurons=${23}
n_forward=${24}
learning_rate=${25}
weight_decay=${26}
n_epochs=${27}
early_stopping_target=${28}
early_stopping_patience=${29}
shuffling_buffer_prop=${30}

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" "$data_pruning_idx" "$val_prop" "$train_val_split")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$bundle_idx" "$transform_chain" "std")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$spark_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$conv1d_filters" "$conv1d_kernel_sizes" "$conv1d_strides" \
"$conv1d_pooling" "$conv1d_batch_norm" "$unit_type" "$n_hidden_neurons" "$n_forward" "$learning_rate" "$weight_decay" "$n_epochs" \
"$early_stopping_target" "$early_stopping_patience" "$shuffling_buffer_prop")
echo "train_window_model_id: ${train_window_model_id}"

exathlon \
dataset="$dataset_name" \
detector="$detector_name" \
step=train_window_model \
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
dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="std" \
dataset.build_features.transform_fit_normal_only="$transform_fit_normal_only" \
detector.make_window_datasets.step_id="$make_window_datasets_id" \
detector.make_window_datasets.window_manager.window_size="$window_size" \
detector.make_window_datasets.window_manager.window_step="$window_step" \
detector.make_window_datasets.window_manager.downsampling_size="$downsampling_size" \
detector.make_window_datasets.window_manager.downsampling_step="$downsampling_step" \
detector.make_window_datasets.window_manager.downsampling_func="$downsampling_func" \
detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
detector.make_window_datasets.window_manager.dataset_name.eq__spark.spark_balancing="$spark_balancing" \
detector.make_window_datasets.window_manager.normal_data_prop.value="$normal_data_prop" \
detector.make_window_datasets.window_manager.anomaly_augmentation.value="$anomaly_augmentation" \
detector.make_window_datasets.window_manager.anomaly_augmentation.ne__none.ano_augment_n_per_normal="$ano_augment_n_per_normal" \
detector.train_window_model.step_id="$train_window_model_id" \
detector.train_window_model.conv1d_filters="$conv1d_filters" \
detector.train_window_model.conv1d_kernel_sizes="$conv1d_kernel_sizes" \
detector.train_window_model.conv1d_strides="$conv1d_strides" \
detector.train_window_model.conv1d_pooling="$conv1d_pooling" \
detector.train_window_model.conv1d_batch_norm="$conv1d_batch_norm" \
detector.train_window_model.unit_type="$unit_type" \
detector.train_window_model.n_hidden_neurons="$n_hidden_neurons" \
detector.train_window_model.n_forward="$n_forward" \
detector.train_window_model.learning_rate="$learning_rate" \
detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$weight_decay" \
detector.train_window_model.n_epochs="$n_epochs" \
detector.train_window_model.early_stopping_target="$early_stopping_target" \
detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
detector.train_window_model.shuffling_buffer_prop="$shuffling_buffer_prop"
