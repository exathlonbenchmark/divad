#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
detector_name=tranad
step_name=train_window_model

# make_datasets args
app_ids=$1
test_app_ids=$2
val_prop=$3

# build_features args
transform_chain=$4
regular_scaling_method=$5
transform_fit_normal_only=yes

# make_window_datasets args
window_size=$6
window_step=$7
downsampling_size=$8
downsampling_step=$9
downsampling_func=${10}
asd_balancing="none"  # never balance as we need sequence information
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
dim_feedforward=${11}
last_activation=${12}
optimizer=${13}
adamw_weight_decay=${14}
learning_rate=${15}
batch_size=${16}
n_epochs=${17}
early_stopping_target=${18}
early_stopping_patience=${19}

make_datasets_id=$(get_make_datasets_id "$app_ids" "$test_app_ids" "$val_prop")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$transform_chain" "$regular_scaling_method" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$asd_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$dim_feedforward" "$last_activation" "$optimizer" \
"$adamw_weight_decay" "$learning_rate" "$batch_size" "$n_epochs" "$early_stopping_target" "$early_stopping_patience")
echo "train_window_model_id: ${train_window_model_id}"

exathlon \
dataset="$dataset_name" \
detector="$detector_name" \
step="$step_name" \
dataset.make_datasets.step_id="$make_datasets_id" \
dataset.make_datasets.data_manager.app_ids="$app_ids" \
dataset.make_datasets.data_manager.test_app_ids="$test_app_ids" \
dataset.make_datasets.data_manager.val_prop="$val_prop" \
dataset.build_features.step_id="$build_features_id" \
dataset.build_features.transform_chain.value="$transform_chain" \
dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="$regular_scaling_method" \
dataset.build_features.transform_fit_normal_only="$transform_fit_normal_only" \
detector.make_window_datasets.step_id="$make_window_datasets_id" \
detector.make_window_datasets.window_manager.window_size="$window_size" \
detector.make_window_datasets.window_manager.window_step="$window_step" \
detector.make_window_datasets.window_manager.downsampling_size="$downsampling_size" \
detector.make_window_datasets.window_manager.downsampling_step="$downsampling_step" \
detector.make_window_datasets.window_manager.downsampling_func="$downsampling_func" \
detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
detector.make_window_datasets.window_manager.dataset_name.eq__asd.asd_balancing="$asd_balancing" \
detector.make_window_datasets.window_manager.normal_data_prop.value="$normal_data_prop" \
detector.make_window_datasets.window_manager.anomaly_augmentation.value="$anomaly_augmentation" \
detector.make_window_datasets.window_manager.anomaly_augmentation.ne__none.ano_augment_n_per_normal="$ano_augment_n_per_normal" \
detector.train_window_model.step_id="$train_window_model_id" \
detector.train_window_model.dim_feedforward="$dim_feedforward" \
detector.train_window_model.last_activation="$last_activation" \
detector.train_window_model.optimizer.value="$optimizer" \
detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$adamw_weight_decay" \
detector.train_window_model.learning_rate="$learning_rate" \
detector.train_window_model.batch_size="$batch_size" \
detector.train_window_model.n_epochs="$n_epochs" \
detector.train_window_model.early_stopping_target="$early_stopping_target" \
detector.train_window_model.early_stopping_patience="$early_stopping_patience"