#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
detector_name=deep_sad
step_name=train_window_model

# make_datasets args
app_ids=$1
test_app_ids=$2
val_prop=$3

# build_features args
transform_chain=$4
regular_scaling_method=$5
transform_fit_normal_only=$6  # "no" for DeepSAD, "yes" for DeepSVDD

# make_window_datasets args
window_size=$7
window_step=$8
downsampling_size=$9
downsampling_step=${10}
downsampling_func=${11}
asd_balancing=${12}
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
normal_as_unlabeled=${13}
remove_anomalies=${14}  # False for DeepSAD, True for DeepSVDD
oversample_anomalies=${15}
n_ano_per_normal=${16}
network=${17}
enc_conv1d_filters=${18}
enc_conv1d_kernel_sizes=${19}
enc_conv1d_strides=${20}
conv1d_batch_norm=${21}
hidden_dims=${22}
rep_dim=${23}
eta=${24}
ae_out_act=${25}
pretrain_optimizer=${26}
pretrain_adamw_weight_decay=${27}
pretrain_learning_rate=${28}
pretrain_lr_milestones=${29}
pretrain_batch_size=${30}
pretrain_n_epochs=${31}
pretrain_early_stopping_target=${32}
pretrain_early_stopping_patience=${33}
optimizer=${34}
adamw_weight_decay=${35}
learning_rate=${36}
lr_milestones=${37}
batch_size=${38}
n_epochs=${39}
early_stopping_target=${40}
early_stopping_patience=${41}
fix_weights_init=${42}

make_datasets_id=$(get_make_datasets_id "$app_ids" "$test_app_ids" "$val_prop")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$transform_chain" "$regular_scaling_method" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$asd_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$normal_as_unlabeled" "$remove_anomalies" "$oversample_anomalies" \
"$n_ano_per_normal" "$network" "$enc_conv1d_filters" "$enc_conv1d_kernel_sizes" "$enc_conv1d_strides" "$conv1d_batch_norm" \
"$hidden_dims" "$rep_dim" "$eta" "$ae_out_act" "$pretrain_optimizer" "$pretrain_adamw_weight_decay" "$pretrain_learning_rate" \
"$pretrain_lr_milestones" "$pretrain_batch_size" "$pretrain_n_epochs" "$pretrain_early_stopping_target" "$pretrain_early_stopping_patience" \
"$optimizer" "$adamw_weight_decay" "$learning_rate" "$lr_milestones" "$batch_size" "$n_epochs" "$early_stopping_target" \
"$early_stopping_patience" "$fix_weights_init")
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
detector.train_window_model.normal_as_unlabeled="$normal_as_unlabeled" \
detector.train_window_model.remove_anomalies="$remove_anomalies" \
detector.train_window_model.oversample_anomalies.value="$oversample_anomalies" \
detector.train_window_model.oversample_anomalies.eq__True.n_ano_per_normal="$n_ano_per_normal" \
detector.train_window_model.network.value="$network" \
detector.train_window_model.network.eq__rec.enc_conv1d_filters="$enc_conv1d_filters" \
detector.train_window_model.network.eq__rec.enc_conv1d_kernel_sizes="$enc_conv1d_kernel_sizes" \
detector.train_window_model.network.eq__rec.enc_conv1d_strides="$enc_conv1d_strides" \
detector.train_window_model.network.eq__rec.conv1d_batch_norm="$conv1d_batch_norm" \
detector.train_window_model.hidden_dims="$hidden_dims" \
detector.train_window_model.rep_dim="$rep_dim" \
detector.train_window_model.eta="$eta" \
detector.train_window_model.ae_out_act="$ae_out_act" \
detector.train_window_model.pretrain_optimizer="$pretrain_optimizer" \
detector.train_window_model.pretrain_adamw_weight_decay="$pretrain_adamw_weight_decay" \
detector.train_window_model.pretrain_learning_rate="$pretrain_learning_rate" \
detector.train_window_model.pretrain_lr_milestones="$pretrain_lr_milestones" \
detector.train_window_model.pretrain_batch_size="$pretrain_batch_size" \
detector.train_window_model.pretrain_n_epochs="$pretrain_n_epochs" \
detector.train_window_model.pretrain_early_stopping_target="$pretrain_early_stopping_target" \
detector.train_window_model.pretrain_early_stopping_patience="$pretrain_early_stopping_patience" \
detector.train_window_model.optimizer="$optimizer" \
detector.train_window_model.adamw_weight_decay="$adamw_weight_decay" \
detector.train_window_model.learning_rate="$learning_rate" \
detector.train_window_model.lr_milestones="$lr_milestones" \
detector.train_window_model.batch_size="$batch_size" \
detector.train_window_model.n_epochs="$n_epochs" \
detector.train_window_model.early_stopping_target="$early_stopping_target" \
detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
detector.train_window_model.fix_weights_init="$fix_weights_init"
