#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=deep_sad

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
transform_fit_normal_only=${12}  # "no" for DeepSAD, "yes" for DeepSVDD

# make_window_datasets args
window_size=${13}
window_step=${14}
downsampling_size=${15}
downsampling_step=${16}
downsampling_func=${17}
spark_balancing=${18}
normal_data_prop=${19}
anomaly_augmentation=${20}
ano_augment_n_per_normal=${21}
window_min_ano_coverage=1.0

# train_window_model args
normal_as_unlabeled=${22}
remove_anomalies=${23}  # False for DeepSAD, True for DeepSVDD
oversample_anomalies=${24}
n_ano_per_normal=${25}
network=${26}
enc_conv1d_filters=${27}
enc_conv1d_kernel_sizes=${28}
enc_conv1d_strides=${29}
conv1d_batch_norm=${30}
hidden_dims=${31}
rep_dim=${32}
eta=${33}
ae_out_act=${34}
pretrain_optimizer=${35}
pretrain_adamw_weight_decay=${36}
pretrain_learning_rate=${37}
pretrain_lr_milestones=${38}
pretrain_batch_size=${39}
pretrain_n_epochs=${40}
pretrain_early_stopping_target=${41}
pretrain_early_stopping_patience=${42}
optimizer=${43}
adamw_weight_decay=${44}
learning_rate=${45}
lr_milestones=${46}
batch_size=${47}
n_epochs=${48}
early_stopping_target=${49}
early_stopping_patience=${50}
fix_weights_init=${51}

# evaluate_online_scorer args
ignored_anomaly_labels="7"
ignored_delayed_window=$((window_size-1))

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" \
"$data_pruning_idx" "$val_prop" "$train_val_split")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$bundle_idx" "$transform_chain" "$regular_scaling_method" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$spark_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
if [ "$window_min_ano_coverage" != "0.0" ]; then
  make_window_datasets_id="${make_window_datasets_id}_${window_min_ano_coverage}"
fi
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$normal_as_unlabeled" "$remove_anomalies" "$oversample_anomalies" \
"$n_ano_per_normal" "$network" "$enc_conv1d_filters" "$enc_conv1d_kernel_sizes" "$enc_conv1d_strides" "$conv1d_batch_norm" \
"$hidden_dims" "$rep_dim" "$eta" "$ae_out_act" "$pretrain_optimizer" "$pretrain_adamw_weight_decay" "$pretrain_learning_rate" \
"$pretrain_lr_milestones" "$pretrain_batch_size" "$pretrain_n_epochs" "$pretrain_early_stopping_target" "$pretrain_early_stopping_patience" \
"$optimizer" "$adamw_weight_decay" "$learning_rate" "$lr_milestones" "$batch_size" "$n_epochs" "$early_stopping_target" \
"$early_stopping_patience" "$fix_weights_init")
echo "train_window_model_id: ${train_window_model_id}"
evaluate_online_scorer_id=$(get_evaluate_online_scorer_id "$ignored_anomaly_labels" "$ignored_delayed_window")
echo "evaluate_online_scorer_id: ${evaluate_online_scorer_id}"

for scores_avg_beta in 0 0.8 0.9 0.95 0.96667 0.975 0.98 0.98333 0.9875 0.99167 0.99375 0.995; do
  train_online_scorer_id=$(get_train_online_scorer_id "$scores_avg_beta")
  echo "train_online_scorer_id: ${train_online_scorer_id}"
  exathlon \
  dataset="$dataset_name" \
  detector="$detector_name" \
  step=evaluate_online_scorer \
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
  detector.make_window_datasets.window_manager.downsampling_size="$downsampling_size" \
  detector.make_window_datasets.window_manager.downsampling_step="$downsampling_step" \
  detector.make_window_datasets.window_manager.downsampling_func="$downsampling_func" \
  detector.make_window_datasets.window_manager.window_min_ano_coverage="$window_min_ano_coverage" \
  detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
  detector.make_window_datasets.window_manager.dataset_name.eq__spark.spark_balancing="$spark_balancing" \
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
  detector.train_window_model.fix_weights_init="$fix_weights_init" \
  detector.train_online_scorer.step_id="$train_online_scorer_id" \
  detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
  detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
  detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
  detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
done
