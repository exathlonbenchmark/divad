#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=omni_anomaly

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
transform_fit_normal_only="yes"

# make_window_datasets args
window_size=${12}
window_step=${13}
downsampling_size=${14}
downsampling_step=${15}
downsampling_func=${16}
spark_balancing=${17}
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
z_dim=${18}
dense_dim=${19}
rnn_num_hidden=${20}
use_connected_z_p=${21}
use_connected_z_q=${22}
std_epsilon=${23}
posterior_flow_type=${24}
nf_layers=${25}
initial_lr=${26}
lr_anneal_epoch_freq=${27}
lr_anneal_factor=${28}
gradient_clip_norm=${29}
valid_step_freq=${30}
max_epoch=${31}
batch_size=${32}
test_batch_size=${33}
test_n_z=${34}

# evaluate_online_scorer args
ignored_anomaly_labels="7"
ignored_delayed_window=$((window_size-1))

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" "$data_pruning_idx" "$val_prop" "$train_val_split")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$bundle_idx" "$transform_chain" "$regular_scaling_method")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$spark_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$z_dim" "$dense_dim" "$rnn_num_hidden" \
"$use_connected_z_p" "$use_connected_z_q" "$std_epsilon" "$posterior_flow_type" "$nf_layers" "$initial_lr" \
"$lr_anneal_epoch_freq" "$lr_anneal_factor" "$gradient_clip_norm" "$valid_step_freq" "$max_epoch" "$batch_size" \
"$test_batch_size" "$test_n_z")
echo "train_window_model_id: ${train_window_model_id}"
evaluate_online_scorer_id=$(get_evaluate_online_scorer_id "$ignored_anomaly_labels" "$ignored_delayed_window")
echo "evaluate_online_scorer_id: ${evaluate_online_scorer_id}"

for scores_avg_beta in 0 0.8 0.9 0.95 0.96667 0.975 0.98 0.98333 0.9875 0.99167 0.99375 0.995; do
  train_online_scorer_id=$(get_train_online_scorer_id "$scores_avg_beta")
  echo "train_online_scorer_id: ${train_online_scorer_id}"
  # in Python 3.6: "no module named run" error when running `exathlon`
  python ../../exathlon/run.py \
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
  detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
  detector.make_window_datasets.window_manager.dataset_name.eq__spark.spark_balancing="$spark_balancing" \
  detector.make_window_datasets.window_manager.normal_data_prop.value="$normal_data_prop" \
  detector.make_window_datasets.window_manager.anomaly_augmentation.value="$anomaly_augmentation" \
  detector.make_window_datasets.window_manager.anomaly_augmentation.ne__none.ano_augment_n_per_normal="$ano_augment_n_per_normal" \
  detector.train_window_model.step_id="$train_window_model_id" \
  detector.train_window_model.z_dim="$z_dim" \
  detector.train_window_model.dense_dim="$dense_dim" \
  detector.train_window_model.rnn_num_hidden="$rnn_num_hidden" \
  detector.train_window_model.use_connected_z_p="$use_connected_z_p" \
  detector.train_window_model.use_connected_z_q="$use_connected_z_q" \
  detector.train_window_model.std_epsilon="$std_epsilon" \
  detector.train_window_model.posterior_flow_type="$posterior_flow_type" \
  detector.train_window_model.nf_layers="$nf_layers" \
  detector.train_window_model.initial_lr="$initial_lr" \
  detector.train_window_model.lr_anneal_epoch_freq="$lr_anneal_epoch_freq" \
  detector.train_window_model.lr_anneal_factor="$lr_anneal_factor" \
  detector.train_window_model.gradient_clip_norm="$gradient_clip_norm" \
  detector.train_window_model.valid_step_freq="$valid_step_freq" \
  detector.train_window_model.max_epoch="$max_epoch" \
  detector.train_window_model.batch_size="$batch_size" \
  detector.train_window_model.test_batch_size="$test_batch_size" \
  detector.train_window_model.test_n_z="$test_n_z" \
  detector.train_online_scorer.step_id="$train_online_scorer_id" \
  detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
  detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
  detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
  detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
done
