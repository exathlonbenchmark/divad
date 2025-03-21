#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=deep_svdd

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
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
n_hidden_neurons=${13}
output_dim=${14}
input_dropout=${15}
hidden_dropout=${16}
learning_rate=${17}
weight_decay=${18}
n_epochs=${19}
early_stopping_target=${20}
early_stopping_patience=${21}
shuffling_buffer_prop=${22}

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" "$data_pruning_idx" "$val_prop" "$train_val_split")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$bundle_idx" "$transform_chain" "std")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$spark_balancing" "$normal_data_prop" "$anomaly_augmentation" "$ano_augment_n_per_normal")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$n_hidden_neurons" "$output_dim" \
"$input_dropout" "$hidden_dropout" "$learning_rate" "$weight_decay" "$n_epochs" "$early_stopping_target" \
"$early_stopping_patience" "$shuffling_buffer_prop")
echo "train_window_model_id: ${train_window_model_id}"

# for scores_avg_beta in 0 0.8 0.9 0.95 0.96 0.9667 0.98 0.9867 0.99 0.9933 0.9950 0.9967; do
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
  dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="std" \
  detector.make_window_datasets.step_id="$make_window_datasets_id" \
  detector.make_window_datasets.window_manager.window_size="$window_size" \
  detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
  detector.make_window_datasets.window_manager.dataset_name.eq__spark.spark_balancing="$spark_balancing" \
  detector.make_window_datasets.window_manager.normal_data_prop.value="$normal_data_prop" \
  detector.make_window_datasets.window_manager.anomaly_augmentation.value="$anomaly_augmentation" \
  detector.make_window_datasets.window_manager.anomaly_augmentation.ne__none.ano_augment_n_per_normal="$ano_augment_n_per_normal" \
  detector.train_window_model.step_id="$train_window_model_id" \
  detector.train_window_model.n_hidden_neurons="$n_hidden_neurons" \
  detector.train_window_model.output_dim="$output_dim" \
  detector.train_window_model.input_dropout="$input_dropout" \
  detector.train_window_model.hidden_dropout="$hidden_dropout" \
  detector.train_window_model.learning_rate="$learning_rate" \
  detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$weight_decay" \
  detector.train_window_model.n_epochs="$n_epochs" \
  detector.train_window_model.early_stopping_target="$early_stopping_target" \
  detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
  detector.train_window_model.shuffling_buffer_prop="$shuffling_buffer_prop" \
  detector.train_online_scorer.step_id="$train_online_scorer_id" \
  detector.train_online_scorer.scores_avg_beta="$scores_avg_beta"
done
