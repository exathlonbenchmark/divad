#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=vae

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
type_=${18}
enc_conv1d_filters=${19}
enc_conv1d_kernel_sizes=${20}
enc_conv1d_strides=${21}
conv1d_pooling=${22}
conv1d_batch_norm=${23}
enc_n_hidden_neurons=${24}
dec_n_hidden_neurons=${25}
dec_conv1d_filters=${26}
dec_conv1d_kernel_sizes=${27}
dec_conv1d_strides=${28}
latent_dim=${29}
activation_rec=${30}
rec_unit_type=${31}
dec_output_dist=${32}
input_dropout=${33}
hidden_dropout=${34}
kl_weight=${35}
optimizer=${36}
learning_rate=${37}
weight_decay=${38}
softplus_scale=${39}
n_epochs=${40}
early_stopping_target=${41}
early_stopping_patience=${42}
shuffling_buffer_prop=${43}
include_extended_effect=True

# train_window_scorer args
reco_prob_n_samples=${44}
scores_avg_beta=${45}

# evaluate_online_scorer args
ignored_anomaly_labels="7"
ignored_delayed_window=$((window_size-1))
include_extended_effect=True

make_datasets_id=$(get_make_datasets_id "$setup" "$generalization_test_prop" "$app_ids" "$label_as_unknown" "$trace_removal_idx" "$data_pruning_idx" "$val_prop" "$train_val_split")
if [ "$include_extended_effect" == "False" ]; then
  make_datasets_id="${make_datasets_id}_no_eei"
fi
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$bundle_idx" "$transform_chain" "$regular_scaling_method")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$spark_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$type_" "$enc_conv1d_filters" "$enc_conv1d_kernel_sizes" \
"$enc_conv1d_strides" "$conv1d_pooling" "$conv1d_batch_norm" "$enc_n_hidden_neurons" "$dec_n_hidden_neurons" "$dec_conv1d_filters" \
"$dec_conv1d_kernel_sizes" "$dec_conv1d_strides" "$latent_dim" "$activation_rec" "$rec_unit_type" "$dec_output_dist" "$input_dropout" \
"$hidden_dropout" "$kl_weight" "$optimizer" "$learning_rate" "$weight_decay" "$softplus_scale" "$n_epochs" "$early_stopping_target" \
"$early_stopping_patience" "$shuffling_buffer_prop")
echo "train_window_model_id: ${train_window_model_id}"
train_window_scorer_id=$(get_train_window_scorer_id "$detector_name" "$reco_prob_n_samples")
echo "train_window_scorer_id: ${train_window_scorer_id}"
evaluate_online_scorer_id=$(get_evaluate_online_scorer_id "$ignored_anomaly_labels" "$ignored_delayed_window")
echo "evaluate_online_scorer_id: ${evaluate_online_scorer_id}"

# for scores_avg_beta in 0 0.8 0.9 0.95 0.96667 0.975 0.98 0.98333 0.9875 0.99167 0.99375 0.995; do
if [ "$scores_avg_beta" == "-1" ] || [ "$scores_avg_beta" == "-1.0" ]; then
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
    dataset.make_datasets.data_manager.include_extended_effect="$include_extended_effect" \
    dataset.make_datasets.data_manager.trace_removal_idx="$trace_removal_idx" \
    dataset.make_datasets.data_manager.data_pruning_idx="$data_pruning_idx" \
    dataset.make_datasets.data_manager.val_prop.value="$val_prop" \
    dataset.make_datasets.data_manager.val_prop.gt__0.train_val_split.value="$train_val_split" \
    dataset.build_features.step_id="$build_features_id" \
    dataset.build_features.feature_crafter.bundle_idx="$bundle_idx" \
    dataset.build_features.transform_chain.value="$transform_chain" \
    dataset.build_features.transform_fit_normal_only="yes" \
    dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="$regular_scaling_method" \
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
    detector.train_window_model.type_="$type_" \
    detector.train_window_model.enc_conv1d_filters="$enc_conv1d_filters" \
    detector.train_window_model.enc_conv1d_kernel_sizes="$enc_conv1d_kernel_sizes" \
    detector.train_window_model.enc_conv1d_strides="$enc_conv1d_strides" \
    detector.train_window_model.conv1d_pooling="$conv1d_pooling" \
    detector.train_window_model.conv1d_batch_norm="$conv1d_batch_norm" \
    detector.train_window_model.enc_n_hidden_neurons="$enc_n_hidden_neurons" \
    detector.train_window_model.dec_n_hidden_neurons="$dec_n_hidden_neurons" \
    detector.train_window_model.dec_conv1d_filters="$dec_conv1d_filters" \
    detector.train_window_model.dec_conv1d_kernel_sizes="$dec_conv1d_kernel_sizes" \
    detector.train_window_model.dec_conv1d_strides="$dec_conv1d_strides" \
    detector.train_window_model.latent_dim="$latent_dim" \
    detector.train_window_model.activation_rec="$activation_rec" \
    detector.train_window_model.rec_unit_type="$rec_unit_type" \
    detector.train_window_model.dec_output_dist="$dec_output_dist" \
    detector.train_window_model.input_dropout="$input_dropout" \
    detector.train_window_model.hidden_dropout="$hidden_dropout" \
    detector.train_window_model.learning_rate="$learning_rate" \
    detector.train_window_model.optimizer.value="$optimizer" \
    detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$weight_decay" \
    detector.train_window_model.softplus_scale="$softplus_scale" \
    detector.train_window_model.n_epochs="$n_epochs" \
    detector.train_window_model.early_stopping_target="$early_stopping_target" \
    detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
    detector.train_window_model.shuffling_buffer_prop="$shuffling_buffer_prop" \
    detector.train_window_scorer.step_id="$train_window_scorer_id" \
    detector.train_window_scorer.reco_prob_n_samples="$reco_prob_n_samples" \
    detector.train_online_scorer.step_id="$train_online_scorer_id" \
    detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
    detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
    detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
    detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
  done
else
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
  dataset.make_datasets.data_manager.include_extended_effect="$include_extended_effect" \
  dataset.make_datasets.data_manager.trace_removal_idx="$trace_removal_idx" \
  dataset.make_datasets.data_manager.data_pruning_idx="$data_pruning_idx" \
  dataset.make_datasets.data_manager.val_prop.value="$val_prop" \
  dataset.make_datasets.data_manager.val_prop.gt__0.train_val_split.value="$train_val_split" \
  dataset.build_features.step_id="$build_features_id" \
  dataset.build_features.feature_crafter.bundle_idx="$bundle_idx" \
  dataset.build_features.transform_chain.value="$transform_chain" \
  dataset.build_features.transform_fit_normal_only="yes" \
  dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="$regular_scaling_method" \
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
  detector.train_window_model.type_="$type_" \
  detector.train_window_model.enc_conv1d_filters="$enc_conv1d_filters" \
  detector.train_window_model.enc_conv1d_kernel_sizes="$enc_conv1d_kernel_sizes" \
  detector.train_window_model.enc_conv1d_strides="$enc_conv1d_strides" \
  detector.train_window_model.conv1d_pooling="$conv1d_pooling" \
  detector.train_window_model.conv1d_batch_norm="$conv1d_batch_norm" \
  detector.train_window_model.enc_n_hidden_neurons="$enc_n_hidden_neurons" \
  detector.train_window_model.dec_n_hidden_neurons="$dec_n_hidden_neurons" \
  detector.train_window_model.dec_conv1d_filters="$dec_conv1d_filters" \
  detector.train_window_model.dec_conv1d_kernel_sizes="$dec_conv1d_kernel_sizes" \
  detector.train_window_model.dec_conv1d_strides="$dec_conv1d_strides" \
  detector.train_window_model.latent_dim="$latent_dim" \
  detector.train_window_model.activation_rec="$activation_rec" \
  detector.train_window_model.rec_unit_type="$rec_unit_type" \
  detector.train_window_model.dec_output_dist="$dec_output_dist" \
  detector.train_window_model.input_dropout="$input_dropout" \
  detector.train_window_model.hidden_dropout="$hidden_dropout" \
  detector.train_window_model.learning_rate="$learning_rate" \
  detector.train_window_model.optimizer.value="$optimizer" \
  detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$weight_decay" \
  detector.train_window_model.softplus_scale="$softplus_scale" \
  detector.train_window_model.n_epochs="$n_epochs" \
  detector.train_window_model.early_stopping_target="$early_stopping_target" \
  detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
  detector.train_window_model.shuffling_buffer_prop="$shuffling_buffer_prop" \
  detector.train_window_scorer.step_id="$train_window_scorer_id" \
  detector.train_window_scorer.reco_prob_n_samples="$reco_prob_n_samples" \
  detector.train_online_scorer.step_id="$train_online_scorer_id" \
  detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
  detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
  detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
  detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
fi
