#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=spark
detector_name=divad

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
transform_fit_normal_only=yes

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
pzy_dist=${19}
pzy_kl_n_samples=${20}
pzy_gm_n_components=${21}
pzy_gm_softplus_scale=${22}
pzy_vamp_n_components=${23}
qz_x_conv1d_filters=${24}
qz_x_conv1d_kernel_sizes=${25}
qz_x_conv1d_strides=${26}
qz_x_n_hidden=${27}
pzd_d_n_hidden=${28}
px_z_conv1d_filters=${29}
px_z_conv1d_kernel_sizes=${30}
px_z_conv1d_strides=${31}
px_z_n_hidden=${32}
time_freq=${33}
sample_normalize_x=${34}
sample_normalize_mag=${35}
apply_hann=${36}
n_freq_modes=${37}
phase_encoding=${38}
phase_cyclical_decoding=${39}
qz_x_freq_conv1d_filters=${40}
qz_x_freq_conv1d_kernel_sizes=${41}
qz_x_freq_conv1d_strides=${42}
px_z_freq_conv1d_filters=${43}
px_z_freq_conv1d_kernel_sizes=${44}
px_z_freq_conv1d_strides=${45}
latent_dim=${46}
rec_unit_type=${47}
activation_rec=${48}
conv1d_pooling=${49}
conv1d_batch_norm=${50}
rec_weight_decay=${51}
weight_decay=${52}
dropout=${53}
dec_output_dist=${54}
min_beta=${55}
max_beta=${56}
beta_n_epochs=${57}
loss_weighting=${58}
d_classifier_weight=${59}
optimizer=${60}
learning_rate=${61}
lr_scheduling=${62}  # either "none", "pw_constant" or "one_cycle"
lrs_pwc_red_factor=${63}
lrs_pwc_red_freq=${64}
adamw_weight_decay=${65}
softplus_scale=${66}
batch_size=${67}
n_epochs=${68}
early_stopping_target=${69}
early_stopping_patience=${70}
include_extended_effect=True
domain_key="$spark_balancing"

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
train_window_model_id=$(get_train_window_model_id "$detector_name" "$type_" "$pzy_dist" "$pzy_kl_n_samples" "$pzy_gm_n_components" "$pzy_gm_softplus_scale" \
"$pzy_vamp_n_components" "$qz_x_conv1d_filters" "$qz_x_conv1d_kernel_sizes" "$qz_x_conv1d_strides" "$qz_x_n_hidden" "$pzd_d_n_hidden" "$px_z_conv1d_filters" \
"$px_z_conv1d_kernel_sizes" "$px_z_conv1d_strides" "$px_z_n_hidden" "$time_freq" "$sample_normalize_x" "$sample_normalize_mag" "$apply_hann" \
"$n_freq_modes" "$phase_encoding" "$phase_cyclical_decoding" "$qz_x_freq_conv1d_filters" "$qz_x_freq_conv1d_kernel_sizes" "$qz_x_freq_conv1d_strides" \
"$px_z_freq_conv1d_filters" "$px_z_freq_conv1d_kernel_sizes" "$px_z_freq_conv1d_strides" "$latent_dim" "$rec_unit_type" "$activation_rec" \
"$conv1d_pooling" "$conv1d_batch_norm" "$rec_weight_decay" "$weight_decay" "$dropout" "$dec_output_dist" "$min_beta" "$max_beta" "$beta_n_epochs" \
"$loss_weighting" "$d_classifier_weight" "$optimizer" "$learning_rate" "$lr_scheduling" "$lrs_pwc_red_factor" "$lrs_pwc_red_freq" \
"$adamw_weight_decay" "$softplus_scale" "$batch_size" "$n_epochs" "$early_stopping_target" "$early_stopping_patience")
if [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nh_normal_0.0_5.0_25_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_time_freq_dropout_hann_kl_5_lin_25_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_5.0_5.0_100_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_tf_no_dropout_no_hann_kl_5_cst_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_0.0_5.0_25_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_tf_no_dropout_no_hann_kl_5_lin_25_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_normal_5.0_5.0_100_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_tf_dropout_no_hann_kl_5_cst_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_normal_0.0_5.0_25_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_tf_dropout_no_hann_kl_5_linear_25_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nh_normal_5.0_5.0_100_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_time_freq_dropout_hann_kl_5_cst_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_0.0_5.0_25_2000.0_adam_"*"_128_"*"_loss_"*"" ]]; then
  train_window_model_id="divad_tf_no_dropout_no_hann_kl_5_lin_25_no_es_${n_epochs}_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_np_nb_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_0.0_5.0_25_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_tf_no_dropout_no_hann_kl_5_lin_25_nbn_np_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_0.0_10.0_25_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_tf_no_dropout_no_hann_kl_10_lin_25_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nh_0.0_normal_0.0_5.0_25_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_time_freq_no_dropout_hann_kl_5_lin_25_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_2_32_"*"_32_237_5_5_2_1_gru_32_32_5_5_1_1_32_5_5_1_1_nh_0.0_normal_5.0_5.0_100_2000.0_adam_"*"_128_300_val_loss_100" ]]; then
  train_window_model_id="divad_time_freq_no_dropout_hann_kl_5_cst_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
# new models
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_1_64_32_32_237_5_5_1_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_0.0_normal_5.0_5.0_100_100000.0_adamw_"*"_0.01_128_300_val_loss_100" ]]; then
  train_window_model_id="tf_divad_32_gm4_nofftnorm_poolbn_adamw_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_1_64_32_32_237_5_5_1_1_gru_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_5.0_5.0_100_100000.0_adamw_"*"_0.01_128_300_val_loss_100" ]]; then
  train_window_model_id="tf_divad_32_gm4_fftnorm_poolbn_adamw_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_1_64_32_32_237_5_5_1_1_gru_np_nb_32_32_5_5_1_1_32_5_5_1_1_nsn_0.0_normal_5.0_5.0_100_100000.0_adamw_"*"_0.01_128_300_val_loss_100" ]]; then
  train_window_model_id="tf_divad_32_gm4_nofftnorm_adamw_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
elif [[ "$train_window_model_id" == "rec_divad_gm_"*"_32_32_5_5_1_1_64_32_32_237_5_5_1_1_gru_np_nb_32_32_5_5_1_1_32_5_5_1_1_nsn_mn_0.0_normal_5.0_5.0_100_100000.0_adamw_"*"_0.01_128_300_val_loss_100" ]]; then
  train_window_model_id="tf_divad_32_gm4_fftnorm_adamw_${pzy_gm_n_components}_${latent_dim}_${learning_rate}"
fi
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
dataset.make_datasets.data_manager.include_extended_effect="$include_extended_effect" \
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
detector.train_window_model.type_.value="$type_" \
detector.train_window_model.domain_key="$domain_key" \
detector.train_window_model.pzy_dist.value="$pzy_dist" \
detector.train_window_model.pzy_dist.ne__standard.pzy_kl_n_samples="$pzy_kl_n_samples" \
detector.train_window_model.pzy_dist.eq__gm.pzy_gm_n_components="$pzy_gm_n_components" \
detector.train_window_model.pzy_dist.eq__gm.pzy_gm_softplus_scale="$pzy_gm_softplus_scale" \
detector.train_window_model.pzy_dist.eq__vamp.pzy_vamp_n_components="$pzy_vamp_n_components" \
detector.train_window_model.qz_x_conv1d_filters="$qz_x_conv1d_filters" \
detector.train_window_model.qz_x_conv1d_kernel_sizes="$qz_x_conv1d_kernel_sizes" \
detector.train_window_model.qz_x_conv1d_strides="$qz_x_conv1d_strides" \
detector.train_window_model.qz_x_n_hidden="$qz_x_n_hidden" \
detector.train_window_model.pzd_d_n_hidden="$pzd_d_n_hidden" \
detector.train_window_model.px_z_conv1d_filters="$px_z_conv1d_filters" \
detector.train_window_model.px_z_conv1d_kernel_sizes="$px_z_conv1d_kernel_sizes" \
detector.train_window_model.px_z_conv1d_strides="$px_z_conv1d_strides" \
detector.train_window_model.px_z_n_hidden="$px_z_n_hidden" \
detector.train_window_model.time_freq.value="$time_freq" \
detector.train_window_model.time_freq.eq__True.sample_normalize_x="$sample_normalize_x" \
detector.train_window_model.time_freq.eq__True.sample_normalize_mag="$sample_normalize_mag" \
detector.train_window_model.time_freq.eq__True.apply_hann="$apply_hann" \
detector.train_window_model.time_freq.eq__True.n_freq_modes="$n_freq_modes" \
detector.train_window_model.time_freq.eq__True.phase_encoding="$phase_encoding" \
detector.train_window_model.time_freq.eq__True.phase_cyclical_decoding="$phase_cyclical_decoding" \
detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_filters="$qz_x_freq_conv1d_filters" \
detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_kernel_sizes="$qz_x_freq_conv1d_kernel_sizes" \
detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_strides="$qz_x_freq_conv1d_strides" \
detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_filters="$px_z_freq_conv1d_filters" \
detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_kernel_sizes="$px_z_freq_conv1d_kernel_sizes" \
detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_strides="$px_z_freq_conv1d_strides" \
detector.train_window_model.latent_dim="$latent_dim" \
detector.train_window_model.type_.eq__rec.rec_unit_type="$rec_unit_type" \
detector.train_window_model.type_.eq__rec.activation_rec="$activation_rec" \
detector.train_window_model.type_.eq__rec.conv1d_pooling="$conv1d_pooling" \
detector.train_window_model.type_.eq__rec.conv1d_batch_norm="$conv1d_batch_norm" \
detector.train_window_model.type_.eq__rec.rec_weight_decay="$rec_weight_decay" \
detector.train_window_model.weight_decay="$weight_decay" \
detector.train_window_model.dropout="$dropout" \
detector.train_window_model.dec_output_dist="$dec_output_dist" \
detector.train_window_model.min_beta="$min_beta" \
detector.train_window_model.max_beta="$max_beta" \
detector.train_window_model.beta_n_epochs="$beta_n_epochs" \
detector.train_window_model.loss_weighting.value="$loss_weighting" \
detector.train_window_model.loss_weighting.eq__fixed.d_classifier_weight="$d_classifier_weight" \
detector.train_window_model.learning_rate="$learning_rate" \
detector.train_window_model.lr_scheduling="$lr_scheduling" \
detector.train_window_model.lrs_pwc_red_factor="$lrs_pwc_red_factor" \
detector.train_window_model.lrs_pwc_red_freq="$lrs_pwc_red_freq" \
detector.train_window_model.optimizer.value="$optimizer" \
detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$adamw_weight_decay" \
detector.train_window_model.softplus_scale="$softplus_scale" \
detector.train_window_model.batch_size="$batch_size" \
detector.train_window_model.n_epochs="$n_epochs" \
detector.train_window_model.early_stopping_target="$early_stopping_target" \
detector.train_window_model.early_stopping_patience="$early_stopping_patience"