#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
detector_name=divad
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
asd_balancing=${11}
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
type_=${12}
pzy_dist=${13}
pzy_kl_n_samples=${14}
pzy_gm_n_components=${15}
pzy_gm_softplus_scale=${16}
pzy_vamp_n_components=${17}
qz_x_conv1d_filters=${18}
qz_x_conv1d_kernel_sizes=${19}
qz_x_conv1d_strides=${20}
qz_x_n_hidden=${21}
pzd_d_n_hidden=${22}
px_z_conv1d_filters=${23}
px_z_conv1d_kernel_sizes=${24}
px_z_conv1d_strides=${25}
px_z_n_hidden=${26}
time_freq=${27}
sample_normalize_x=${28}
sample_normalize_mag=${29}
apply_hann=${30}
n_freq_modes=${31}
phase_encoding=${32}
phase_cyclical_decoding=${33}
qz_x_freq_conv1d_filters=${34}
qz_x_freq_conv1d_kernel_sizes=${35}
qz_x_freq_conv1d_strides=${36}
px_z_freq_conv1d_filters=${37}
px_z_freq_conv1d_kernel_sizes=${38}
px_z_freq_conv1d_strides=${39}
latent_dim=${40}
rec_unit_type=${41}
activation_rec=${42}
conv1d_pooling=${43}
conv1d_batch_norm=${44}
rec_weight_decay=${45}
weight_decay=${46}
dropout=${47}
dec_output_dist=${48}
min_beta=${49}
max_beta=${50}
beta_n_epochs=${51}
loss_weighting=${52}
d_classifier_weight=${53}
optimizer=${54}
learning_rate=${55}
lr_scheduling=${56}  # either "none", "pw_constant" or "one_cycle"
lrs_pwc_red_factor=${57}
lrs_pwc_red_freq=${58}
adamw_weight_decay=${59}
softplus_scale=${60}
batch_size=${61}
n_epochs=${62}
early_stopping_target=${63}
early_stopping_patience=${64}
domain_key="file_name"

make_datasets_id=$(get_make_datasets_id "$app_ids" "$test_app_ids" "$val_prop")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$transform_chain" "$regular_scaling_method" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$asd_balancing" "$normal_data_prop" \
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