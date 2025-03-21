#!/bin/bash

# utilities shared by all datasets

function join_by {
    # credit: https://bit.ly/3nVaCW2
    local d=${1-} f=${2-}
    if shift 2; then
      printf %s "$f" "${@/#/$d}"
    fi
}

function get_make_window_datasets_id(){
    window_size=$1
    window_step=$2
    spark_balancing=$3  # TODO: rename to window_balancing.
    normal_data_prop=$4
    anomaly_augmentation=$5
    ano_augment_n_per_normal=$6
    downsampling_size=$7
    downsampling_step=$8
    downsampling_func=$9
    if [ "$window_step" == "1" ] || [ "$window_step" == "1.0" ]; then
      window_step_str=""
    else
      window_step_str="_${window_step}"
    fi
    if [ "$spark_balancing" == "app-type-rate" ]; then
      spark_balancing_str=""
    else
      spark_balancing_str="_${spark_balancing}"
    fi
    if [ "$normal_data_prop" == "1.0" ]; then
      normal_data_prop_str=""
    else
      normal_data_prop_str="_normal\=${normal_data_prop}"
    fi
    if [ "$anomaly_augmentation" == "none" ]; then
      anomaly_augmentation_str=""
    elif [ "$anomaly_augmentation" == "borderline_smote" ]; then
      anomaly_augmentation_str="_bsm_${ano_augment_n_per_normal}"
    else
      anomaly_augmentation_str="_${anomaly_augmentation}_${ano_augment_n_per_normal}"
    fi
    if [ -z "$downsampling_size" ] || [ "$downsampling_size" == "1" ]; then
      downsampling_str=""
    else
      downsampling_str="_${downsampling_size}_${downsampling_step}"
      if [ "$downsampling_func" != "mean" ]; then
        downsampling_str="${downsampling_str}_${downsampling_func}"
      fi
    fi
    echo "w\=${window_size}${window_step_str}${downsampling_str}${spark_balancing_str}${normal_data_prop_str}${anomaly_augmentation_str}"
}

function get_train_window_model_id() {
    detector=$1
    if [ "$detector" == "autoencoder" ]; then
      type_=$2
      enc_conv1d_filters=$3
      enc_conv1d_kernel_sizes=$4
      enc_conv1d_strides=$5
      conv1d_pooling=$6
      conv1d_batch_norm=$7
      enc_n_hidden_neurons=$8
      latent_dim=$9
      activation_rec=${10}
      rec_unit_type=${11}
      input_dropout=${12}
      hidden_dropout=${13}
      linear_latent_activation=${14}
      rec_latent_type=${15}
      conv_add_dense_for_latent=${16}
      learning_rate=${17}
      weight_decay=${18}
      n_epochs=${19}
      early_stopping_target=${20}
      early_stopping_patience=${21}
      shuffling_buffer_prop=${22}
      if [ "$conv1d_pooling" == "False" ]; then
        conv1d_pooling_str=""
      else
        conv1d_pooling_str="_p"
      fi
      if [ "$conv1d_batch_norm" == "False" ]; then
        conv1d_batch_norm_str=""
      else
        conv1d_batch_norm_str="_bn"
      fi
      if [ "$input_dropout" == "0.0" ] || [ "$input_dropout" == "0" ]; then
        input_dropout_str=""
      else
        input_dropout_str="_${input_dropout}"
      fi
      if [ "$hidden_dropout" == "0.0" ] || [ "$hidden_dropout" == "0" ]; then
        hidden_dropout_str=""
      else
        hidden_dropout_str="_${hidden_dropout}"
      fi
      if [ "$linear_latent_activation" == "False" ]; then
        linear_latent_activation_str=""
      else
        linear_latent_activation_str="_lla"  # linear latent activation
      fi
      if [ "$weight_decay" == "0.0" ] || [ "$weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${weight_decay}"
      fi
      # optional arg `shuffling_buffer_prop`
      if [ -z "$shuffling_buffer_prop" ] || [ "$shuffling_buffer_prop" == "1.0" ] || [ "$shuffling_buffer_prop" == "1" ]; then
        shuffling_buffer_prop_str=""
      else
        shuffling_buffer_prop_str="_${shuffling_buffer_prop}"
      fi
      if [ "$type_" != "rec" ]; then
        rec_str=""
      else
        rec_str="_${rec_unit_type}_${activation_rec}"
      fi
      # dense architecture
      if [ "$type_" == "dense" ] || { [ -z "$enc_n_hidden_neurons" ] && [ "$conv_add_dense_for_latent" == "True" ]; } || { [ -n "$enc_n_hidden_neurons" ] && [ "$rec_latent_type" == "dense" ]; }; then
        arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $enc_n_hidden_neurons $latent_dim)  #! no quotes
      else
        # latent dimension is not relevant
        arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $enc_n_hidden_neurons)  #! no quotes
        if [ -z "$enc_n_hidden_neurons" ] && [ "$conv_add_dense_for_latent" == "False" ]; then
          arch="${arch}_cv"  # convolutional latent
        fi
        if [ -n "$enc_n_hidden_neurons" ] && [ "$rec_latent_type" == "rec" ]; then
          arch="${arch}_rl"  # recurrent latent
        fi
      fi
      id_str="${type_}_ae_${arch}${conv1d_pooling_str}${conv1d_batch_norm_str}${rec_str}${input_dropout_str}"
      id_str="${id_str}${hidden_dropout_str}${linear_latent_activation_str}_${learning_rate}${weight_decay_str}_${n_epochs}"
      id_str="${id_str}_${early_stopping_target}_${early_stopping_patience}${shuffling_buffer_prop_str}"
      echo "$id_str"
    fi
    if [ "$detector" == "vae" ]; then
      type_=$2
      enc_conv1d_filters=$3
      enc_conv1d_kernel_sizes=$4
      enc_conv1d_strides=$5
      conv1d_pooling=$6
      conv1d_batch_norm=$7
      enc_n_hidden_neurons=$8
      dec_n_hidden_neurons=$9
      dec_conv1d_filters=${10}
      dec_conv1d_kernel_sizes=${11}
      dec_conv1d_strides=${12}
      latent_dim=${13}
      activation_rec=${14}
      rec_unit_type=${15}
      dec_output_dist=${16}
      input_dropout=${17}
      hidden_dropout=${18}
      kl_weight=${19}
      optimizer=${20}
      learning_rate=${21}
      weight_decay=${22}
      softplus_scale=${23}
      n_epochs=${24}
      early_stopping_target=${25}
      early_stopping_patience=${26}
      shuffling_buffer_prop=${27}
      if [ "$conv1d_pooling" == "False" ]; then
        conv1d_pooling_str=""
      else
        conv1d_pooling_str="_p"
      fi
      if [ "$conv1d_batch_norm" == "False" ]; then
        conv1d_batch_norm_str=""
      else
        conv1d_batch_norm_str="_bn"
      fi
      if [ "$input_dropout" == "0.0" ] || [ "$input_dropout" == "0" ]; then
        input_dropout_str=""
      else
        input_dropout_str="_${input_dropout}"
      fi
      if [ "$hidden_dropout" == "0.0" ] || [ "$hidden_dropout" == "0" ]; then
        hidden_dropout_str=""
      else
        hidden_dropout_str="_${hidden_dropout}"
      fi
      if [ "$optimizer" != "adamw" ] || [ "$weight_decay" == "0.0" ] || [ "$weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${weight_decay}"
      fi
      if [ "$softplus_scale" == "1.0" ] || [ "$softplus_scale" == "1" ]; then
        softplus_scale_str=""
      else
        softplus_scale_str="_${softplus_scale}"
      fi
      # optional arg `shuffling_buffer_prop`
      if [ -z "$shuffling_buffer_prop" ] || [ "$shuffling_buffer_prop" == "1.0" ] || [ "$shuffling_buffer_prop" == "1" ]; then
        shuffling_buffer_prop_str=""
      else
        shuffling_buffer_prop_str="_${shuffling_buffer_prop}"
      fi
      if [ "$type_" != "rec" ]; then
        rec_str=""
      else
        rec_str="_${rec_unit_type}_${activation_rec}"
      fi
      arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $enc_n_hidden_neurons $latent_dim $dec_n_hidden_neurons $dec_conv1d_filters $dec_conv1d_kernel_sizes $dec_conv1d_strides)  #! no quotes
      id_str="${type_}_vae_${arch}${conv1d_pooling_str}${conv1d_batch_norm_str}${rec_str}_${dec_output_dist}${input_dropout_str}"
      id_str="${id_str}${hidden_dropout_str}_${kl_weight}_${optimizer}_${learning_rate}${weight_decay_str}${softplus_scale_str}"
      id_str="${id_str}_${n_epochs}_${early_stopping_target}_${early_stopping_patience}${shuffling_buffer_prop_str}"
      echo "$id_str"
    fi
    if [ "$detector" == "divad" ]; then
      type_=$2
      pzy_dist=$3
      pzy_kl_n_samples=$4
      pzy_gm_n_components=$5
      pzy_gm_softplus_scale=$6
      pzy_vamp_n_components=$7
      qz_x_conv1d_filters=$8
      qz_x_conv1d_kernel_sizes=$9
      qz_x_conv1d_strides=${10}
      qz_x_n_hidden=${11}
      pzd_d_n_hidden=${12}
      px_z_conv1d_filters=${13}
      px_z_conv1d_kernel_sizes=${14}
      px_z_conv1d_strides=${15}
      px_z_n_hidden=${16}
      time_freq=${17}
      sample_normalize_x=${18}
      sample_normalize_mag=${19}
      apply_hann=${20}
      n_freq_modes=${21}
      phase_encoding=${22}
      phase_cyclical_decoding=${23}
      qz_x_freq_conv1d_filters=${24}
      qz_x_freq_conv1d_kernel_sizes=${25}
      qz_x_freq_conv1d_strides=${26}
      px_z_freq_conv1d_filters=${27}
      px_z_freq_conv1d_kernel_sizes=${28}
      px_z_freq_conv1d_strides=${29}
      latent_dim=${30}
      rec_unit_type=${31}
      activation_rec=${32}
      conv1d_pooling=${33}
      conv1d_batch_norm=${34}
      rec_weight_decay=${35}
      weight_decay=${36}
      dropout=${37}
      dec_output_dist=${38}
      min_beta=${39}
      max_beta=${40}
      beta_n_epochs=${41}
      loss_weighting=${42}
      d_classifier_weight=${43}
      optimizer=${44}
      learning_rate=${45}
      lr_scheduling=${46}  # either "none", "pw_constant" or "one_cycle"
      lrs_pwc_red_factor=${47}
      lrs_pwc_red_freq=${48}
      adamw_weight_decay=${49}
      softplus_scale=${50}
      batch_size=${51}
      n_epochs=${52}
      early_stopping_target=${53}
      early_stopping_patience=${54}
      if [ "$type_" != "rec" ]; then
        rec_str=""
      else
        if [ "$rec_unit_type" == "lstm" ]; then
          rec_unit_type_str=""
        else
          rec_unit_type_str="_${rec_unit_type}"
        fi
        if [ "$activation_rec" == "tanh" ]; then
          activation_rec_str=""
        else
          activation_rec_str="_${activation_rec}"
        fi
        if [ "$conv1d_pooling" == "False" ]; then
          conv1d_pooling_str="_np"  # no pooling
        else
          conv1d_pooling_str=""
        fi
        if [ "$conv1d_batch_norm" == "False" ]; then
          conv1d_batch_norm_str="_nb"  # no batch norm
        else
          conv1d_batch_norm_str=""
        fi
        rec_str="${rec_unit_type_str}${activation_rec_str}${conv1d_pooling_str}${conv1d_batch_norm_str}"
        if [ "$rec_weight_decay" != "0.0" ] && [ "$rec_weight_decay" != "0" ]; then
          rec_str="${rec_str}_${rec_weight_decay}"
        fi
      fi
      if [ "$weight_decay" != "0.0" ] && [ "$weight_decay" != "0" ]; then
        weight_decay_str="_${weight_decay}"
      else
        weight_decay_str=""
      fi
      if [ "$dropout" == "0.5" ]; then
        dropout_str=""
      else
        dropout_str="_${dropout}"
      fi
      if [ "$pzy_dist" != "standard" ]; then
        pzy_dist_str="_${pzy_dist}"
        if [ "$pzy_kl_n_samples" != 1 ]; then
          pzy_kl_n_samples_str="_${pzy_kl_n_samples}"
        else
          pzy_kl_n_samples_str=""
        fi
        if [ "$pzy_dist" == "gm" ]; then
          pzy_gm_str="_${pzy_gm_n_components}"
          if [ "$pzy_gm_softplus_scale" != "1.0" ] && [ "$pzy_gm_softplus_scale" != "1" ]; then
            pzy_gm_str="${pzy_gm_str}_${pzy_gm_softplus_scale}"
          fi
        else
          pzy_gm_str=""
        fi
        if [ "$pzy_dist" == "vamp" ]; then
          pzy_vamp_str="_${pzy_vamp_n_components}"
        else
          pzy_vamp_str=""
        fi
      else
        pzy_dist_str=""
      fi
      if [ "$loss_weighting" == "fixed" ]; then
        loss_weighting_str=""
        d_weight_str="_${d_classifier_weight}"
      else
        loss_weighting_str="_cov"
        d_weight_str=""
      fi
      if [ "$lr_scheduling" == "none" ]; then
        scheduling_str=""
      else
        scheduling_str="_${lrs_pwc_red_factor}_${lrs_pwc_red_freq}"
      fi
      if [ "$optimizer" != "adamw" ] || [ "$adamw_weight_decay" == "0.0" ] || [ "$adamw_weight_decay" == "0" ]; then
        adamw_weight_decay_str=""
      else
        adamw_weight_decay_str="_${adamw_weight_decay}"
      fi
      if [ "$softplus_scale" == "1.0" ] || [ "$softplus_scale" == "1" ]; then
        softplus_scale_str=""
      else
        softplus_scale_str="_${softplus_scale}"
      fi
      if [ "$batch_size" == "32" ]; then
        batch_str=""
      else
        batch_str="_${batch_size}"
      fi
      if [ "$time_freq" == "False" ]; then
        time_freq_str=""
      else
        if [ "$sample_normalize_x" == "True" ]; then
          sample_normalize_x_str=""
        else
          sample_normalize_x_str="_nsn"  # no sample normalization
        fi
        if [ "$sample_normalize_mag" == "False" ]; then
          sample_normalize_mag_str=""
        else
          sample_normalize_mag_str="_mn"  # mag normalization
        fi
        if [ "$apply_hann" == "False" ]; then
          apply_hann_str=""
        else
          apply_hann_str="_nh"  # no hann windowing
        fi
        if [ "$n_freq_modes" == "-1" ]; then
          n_freq_modes_str=""
        else
          n_freq_modes_str="_${n_freq_modes}"
        fi
        if [ "$phase_encoding" == "raw" ]; then
          phase_encoding_str=""
        elif [ "$phase_encoding" == "cyclical" ]; then
          phase_encoding_str="_cp"  # cyclical phase encoding
        else
          phase_encoding_str="_np"  # no phase encoding
        fi
        if [ "$phase_cyclical_decoding" == "False" ]; then
          phase_cyclical_decoding_str=""
        else
          phase_cyclical_decoding_str="_cpd"  # cyclical phase decoding
        fi
        tf_arch=$(join_by "_" $qz_x_freq_conv1d_filters $qz_x_freq_conv1d_kernel_sizes $qz_x_freq_conv1d_strides $px_z_freq_conv1d_filters $px_z_freq_conv1d_kernel_sizes $px_z_freq_conv1d_strides)  #! no quotes
        time_freq_str="_${tf_arch}${sample_normalize_x_str}${sample_normalize_mag_str}${apply_hann_str}${n_freq_modes_str}${phase_encoding_str}${phase_cyclical_decoding_str}"
      fi
      arch=$(join_by "_" $qz_x_conv1d_filters $qz_x_conv1d_kernel_sizes $qz_x_conv1d_strides $qz_x_n_hidden $pzd_d_n_hidden $latent_dim $px_z_conv1d_filters $px_z_conv1d_kernel_sizes $px_z_conv1d_strides $px_z_n_hidden)  #! no quotes
      id_str="${type_}_divad${pzy_dist_str}${pzy_kl_n_samples_str}${pzy_gm_str}${pzy_vamp_str}_${arch}${rec_str}${time_freq_str}"
      id_str="${id_str}${weight_decay_str}${dropout_str}_${dec_output_dist}${input_dropout_str}${hidden_dropout_str}_${min_beta}_${max_beta}"
      id_str="${id_str}_${beta_n_epochs}${loss_weighting_str}${d_weight_str}_${optimizer}_${learning_rate}"
      id_str="${id_str}${scheduling_str}${adamw_weight_decay_str}${softplus_scale_str}${batch_str}_${n_epochs}"
      id_str="${id_str}_${early_stopping_target}_${early_stopping_patience}"
      echo "$id_str"
    fi
    if [ "$detector" == "divadw" ]; then
      y_classifier_mode=$2
      type_=$3
      pzy_y_dist=$4
      pzy_y_kl_n_samples=$5
      pzy_y_gm_n_components=$6
      pzy_y_gm_softplus_scale=$7
      pzy_y_vamp_n_components=$8
      qz_x_conv1d_filters=$9
      qz_x_conv1d_kernel_sizes=${10}
      qz_x_conv1d_strides=${11}
      qz_x_n_hidden=${12}
      pzy_y_n_hidden=${13}
      pzd_d_n_hidden=${14}
      px_z_conv1d_filters=${15}
      px_z_conv1d_kernel_sizes=${16}
      px_z_conv1d_strides=${17}
      px_z_n_hidden=${18}
      time_freq=${19}
      sample_normalize_x=${20}
      sample_normalize_mag=${21}
      apply_hann=${22}
      n_freq_modes=${23}
      phase_encoding=${24}
      phase_cyclical_decoding=${25}
      qz_x_freq_conv1d_filters=${26}
      qz_x_freq_conv1d_kernel_sizes=${27}
      qz_x_freq_conv1d_strides=${28}
      px_z_freq_conv1d_filters=${29}
      px_z_freq_conv1d_kernel_sizes=${30}
      px_z_freq_conv1d_strides=${31}
      latent_dim=${32}
      rec_unit_type=${33}
      activation_rec=${34}
      conv1d_pooling=${35}
      conv1d_batch_norm=${36}
      rec_weight_decay=${37}
      weight_decay=${38}
      dropout=${39}
      dec_output_dist=${40}
      include_zx=${41}
      min_beta=${42}
      max_beta=${43}
      beta_n_epochs=${44}
      loss_weighting=${45}
      y_loss_weight=${46}
      d_classifier_weight=${47}
      optimizer=${48}
      learning_rate=${49}
      lr_scheduling=${50}  # either "none", "pw_constant" or "one_cycle"
      lrs_pwc_red_factor=${51}
      lrs_pwc_red_freq=${52}
      adamw_weight_decay=${53}
      softplus_scale=${54}
      binary_mode=${55}
      class_balancing=${56}
      balanced_n_ano_per_normal=${57}
      only_weigh_y_loss=${58}
      balance_ano_types=${59}
      balance_val=${60}
      batch_size=${61}
      n_epochs=${62}
      epochs_breakdown_prop=${63}
      early_stopping_target=${64}
      early_stopping_patience=${65}
      if [ "$type_" != "rec" ]; then
        rec_str=""
      else
        if [ "$rec_unit_type" == "lstm" ]; then
          rec_unit_type_str=""
        else
          rec_unit_type_str="_${rec_unit_type}"
        fi
        if [ "$activation_rec" == "tanh" ]; then
          activation_rec_str=""
        else
          activation_rec_str="_${activation_rec}"
        fi
        if [ "$conv1d_pooling" == "False" ]; then
          conv1d_pooling_str="_np"  # no pooling
        else
          conv1d_pooling_str=""
        fi
        if [ "$conv1d_batch_norm" == "False" ]; then
          conv1d_batch_norm_str="_nb"  # no batch norm
        else
          conv1d_batch_norm_str=""
        fi
        rec_str="${rec_unit_type_str}${activation_rec_str}${conv1d_pooling_str}${conv1d_batch_norm_str}"
        if [ "$rec_weight_decay" != "0.0" ] && [ "$rec_weight_decay" != "0" ]; then
          rec_str="${rec_str}_${rec_weight_decay}"
        fi
      fi
      if [ "$weight_decay" != "0.0" ] && [ "$weight_decay" != "0" ]; then
        weight_decay_str="_${weight_decay}"
      else
        weight_decay_str=""
      fi
      if [ "$dropout" == "0.5" ]; then
        dropout_str=""
      else
        dropout_str="_${dropout}"
      fi
      if [ "$pzy_y_dist" != "normal" ]; then
        pzy_y_dist_str="_${pzy_y_dist}"
        if [ "$pzy_y_kl_n_samples" != 1 ]; then
          pzy_y_kl_n_samples_str="_${pzy_y_kl_n_samples}"
        else
          pzy_y_kl_n_samples_str=""
        fi
        if [ "$pzy_y_dist" == "gm" ]; then
          pzy_y_gm_str="_${pzy_y_gm_n_components}"
          if [ "$pzy_y_gm_softplus_scale" != "1.0" ] && [ "$pzy_y_gm_softplus_scale" != "1" ]; then
            pzy_y_gm_str="${pzy_y_gm_str}_${pzy_y_gm_softplus_scale}"
          fi
        else
          pzy_y_gm_str=""
        fi
        if [ "$pzy_y_dist" == "vamp" ]; then
          pzy_y_vamp_str="_${pzy_y_vamp_n_components}"
        else
          pzy_y_vamp_str=""
        fi
      else
        pzy_y_dist_str=""
      fi
      if [ "$loss_weighting" == "fixed" ]; then
        loss_weighting_str=""
        weights_str="_${y_loss_weight}_${d_classifier_weight}"
      else
        loss_weighting_str="_cov"
        weights_str=""
      fi
      if [ "$lr_scheduling" == "none" ]; then
        scheduling_str=""
      else
        scheduling_str="_p_${lrs_pwc_red_factor}_${lrs_pwc_red_freq}"
      fi
      if [ "$optimizer" != "adamw" ] || [ "$adamw_weight_decay" == "0.0" ] || [ "$adamw_weight_decay" == "0" ]; then
        adamw_weight_decay_str=""
      else
        adamw_weight_decay_str="_${adamw_weight_decay}"
      fi
      if [ "$include_zx" == "True" ]; then
        include_zx_str=""
      else
        include_zx_str="_nx"
      fi
      if [ "$softplus_scale" == "1.0" ] || [ "$softplus_scale" == "1" ]; then
        softplus_scale_str=""
      else
        softplus_scale_str="_${softplus_scale}"
      fi
      if [ "$binary_mode" == "True" ]; then
        mode_str=""
      else
        mode_str="_m"
      fi
      if [ "$class_balancing" == "none" ]; then
        balancing_str=""
      else
        if [ "$class_balancing" == "samples" ]; then
          balancing_str="_s"
        elif [ "$class_balancing" == "weights" ]; then
          balancing_str="_w"
          if [ "$only_weigh_y_loss" == "True" ]; then
            balancing_str="${balancing_str}_y"
          fi
        fi
        if [ "$balanced_n_ano_per_normal" != "1.0" ] && [ "$balanced_n_ano_per_normal" != "1" ]; then
          balancing_str="${balancing_str}_${balanced_n_ano_per_normal}"
        fi
        if [ "$balance_ano_types" == "False" ]; then
          balancing_str="${balancing_str}_nb"  # no anomaly balancing
        fi
        if [ "$balance_val" == "True" ]; then
          balancing_str="${balancing_str}_v"
        fi
      fi
      if [ "$batch_size" == "32" ]; then
        batch_str=""
      else
        batch_str="_${batch_size}"
      fi
      if [ "$time_freq" == "False" ]; then
        time_freq_str=""
      else
        if [ "$sample_normalize_x" == "True" ]; then
          sample_normalize_x_str=""
        else
          sample_normalize_x_str="_nsn"  # no sample normalization
        fi
        if [ "$sample_normalize_mag" == "False" ]; then
          sample_normalize_mag_str=""
        else
          sample_normalize_mag_str="_mn"  # mag normalization
        fi
        if [ "$apply_hann" == "False" ]; then
          apply_hann_str=""
        else
          apply_hann_str="_nh"  # no hann windowing
        fi
        if [ "$n_freq_modes" == "-1" ]; then
          n_freq_modes_str=""
        else
          n_freq_modes_str="_${n_freq_modes}"
        fi
        if [ "$phase_encoding" == "raw" ]; then
          phase_encoding_str=""
        elif [ "$phase_encoding" == "cyclical" ]; then
          phase_encoding_str="_cp"  # cyclical phase encoding
        else
          phase_encoding_str="_np"  # no phase encoding
        fi
        if [ "$phase_cyclical_decoding" == "False" ]; then
          phase_cyclical_decoding_str=""
        else
          phase_cyclical_decoding_str="_cpd"  # cyclical phase decoding
        fi
        tf_arch=$(join_by "_" $qz_x_freq_conv1d_filters $qz_x_freq_conv1d_kernel_sizes $qz_x_freq_conv1d_strides $px_z_freq_conv1d_filters $px_z_freq_conv1d_kernel_sizes $px_z_freq_conv1d_strides)  #! no quotes
        time_freq_str="_${tf_arch}${sample_normalize_x_str}${sample_normalize_mag_str}${apply_hann_str}${n_freq_modes_str}${phase_encoding_str}${phase_cyclical_decoding_str}"
      fi
      if [ "$epochs_breakdown_prop" == "1.0" ] || [ "$epochs_breakdown_prop" == "1" ]; then
        epoch_breakdown_str=""
      else
        epoch_breakdown_str="_${epochs_breakdown_prop}"
      fi
      if [ "$y_classifier_mode" == "True" ]; then
        detector_str="_diva"
      else
        detector_str="_divadw"
      fi
      arch=$(join_by "_" $qz_x_conv1d_filters $qz_x_conv1d_kernel_sizes $qz_x_conv1d_strides $qz_x_n_hidden $pzy_y_n_hidden $pzd_d_n_hidden $latent_dim $px_z_conv1d_filters $px_z_conv1d_kernel_sizes $px_z_conv1d_strides $px_z_n_hidden)  #! no quotes
      id_str="${type_}${detector_str}${mode_str}${pzy_y_dist_str}${pzy_y_kl_n_samples_str}${pzy_y_gm_str}${pzy_y_vamp_str}"
      id_str="${id_str}${balancing_str}_${arch}${rec_str}${time_freq_str}${weight_decay_str}${dropout_str}_${dec_output_dist}"
      id_str="${id_str}${include_zx_str}${input_dropout_str}${hidden_dropout_str}_${min_beta}_${max_beta}_${beta_n_epochs}"
      id_str="${id_str}${loss_weighting_str}${weights_str}_${optimizer}_${learning_rate}${scheduling_str}"
      id_str="${id_str}${adamw_weight_decay_str}${softplus_scale_str}${batch_str}_${n_epochs}${epoch_breakdown_str}"
      id_str="${id_str}_${early_stopping_target}_${early_stopping_patience}"
      echo "$id_str"
    fi
    if [ "$detector" == "deep_svdd" ]; then
      n_hidden_neurons=$2
      output_dim=$3
      input_dropout=$4
      hidden_dropout=$5
      learning_rate=$6
      weight_decay=$7
      n_epochs=$8
      early_stopping_target=$9
      early_stopping_patience=${10}
      shuffling_buffer_prop=${11}
      if [ "$input_dropout" == "0.0" ] || [ "$input_dropout" == "0" ]; then
        input_dropout_str=""
      else
        input_dropout_str="_in_dropout\=${input_dropout}"
      fi
      if [ "$hidden_dropout" == "0.0" ] || [ "$hidden_dropout" == "0" ]; then
        hidden_dropout_str=""
      else
        hidden_dropout_str="_dropout\=${hidden_dropout}"
      fi
      if [ "$weight_decay" == "0.0" ] || [ "$weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_wd\=${weight_decay}"
      fi
      # optional arg `shuffling_buffer_prop`
      if [ -z "$shuffling_buffer_prop" ] || [ "$shuffling_buffer_prop" == "1.0" ] || [ "$shuffling_buffer_prop" == "1" ]; then
        shuffling_buffer_prop_str=""
      else
        shuffling_buffer_prop_str="_shuffle\=${shuffling_buffer_prop}"
      fi
      arch=$(join_by "_" $n_hidden_neurons $output_dim)  #! no quotes
      id_str="dense_deep_svdd_${arch}${input_dropout_str}${hidden_dropout_str}_lr\=${learning_rate}"
      id_str="${id_str}${weight_decay_str}_epochs\=${n_epochs}_${early_stopping_target}_${early_stopping_patience}${shuffling_buffer_prop_str}"
      echo "$id_str"
    fi
    if [ "$detector" == "deep_sad" ]; then
      normal_as_unlabeled=$2
      remove_anomalies=$3
      oversample_anomalies=$4
      n_ano_per_normal=$5
      network=$6
      enc_conv1d_filters=$7
      enc_conv1d_kernel_sizes=$8
      enc_conv1d_strides=$9
      conv1d_batch_norm=${10}
      hidden_dims=${11}
      rep_dim=${12}
      eta=${13}
      ae_out_act=${14}
      pretrain_optimizer=${15}
      pretrain_adamw_weight_decay=${16}
      pretrain_learning_rate=${17}
      pretrain_lr_milestones=${18}
      pretrain_batch_size=${19}
      pretrain_n_epochs=${20}
      pretrain_early_stopping_target=${21}
      pretrain_early_stopping_patience=${22}
      optimizer=${23}
      adamw_weight_decay=${24}
      learning_rate=${25}
      lr_milestones=${26}
      batch_size=${27}
      n_epochs=${28}
      early_stopping_target=${29}
      early_stopping_patience=${30}
      fix_weights_init=${31}
      if [ "$normal_as_unlabeled" == "True" ]; then
        normal_as_unlabeled_str=""
      else
        normal_as_unlabeled_str="_ln"  # labeled normal
      fi
      if [ "$eta" == "1.0" ]; then
        eta_str=""
      else
        eta_str="_${eta}"
      fi
      if [ "$pretrain_adamw_weight_decay" == "0.0" ] || [ "$pretrain_adamw_weight_decay" == "0" ]; then
        pretrain_weight_decay_str=""
      else
        pretrain_weight_decay_str="_${pretrain_adamw_weight_decay}"
      fi
      pretrain_lr_milestones_str=$(join_by "_" $pretrain_lr_milestones)
      pretrain_str="${pretrain_optimizer}${pretrain_weight_decay_str}_${pretrain_learning_rate}_${pretrain_lr_milestones_str}"
      pretrain_str="${pretrain_str}_${pretrain_batch_size}_${pretrain_n_epochs}_${pretrain_early_stopping_target}"
      pretrain_str="${pretrain_str}_${pretrain_early_stopping_patience}"
      if [ "$ae_out_act" == "sigmoid" ]; then
        ae_out_act_str="_sig"
      else
        ae_out_act_str=""
      fi
      if [ "$adamw_weight_decay" == "0.0" ] || [ "$adamw_weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${adamw_weight_decay}"
      fi
      lr_milestones_str=$(join_by "_" $lr_milestones)
      train_str="${optimizer}${weight_decay_str}_${learning_rate}_${lr_milestones_str}"
      train_str="${train_str}_${batch_size}_${n_epochs}_${early_stopping_target}"
      train_str="${train_str}_${early_stopping_patience}"
      arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $hidden_dims $rep_dim)  #! no quotes
      if [ "$conv1d_batch_norm" == "True" ]; then
        conv1d_batch_norm_str=""
      else
        conv1d_batch_norm_str="_nbn"  # no batch norm
      fi
      if [ "$fix_weights_init" == "True" ]; then
        fix_weights_init_str=""
      else
        fix_weights_init_str="_nwif"  # no weights init fix
      fi
      if [ "$remove_anomalies" == "False" ]; then
        detector_str="dsad"
        if [ "$oversample_anomalies" == "True" ]; then
          detector_str="${detector_str}_os_${n_ano_per_normal}"  # "os" for "oversample"
        fi
      else
        detector_str="dsvdd"
      fi
      id_str="${network}_${detector_str}${normal_as_unlabeled_str}_${arch}${conv1d_batch_norm_str}"
      id_str="${id_str}${ae_out_act_str}${eta_str}_${pretrain_str}_${train_str}${fix_weights_init_str}"
      echo "$id_str"
    fi
    if [ "$detector" == "lstm_ad" ]; then
      conv1d_filters=$2
      conv1d_kernel_sizes=$3
      conv1d_strides=$4
      conv1d_pooling=$5
      conv1d_batch_norm=$6
      unit_type=$7
      n_hidden_neurons=$8
      n_forward=$9
      learning_rate=${10}
      weight_decay=${11}
      n_epochs=${12}
      early_stopping_target=${13}
      early_stopping_patience=${14}
      shuffling_buffer_prop=${15}
      if [ -z "$conv1d_strides" ]; then
        conv1d_arch_str=""
      else
        conv1d_arch_str="_$(join_by "_" $conv1d_filters $conv1d_kernel_sizes $conv1d_strides)_"
      fi
      if [ -z "$n_hidden_neurons" ]; then
        rec_arch_str=""
      else
        rec_arch_str="_${unit_type}_$(join_by "_" $n_hidden_neurons)"
      fi
      if [ "$conv1d_pooling" == "False" ]; then
        conv1d_pooling_str=""
      else
        conv1d_pooling_str="_p"
      fi
      if [ "$conv1d_batch_norm" == "False" ]; then
        conv1d_batch_norm_str=""
      else
        conv1d_batch_norm_str="_bn"
      fi
      if [ "$weight_decay" == "0.0" ] || [ "$weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${weight_decay}"
      fi
      # optional arg `shuffling_buffer_prop`
      if [ -z "$shuffling_buffer_prop" ] || [ "$shuffling_buffer_prop" == "1.0" ] || [ "$shuffling_buffer_prop" == "1" ]; then
        shuffling_buffer_prop_str=""
      else
        shuffling_buffer_prop_str="_${shuffling_buffer_prop}"
      fi
      id_str="lstm_ad${conv1d_arch_str}${rec_arch_str}${conv1d_pooling_str}${conv1d_batch_norm_str}_${n_forward}_${learning_rate}"
      id_str="${id_str}${weight_decay_str}_${n_epochs}_${early_stopping_target}_${early_stopping_patience}${shuffling_buffer_prop_str}"
      echo "$id_str"
    fi
    if [ "$detector" == "dc_detector" ]; then
      n_encoder_layers=$2
      n_attention_heads=$3
      patch_sizes=$4
      d_model=$5
      dropout=$6
      optimizer=$7
      adamw_weight_decay=$8
      learning_rate=$9
      batch_size=${10}
      n_epochs=${11}
      early_stopping_target=${12}
      early_stopping_patience=${13}
      if [ "$dropout" == "0.0" ] || [ "$dropout" == "0" ]; then
        dropout_str=""
      else
        dropout_str="_${dropout}"
      fi
      if [ "$adamw_weight_decay" == "0.0" ] || [ "$adamw_weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${weight_decay}"
      fi
      patch_sizes_arch=$(join_by "_" $patch_sizes)  #! no quotes
      id_str="dcd_${n_encoder_layers}_${n_attention_heads}_${patch_sizes_arch}_${d_model}${dropout_str}_${optimizer}"
      id_str="${id_str}${weight_decay_str}_${learning_rate}_${batch_size}_${n_epochs}_${early_stopping_target}_${early_stopping_patience}"
      echo "$id_str"
    fi
    if [ "$detector" == "tranad" ]; then
      dim_feedforward=$2
      last_activation=$3
      optimizer=$4
      adamw_weight_decay=$5
      learning_rate=$6
      batch_size=$7
      n_epochs=$8
      early_stopping_target=$9
      early_stopping_patience=${10}
      if [ "$last_activation" == "sigmoid" ]; then
        last_activation_str=""
      else
        last_activation_str="_${last_activation}"
      fi
      if [ "$adamw_weight_decay" == "0.0" ] || [ "$adamw_weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${adamw_weight_decay}"
      fi
      id_str="tranad_${dim_feedforward}${last_activation_str}_${optimizer}${weight_decay_str}_${learning_rate}_${batch_size}_${n_epochs}"
      id_str="${id_str}_${early_stopping_target}_${early_stopping_patience}"
      echo "$id_str"
    fi
    if [ "$detector" == "mscred" ]; then
      signature_lengths=$2
      filters=$3
      kernel_sizes=$4
      strides=$5
      optimizer=$6
      adamw_weight_decay=$7
      learning_rate=$8
      batch_size=$9
      n_epochs=${10}
      early_stopping_target=${11}
      early_stopping_patience=${12}
      if [ "$adamw_weight_decay" == "0.0" ] || [ "$adamw_weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${adamw_weight_decay}"
      fi
      arch="$(join_by "_" $signature_lengths $filters $kernel_sizes $strides)"  #! no quotes
      id_str="mscred_${arch}_${optimizer}${weight_decay_str}_${learning_rate}_${batch_size}_${n_epochs}"
      id_str="${id_str}_${early_stopping_target}_${early_stopping_patience}"
      echo "$id_str"
    fi
    if [ "$detector" == "contrastive_autoencoder" ]; then
      type_=$2
      pair_mining=$3
      batch_balanced_types=$4
      concordant_batching=$5
      discordant_batching=$6
      version=$7
      contrastive_margin=$8
      enc_conv1d_filters=$9
      enc_conv1d_kernel_sizes=${10}
      enc_conv1d_strides=${11}
      conv1d_pooling=${12}
      conv1d_batch_norm=${13}
      enc_n_hidden_neurons=${14}
      latent_dim=${15}
      rec_unit_type=${16}
      rec_dropout=${17}
      input_dropout=${18}
      hidden_dropout=${19}
      linear_latent_activation=${20}
      rec_latent_type=${21}
      conv_add_dense_for_latent=${22}
      learning_rate=${23}
      lr_scheduling=${24}
      lrs_pwc_red_factor=${25}
      lrs_pwc_red_freq=${26}
      weight_decay=${27}
      batch_size=${28}
      n_epochs=${29}
      early_stopping_target=${30}
      early_stopping_patience=${31}
      shuffling_buffer_prop=${32}
      if [ "$type_" == "rec" ]; then
        rec_str="_${rec_unit_type}"
        if [ "$rec_dropout" != "0.0" ] && [ "$rec_dropout" != "0" ]; then
          rec_str="${rec_str}_${rec_dropout}"
        fi
      else
        rec_str=""
      fi
      if [ "$pair_mining" == "fixed_offline" ]; then
        pair_mining_str=""
        concordant_batching_str=""
        discordant_batching_str=""
        batch_balanced_types_str=""
      else
        pair_mining_str="_${pair_mining}"
        concordant_batching_str="_${concordant_batching}"
        discordant_batching_str="_${discordant_batching}"
        if [ "$batch_balanced_types" == "True" ]; then
          batch_balanced_types_str="_bal"
        else
          batch_balanced_types_str=""
        fi
      fi
      if [ "$conv1d_pooling" == "False" ]; then
        conv1d_pooling_str=""
      else
        conv1d_pooling_str="_p"
      fi
      if [ "$conv1d_batch_norm" == "False" ]; then
        conv1d_batch_norm_str=""
      else
        conv1d_batch_norm_str="_bn"
      fi
      if [ "$contrastive_margin" == "10.0" ] || [ "$contrastive_margin" == "10" ]; then
        contrastive_margin_str=""
      else
        contrastive_margin_str="_${contrastive_margin}"
      fi
      if [ "$input_dropout" == "0.0" ] || [ "$input_dropout" == "0" ]; then
        input_dropout_str=""
      else
        input_dropout_str="_${input_dropout}"
      fi
      if [ "$hidden_dropout" == "0.0" ] || [ "$hidden_dropout" == "0" ]; then
        hidden_dropout_str=""
      else
        hidden_dropout_str="_${hidden_dropout}"
      fi
      if [ "$linear_latent_activation" == "True" ]; then
        linear_latent_activation_str=""
      else
        linear_latent_activation_str="_nla"  # non-linear latent activation
      fi
      if [ "$lr_scheduling" == "none" ]; then
        scheduling_str=""
      else
        scheduling_str="_p_${lrs_pwc_red_factor}_${lrs_pwc_red_freq}"
      fi
      if [ "$weight_decay" == "0.0" ] || [ "$weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${weight_decay}"
      fi
      if [ "$batch_size" == "32" ]; then
        batch_size_str=""
      else
        batch_size_str="_${batch_size}"
      fi
      # optional arg `shuffling_buffer_prop`
      if [ -z "$shuffling_buffer_prop" ] || [ "$shuffling_buffer_prop" == "1.0" ] || [ "$shuffling_buffer_prop" == "1" ]; then
        shuffling_buffer_prop_str=""
      else
        shuffling_buffer_prop_str="_${shuffling_buffer_prop}"
      fi
      # dense architecture
      if [ "$type_" == "dense" ] || { [ -z "$enc_n_hidden_neurons" ] && [ "$conv_add_dense_for_latent" == "True" ]; } || { [ -n "$enc_n_hidden_neurons" ] && [ "$rec_latent_type" == "dense" ]; }; then
        arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $enc_n_hidden_neurons $latent_dim)  #! no quotes
      else
        # latent dimension is not relevant
        arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $enc_n_hidden_neurons)  #! no quotes
        if [ -z "$enc_n_hidden_neurons" ] && [ "$conv_add_dense_for_latent" == "False" ]; then
          arch="${arch}_cv"  # convolutional latent
        fi
        if [ -n "$enc_n_hidden_neurons" ] && [ "$rec_latent_type" == "rec" ]; then
          arch="${arch}_rl"  # recurrent latent
        fi
      fi
      id_str="${type_}_ce${pair_mining_str}${batch_balanced_types_str}${concordant_batching_str}${discordant_batching_str}_${version}"
      id_str="${id_str}${contrastive_margin_str}_${arch}${conv1d_pooling_str}${conv1d_batch_norm_str}${rec_str}${input_dropout_str}"
      id_str="${id_str}${hidden_dropout_str}_${learning_rate}${scheduling_str}${weight_decay_str}${batch_size_str}_${n_epochs}"
      id_str="${id_str}_${early_stopping_target}_${early_stopping_patience}${shuffling_buffer_prop_str}"
      echo "$id_str"
    fi
    if [ "$detector" == "triplet_encoder" ]; then
      type_=$2
      batch_balanced_types=$3
      positive_mining=$4
      negative_mining=$5
      triplet_margin=$6
      enc_conv1d_filters=$7
      enc_conv1d_kernel_sizes=$8
      enc_conv1d_strides=$9
      conv1d_pooling=${10}
      conv1d_batch_norm=${11}
      enc_n_hidden_neurons=${12}
      latent_dim=${13}
      rec_unit_type=${14}
      rec_dropout=${15}
      input_dropout=${16}
      hidden_dropout=${17}
      learning_rate=${18}
      lr_scheduling=${19}
      lrs_pwc_red_factor=${20}
      lrs_pwc_red_freq=${21}
      weight_decay=${22}
      batch_size=${23}
      n_epochs=${24}
      early_stopping_target=${25}
      early_stopping_patience=${26}
      shuffling_buffer_prop=${27}
      if [ "$type_" == "rec" ]; then
        type_str="rec_"
        rec_str="_${rec_unit_type}"
        if [ "$rec_dropout" != "0.0" ] && [ "$rec_dropout" != "0" ]; then
          rec_str="${rec_str}_${rec_dropout}"
        fi
      else
        type_str=""
        rec_str=""
      fi
      if [ "$batch_balanced_types" == "True" ]; then
        batch_balanced_types_str="_bal"
      else
        batch_balanced_types_str=""
      fi
      if [ "$conv1d_pooling" == "False" ]; then
        conv1d_pooling_str=""
      else
        conv1d_pooling_str="_p"
      fi
      if [ "$conv1d_batch_norm" == "False" ]; then
        conv1d_batch_norm_str=""
      else
        conv1d_batch_norm_str="_bn"
      fi
      if [ "$input_dropout" == "0.0" ] || [ "$input_dropout" == "0" ]; then
        input_dropout_str=""
      else
        input_dropout_str="_${input_dropout}"
      fi
      if [ "$hidden_dropout" == "0.0" ] || [ "$hidden_dropout" == "0" ]; then
        hidden_dropout_str=""
      else
        hidden_dropout_str="_${hidden_dropout}"
      fi
      if [ "$lr_scheduling" == "none" ]; then
        scheduling_str=""
      else
        scheduling_str="_p_${lrs_pwc_red_factor}_${lrs_pwc_red_freq}"
      fi
      if [ "$weight_decay" == "0.0" ] || [ "$weight_decay" == "0" ]; then
        weight_decay_str=""
      else
        weight_decay_str="_${weight_decay}"
      fi
      # optional arg `shuffling_buffer_prop`
      if [ -z "$shuffling_buffer_prop" ] || [ "$shuffling_buffer_prop" == "1.0" ] || [ "$shuffling_buffer_prop" == "1" ]; then
        shuffling_buffer_prop_str=""
      else
        shuffling_buffer_prop_str="_shuffle\=${shuffling_buffer_prop}"
      fi
      arch=$(join_by "_" $enc_conv1d_filters $enc_conv1d_kernel_sizes $enc_conv1d_strides $enc_n_hidden_neurons $latent_dim)  #! no quotes
      id_str="${type_str}te${batch_balanced_types_str}_${positive_mining}_${negative_mining}_${triplet_margin}_${arch}${conv1d_pooling_str}"
      id_str="${id_str}${conv1d_batch_norm_str}${rec_str}${input_dropout_str}${hidden_dropout_str}_${learning_rate}${scheduling_str}"
      id_str="${id_str}${weight_decay_str}_${batch_size}_${n_epochs}_${early_stopping_target}_${early_stopping_patience}"
      id_str="${id_str}${shuffling_buffer_prop_str}"
      echo "$id_str"
    fi

}

function get_train_window_scorer_id() {
    detector=$1
    if [ "$detector" == "pca" ]; then
      method=$2
      n_selected_components=$3
      echo "${method}_${n_selected_components}"
    fi
    if [ "$detector" == "iforest" ]; then
      drop_anomalies=$2
      n_estimators=$3
      max_samples=$4
      max_features=$5
      if [ "$drop_anomalies" == "True" ]; then
        drop_anomalies_str="_drop_ano"
      else
        drop_anomalies_str=""
      fi
      echo "iforest${drop_anomalies_str}_${n_estimators}_${max_samples}_${max_features}"
    fi
    if [ "$detector" == "xgboost_" ]; then
      binary_mode=$2
      imbalance_handling=$3
      n_estimators=$4
      max_depth=$5
      min_child_weight=$6
      subsample=$7
      learning_rate=$8
      gamma=$9
      max_delta_step=${10}
      if [ "$binary_mode" == "True" ]; then
        binary_mode_str=""
      else
        binary_mode_str="_multi"
      fi
      if [ "$imbalance_handling" == "none" ]; then
        imbalance_handling_str=""
      else
        imbalance_handling_str="_bal"
      fi
      id_str="xgb${binary_mode_str}${imbalance_handling_str}_${n_estimators}_${max_depth}_${min_child_weight}"
      id_str="${id_str}_${subsample}_${learning_rate}_${gamma}_${max_delta_step}"
      echo "$id_str"
    fi
    if [ "$detector" == "vae" ]; then
      reco_prob_n_samples=$2
      echo "${reco_prob_n_samples}_samples"
    fi
    if [ "$detector" == "dc_detector" ]; then
      temperature=$2
      id_str="tmp\=${temperature}"
      echo "$id_str"
    fi
    if [ "$detector" == "divad" ] || [ "$detector" == "divadw" ]; then
      scoring_method=$2
      agg_post_dist=$3
      agg_post_gm_n_components=$4
      mean_nll_n_samples=$5
      if [[ $scoring_method == *mean_nll* ]]; then
        mean_nll_n_samples_str="_${mean_nll_n_samples}"
      else
        mean_nll_n_samples_str=""
      fi
      if [[ $scoring_method == *agg_post* ]]; then
        agg_post_str="_${agg_post_dist}"
        if [ "$agg_post_dist" == "gm" ]; then
          agg_post_str="_${agg_post_str}_${agg_post_gm_n_components}"
        fi
      else
        agg_post_str=""
      fi
      echo "${scoring_method}${agg_post_str}${mean_nll_n_samples_str}"
    fi
}

function get_train_online_scorer_id(){
    scores_avg_beta=$1
    echo "b${scores_avg_beta}"
}

function get_evaluate_online_scorer_id(){
    ignored_anomaly_labels=$1
    ignored_delayed_window=$2
    if [ "$ignored_anomaly_labels" == "" ]; then
      ignored_anomaly_labels_str=""
    else
      ignored_anomaly_labels_str=_l$(join_by "-" $ignored_anomaly_labels)  #! no quotes
    fi
    if [ "$ignored_delayed_window" == "0" ]; then
      ignored_delayed_window_str=""
    else
      ignored_delayed_window_str="_w${ignored_delayed_window}"
    fi
    echo "t_pt${ignored_anomaly_labels_str}${ignored_delayed_window_str}"
}