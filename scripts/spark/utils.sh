#!/bin/bash

source ../utils.sh

function get_make_datasets_id(){
    setup=$1
    generalization_test_prop=$2
    app_ids=$3
    label_as_unknown=$4
    trace_removal_idx=$5
    data_pruning_idx=$6
    val_prop=$7
    train_val_split=$8
    if [ "$setup" == "generalization" ]; then
      setup_str="g"
      generalization_test_prop_str="_${generalization_test_prop}"
    else
      generalization_test_prop_str=""
      if [ "$setup" == "weakly" ]; then
        setup_str="w"
      fi
      if [ "$setup" == "unsupervised" ]; then
        setup_str="u"
      fi
    fi
    apps_str=$(join_by "-" $app_ids)  #! no quotes
    #! escaping "=" with backslash is important
    if [ "$label_as_unknown" == "os_only" ]; then
      label_as_unknown_str=""
    elif [ "$label_as_unknown" == "2.3.4.5.6" ]; then
      label_as_unknown_str="_ab1"  # all but 1
    else
      label_as_unknown_str="_${label_as_unknown}"
    fi
    if [ "$trace_removal_idx" == "0" ]; then
      trace_removal_idx_str=""
    else
      trace_removal_idx_str="_${trace_removal_idx}"
    fi
    if [ "$data_pruning_idx" == "0" ]; then
      data_pruning_idx_str=""
    else
      data_pruning_idx_str="_${data_pruning_idx}"
    fi
    if [ "$val_prop" == "0.0" ] || [ "$val_prop" == "0" ]; then
      val_prop_str=""
      train_val_split_str=""
    else
      val_prop_str="_${val_prop}"
      if [ "$setup" != "generalization" ]; then
        if [ "$train_val_split" == "random" ]; then
          train_val_split_str="_r"
        elif [ "$train_val_split" == "time" ]; then
          train_val_split_str="_t"
        fi
      else
        train_val_split_str=""  # always "time"
      fi
    fi
    id_str="${setup_str}${generalization_test_prop_str}_${apps_str}${label_as_unknown_str}${trace_removal_idx_str}"
    id_str="${id_str}${data_pruning_idx_str}${val_prop_str}${train_val_split_str}"
    echo "$id_str"
}

function get_build_features_id(){
    bundle_idx=$1
    transform_chain=$2
    regular_scaling_method=$3
    transform_fit_normal_only=$4
    feature_set="error"
    if [ "$bundle_idx" == 1 ]; then
      feature_set="c"
    elif [ "$bundle_idx" == 5 ]; then
      feature_set="r"
    elif [ "$bundle_idx" == 6 ]; then
      feature_set="nc"
    elif [ "$bundle_idx" == 7 ]; then
      feature_set="rnos"
    elif [ "$bundle_idx" == 8 ]; then
      feature_set="cnos"
    elif [ "$bundle_idx" == 9 ]; then
      feature_set="lbr"
    elif [ "$bundle_idx" == 10 ]; then
      feature_set="dg4"  # 4-feature set reflecting the dg challenge
    elif [ "$bundle_idx" == 11 ]; then
      feature_set="bmm"  # block manager memory
    fi
    head_suffix=""
    if [ "$transform_chain" == "." ]; then
      transform_chain_str=""
    elif [ "$transform_chain" == "regular_scaling" ]; then
      transform_chain_str="_s"
    elif [ "$transform_chain" == "regular_scaling.oh_app_ext" ]; then
      transform_chain_str="_s_oha"
    else
      transform_chain_str="_${transform_chain}"
    fi
    if [[ $transform_chain == *head* ]]; then
      head_suffix="_10"
    fi
    if [[ $transform_chain == *regular_scaling* ]] && [ "$regular_scaling_method" != "std" ]; then
      if [ "$regular_scaling_method" == "minmax" ]; then
        reg_scaling_method_str="_mm"
      else
        reg_scaling_method_str="_$regular_scaling_method"
      fi
    else
      reg_scaling_method_str=""
    fi
    if [ -z "$transform_fit_normal_only" ] || [ "$transform_fit_normal_only" == "yes" ]; then
      transform_fit_normal_only_str=""
    else
      transform_fit_normal_only_str="_fa"  # "fit all"
    fi
    echo "${feature_set}${transform_chain_str}${reg_scaling_method_str}${head_suffix}${transform_fit_normal_only_str}"
}
