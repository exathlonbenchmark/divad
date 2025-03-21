#!/bin/bash

source ../utils.sh

function get_make_datasets_id(){
    app_ids=$1
    test_app_ids=$2
    val_prop=$3
    if [ -z "$app_ids" ] ; then
      apps_str="all"
    else
      apps_str=$(join_by "-" $app_ids)  #! no quotes
    fi
    test_apps_str=$(join_by "-" $test_app_ids)  #! no quotes
    if [ "$val_prop" == "0.0" ] || [ "$val_prop" == "0" ]; then
      val_prop_str=""
    else
      val_prop_str="_${val_prop}"
    fi
    id_str="${apps_str}_${test_apps_str}${val_prop_str}"
    echo "$id_str"
}

function get_build_features_id(){
    transform_chain=$1
    regular_scaling_method=$2
    transform_fit_normal_only=$3
    if [ "$transform_chain" == "." ]; then
      transform_chain_str=""
    elif [ "$transform_chain" == "regular_scaling" ]; then
      transform_chain_str="s"
    else
      transform_chain_str="${transform_chain}"
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
    echo "${transform_chain_str}${reg_scaling_method_str}${transform_fit_normal_only_str}"
}
