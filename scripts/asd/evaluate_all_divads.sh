#!/bin/bash

for test_entity in "$@"; do
  ./evaluate_divad.sh "" "$test_entity" 0.15 regular_scaling std 20 1 1 1 mean none rec gm 1 8 \
  1.0 -1 "32" "5" "1" "32" "32" "19" "5" "1" "32" False False False False -1 -1 False \
  "" "" "" "" "" "" 16 gru tanh False False 0.0 0.0 0.0 normal 5.0 5.0 100 fixed 1000.0 \
  adamw "1e-5" none -1 -1 0.01 1.0 128 300 val_loss 100 prior_nll_of_mean -1 -1 -1 -1
  ./evaluate_divad.sh "" "$test_entity" 0.15 regular_scaling std 20 1 1 1 mean none rec gm 1 8 \
  1.0 -1 "32" "5" "1" "32" "32" "19" "5" "1" "32" False False False False -1 -1 False \
  "" "" "" "" "" "" 16 gru tanh False False 0.0 0.0 0.0 normal 5.0 5.0 100 fixed 1000.0 \
  adamw "1e-5" none -1 -1 0.01 1.0 128 300 val_loss 100 agg_post_nll_of_mean gm 8 -1 -1
done