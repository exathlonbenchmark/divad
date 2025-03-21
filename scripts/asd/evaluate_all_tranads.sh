#!/bin/bash

for test_entity in "$@"; do
  ./evaluate_tranad.sh "" "$test_entity" 0.15 regular_scaling minmax 20 1 1 1 mean 64 sigmoid adamw "1e-5" "1e-3" 128 300 val_loss 100
done