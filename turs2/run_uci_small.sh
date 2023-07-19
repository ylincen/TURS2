#!/bin/bash

# Define array of data names for the first python script

data_names1=("chess" "iris" "backnote" "contracept" "ionosphere" "car" "tic-tac-toe" "wine" "diabetes")
data_names1=("iris" "wine")

# Run the script for each data name in the first list
for data_name in "${data_names1[@]}"; do
    python3 exp_predictive_perf.py "$data_name" &
done

wait
